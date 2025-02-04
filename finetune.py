"""
This function jointly finetunes the policy network and high resolution classifier
using only high resolution classifier. You should load the pre-trained model
as described in the paper.
How to run on different benchmarks:
    python finetune.py --model R32_C10, R32_C100, R34_fMoW, R50_ImgNet
       --lr 1e-4
       --cv_dir checkpoint directory
       --batch_size 1048
       --ckpt_hr_cl Load the checkpoint from the directory (hr_classifier)
"""
import os
import argparse
import numpy as np
import tqdm
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state, orbax
from flax.training.common_utils import shard
from flax.metrics import tensorboard
from flax.jax_utils import replicate, unreplicate
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
import utils
from utils import utils

parser = argparse.ArgumentParser(description='Policy Network Finetuning-I')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--model', default='R32_C10', help='R<depth>_<dataset> see utils.py for a list of configurations')
parser.add_argument('--ckpt_hr_cl', help='checkpoint directory for the high resolution classifier')
parser.add_argument('--data_dir', default='data/', help='data directory')
parser.add_argument('--load', default=None, help='checkpoint to load agent from')
parser.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--max_epochs', type=int, default=10000, help='total epochs to run')
parser.add_argument('--penalty', type=float, default=-10, help='to penalize the PN for incorrect predictions')
parser.add_argument('--alpha', type=float, default=0.8, help='probability bounding factor')
parser.add_argument('--lr_size', type=int, default=8, help='Policy Network Image Size')
parser.add_argument('--test_interval', type=int, default=5, help='At what epoch to test the model')
args = parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.makedirs(args.cv_dir)
utils.save_args(__file__, args)

class TrainState(train_state.TrainState):
    pass

def create_train_state(rng, model, learning_rate):
    params = model.init(rng, jnp.ones([1, 3, args.lr_size, args.lr_size]))['params']
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def train_step(state, batch):
    def loss_fn(params):
        inputs, targets = batch
        inputs_agent = jax.image.resize(inputs, (inputs.shape[0], 3, args.lr_size, args.lr_size), method='bilinear')
        probs = nn.sigmoid(state.apply_fn({'params': params}, inputs_agent, args.model.split('_')[1], 'lr'))
        probs = probs * args.alpha + (1 - probs) * (1 - args.alpha)
        distr = jax.random.bernoulli(jax.random.PRNGKey(0), probs)
        policy_sample = distr.astype(jnp.float32)
        policy_map = (probs >= 0.5).astype(jnp.float32)
        inputs_map = utils.agent_chosen_input(inputs, policy_map, mappings, patch_size)
        inputs_sample = utils.agent_chosen_input(inputs, policy_sample, mappings, patch_size)
        preds_map = rnet.apply({'params': rnet_params}, inputs_map, args.model.split('_')[1], 'hr')
        preds_sample = rnet.apply({'params': rnet_params}, inputs_sample, args.model.split('_')[1], 'hr')
        reward_map, match = utils.compute_reward(preds_map, targets, policy_map, args.penalty)
        reward_sample, _ = utils.compute_reward(preds_sample, targets, policy_sample, args.penalty)
        advantage = reward_sample - reward_map
        loss = -jnp.sum(distr * advantage, axis=1).mean()
        loss += optax.softmax_cross_entropy(preds_sample, targets).mean()
        return loss, (match, reward_sample, reward_map, policy_sample)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (match, reward_sample, reward_map, policy_sample)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, match, reward_sample, reward_map, policy_sample

def train_epoch(state, trainloader):
    matches, rewards, rewards_baseline, policies = [], [], [], []
    for batch in tqdm.tqdm(trainloader):
        state, loss, match, reward_sample, reward_map, policy_sample = train_step(state, batch)
        matches.append(match)
        rewards.append(reward_sample)
        rewards_baseline.append(reward_map)
        policies.append(policy_sample)
    accuracy, reward, sparsity, variance, policy_set = utils.performance_stats(policies, rewards, matches)
    return state, accuracy, reward, sparsity, variance, policy_set

def test_step(state, batch):
    inputs, targets = batch
    inputs_agent = jax.image.resize(inputs, (inputs.shape[0], 3, args.lr_size, args.lr_size), method='bilinear')
    probs = nn.sigmoid(state.apply_fn({'params': state.params}, inputs_agent, args.model.split('_')[1], 'lr'))
    policy = (probs >= 0.5).astype(jnp.float32)
    inputs = utils.agent_chosen_input(inputs, policy, mappings, patch_size)
    preds = rnet.apply({'params': rnet_params}, inputs, args.model.split('_')[1], 'hr')
    reward, match = utils.compute_reward(preds, targets, policy, args.penalty)
    return match, reward, policy

def test_epoch(state, testloader):
    matches, rewards, policies = [], [], []
    for batch in tqdm.tqdm(testloader):
        match, reward, policy = test_step(state, batch)
        matches.append(match)
        rewards.append(reward)
        policies.append(policy)
    accuracy, reward, sparsity, variance, policy_set = utils.performance_stats(policies, rewards, matches)
    return accuracy, reward, sparsity, variance, policy_set

# Load datasets and models
trainset, testset = utils.get_dataset(args.model, args.data_dir)
trainloader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
testloader = torchdata.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)
rnet, _, agent = utils.get_model(args.model)

# Initialize models and optimizer
rng = jax.random.PRNGKey(0)
state = create_train_state(rng, agent, args.lr)
rnet_params = rnet.init(rng, jnp.ones([1, 3, 224, 224]))['params']

# Load checkpoints if available
if args.load is not None:
    state = restore_checkpoint(args.load, state)
    print('loaded pretrained model from', args.load)

if args.ckpt_hr_cl is not None:
    rnet_params = restore_checkpoint(args.ckpt_hr_cl, rnet_params)
    print('loaded the high resolution classifier')

# Action Space for the Policy Network
mappings, _, patch_size = utils.action_space_model(args.model.split('_')[1])

# Train and test the model
for epoch in range(args.max_epochs):
    state, train_accuracy, train_reward, train_sparsity, train_variance, train_policy_set = train_epoch(state, trainloader)
    print(f'Train: {epoch} | Acc: {train_accuracy:.3f} | Rw: {train_reward:.2E} | S: {train_sparsity:.3f} | V: {train_variance:.3f} | #: {len(train_policy_set)}')
    if epoch % args.test_interval == 0:
        test_accuracy, test_reward, test_sparsity, test_variance, test_policy_set = test_epoch(state, testloader)
        print(f'Test - Acc: {test_accuracy:.3f} | Rw: {test_reward:.2E} | S: {test_sparsity:.3f} | V: {test_variance:.3f} | #: {len(test_policy_set)}')
        save_checkpoint(args.cv_dir, state, epoch, prefix='ckpt_', keep=3)