"""
This function pretrains the high and low resolution classifiers.
How to run on different benchmarks:
    python classifer_training.py --model R32_C10, R32_C100, R34_fMoW, R50_ImgNet
       --lr 1e-1 (Different learning rates should be used for different benchmarks)
       --cv_dir checkpoint directory
       --batch_size 128
       --img_size 32, 224, 8, 56
"""
import os
from tensorboard_logger import configure, log_value
import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.training.train_state as train_state
import tqdm
import optax
import orbax.checkpoint

from utils import utils

import argparse
parser = argparse.ArgumentParser(description='Classifier Training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--beta', type=float, default=1e-1, help='entropy multiplier')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--model', default='R32_C10', help='R<depth>_<dataset> see utils.py for a list of configurations')
parser.add_argument('--data_dir', default='data/', help='data directory')
parser.add_argument('--load', help='checkpoint directory for trained model')
parser.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epoch_step', type=int, default=10000, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=300, help='maximum number of epochs')
parser.add_argument('--parallel', action ='store_true', default=False, help='use multiple GPUs for training')
parser.add_argument('--test_interval', type=int, default=5, help='At what epoch to test the model')
parser.add_argument('--img_size', type=int, default=32, help='image size for the classification network')
parser.add_argument('--mode', default='hr', help='Type of the classifier - LR or HR')
args = parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
utils.save_args(__file__, args)

def create_train_state(rng, model, learning_rate, weight_decay):
    params = model.init(rng, jnp.ones([1, args.img_size, args.img_size, 3]))['params']
    tx = optax.adamw(learning_rate, weight_decay=weight_decay)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def train(state, trainloader, epoch):
    def loss_fn(params, batch):
        inputs, targets = batch
        preds = state.apply_fn({'params': params}, inputs, args.model.split("_")[1], args.mode)
        loss = optax.softmax_cross_entropy(preds, jax.nn.one_hot(targets, preds.shape[-1])).mean()
        return loss, preds

    matches, losses = [], []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
        inputs, targets = jnp.array(inputs), jnp.array(targets)
        inputs = jax.image.resize(inputs, (inputs.shape[0], args.img_size, args.img_size, 3), method='bilinear')

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, preds), grads = grad_fn(state.params, (inputs, targets))
        state = state.apply_gradients(grads=grads)

        match = (jnp.argmax(preds, axis=1) == targets)
        matches.append(match)
        losses.append(loss)

    accuracy = jnp.concatenate(matches).mean()
    loss = jnp.stack(losses).mean()

    log_str = 'E: %d | A: %.3f | L: %.3f' % (epoch, accuracy, loss)
    print(log_str)
    log_value('train_accuracy', accuracy, epoch)
    log_value('train_loss', loss, epoch)
    return state

def test(state, testloader, epoch):
    matches = []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
        inputs, targets = jnp.array(inputs), jnp.array(targets)
        inputs = jax.image.resize(inputs, (inputs.shape[0], args.img_size, args.img_size, 3), method='bilinear')

        preds = state.apply_fn({'params': state.params}, inputs, args.model.split("_")[1], args.mode)
        match = (jnp.argmax(preds, axis=1) == targets)
        matches.append(match)

    accuracy = jnp.concatenate(matches).mean()
    log_str = 'TS: %d | A: %.3f' % (epoch, accuracy)
    print(log_str)
    log_value('test_accuracy', accuracy, epoch)

    orbax.checkpoint.save_checkpoint(args.cv_dir, {'params': state.params}, step=epoch)

trainset, testset = utils.get_dataset(args.model, args.data_dir)
trainloader = orbax.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
testloader = orbax.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)

rnet, _, _ = utils.get_model(args.model)
rng = jax.random.PRNGKey(0)
state = create_train_state(rng, rnet, args.lr, args.wd)

if args.load:
    state = orbax.checkpoint.restore_checkpoint(args.load, state)

configure(args.cv_dir + '/log', flush_secs=5)

for epoch in range(args.max_epochs):
    state = train(state, trainloader, epoch)
    if epoch % args.test_interval == 0:
        test(state, testloader, epoch)