import os
import re
import numpy as np
import shutil
from random import randint, sample
import jax
import jax.numpy as jnp
from flax import linen as nn
import orbax
from flax.training.common_utils import shard
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
import torchvision.models as torchmodels
from utils.fmow_dataloader import CustomDatasetFromImages


def save_args(__file__, args):
    shutil.copy(os.path.basename(__file__), args.cv_dir)
    with open(args.cv_dir + "/args.txt", "w") as f:
        f.write(str(args))


def performance_stats(policies, rewards, matches):
    policies = jnp.concatenate(policies, axis=0)
    rewards = jnp.concatenate(rewards, axis=0)
    accuracy = jnp.mean(jnp.concatenate(matches, axis=0))

    reward = jnp.mean(rewards)
    num_unique_policy = jnp.mean(jnp.sum(policies, axis=1))
    variance = jnp.std(jnp.sum(policies, axis=1))

    policy_set = [p.astype(int).astype(str) for p in policies]
    policy_set = set(["".join(p) for p in policy_set])

    return accuracy, reward, num_unique_policy, variance, policy_set


def compute_reward(preds, targets, policy, penalty):
    patch_use = jnp.sum(policy, axis=1).astype(float) / policy.shape[1]
    sparse_reward = 1.0 - patch_use**2

    pred_idx = jnp.argmax(preds, axis=1)
    match = (pred_idx == targets).astype(float)

    reward = sparse_reward
    reward = jnp.where(match == 0, penalty, reward)
    reward = reward[:, None]

    return reward, match


def get_transforms(rnet, dset):
    if dset == "C10" or dset == "C100":
        mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
        std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    elif dset == "ImgNet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        transform_test = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

    elif dset == "fMoW":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose(
            [transforms.Resize(224), transforms.RandomCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        transform_test = transforms.Compose(
            [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

    return transform_train, transform_test


def agent_chosen_input(input_org, policy, mappings, patch_size):
    input_full = input_org.copy()
    sampled_img = jnp.zeros_like(input_org)
    for pl_ind in range(policy.shape[1]):
        mask = (policy[:, pl_ind] == 1)
        sampled_img = sampled_img.at[
            :,
            :,
            mappings[pl_ind][0] : mappings[pl_ind][0] + patch_size,
            mappings[pl_ind][1] : mappings[pl_ind][1] + patch_size,
        ].set(
            input_full[
                :,
                :,
                mappings[pl_ind][0] : mappings[pl_ind][0] + patch_size,
                mappings[pl_ind][1] : mappings[pl_ind][1] + patch_size,
            ] * mask[:, None, None, None]
        )
    return sampled_img


def action_space_model(dset):
    if dset == "C10" or dset == "C100":
        img_size = 32
        patch_size = 8
    elif dset == "fMoW" or dset == "ImgNet":
        img_size = 224
        patch_size = 56

    mappings = []
    for cl in range(0, img_size, patch_size):
        for rw in range(0, img_size, patch_size):
            mappings.append([cl, rw])

    return mappings, img_size, patch_size


def get_dataset(model, root="data/"):
    rnet, dset = model.split("_")
    transform_train, transform_test = get_transforms(rnet, dset)
    if dset == "C10":
        trainset = torchdata.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        testset = torchdata.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    elif dset == "C100":
        trainset = torchdata.CIFAR100(root=root, train=True, download=True, transform=transform_train)
        testset = torchdata.CIFAR100(root=root, train=False, download=True, transform=transform_test)
    elif dset == "ImgNet":
        trainset = torchdata.ImageFolder(root + "/ImageNet/train/", transform_train)
        testset = torchdata.ImageFolder(root + "/ImageNet/test/", transform_test)
    elif dset == "fMoW":
        trainset = CustomDatasetFromImages(root + "/fMoW/train.csv", transform_train)
        testset = CustomDatasetFromImages(root + "/fMoW/test.csv", transform_test)

    return trainset, testset


def get_model(model):
    if "C10" in model or "C100" in model:
        from models.resnet_cifar import ResNet, BasicBlock
    else:
        from models.resnet_in import ResNet, BasicBlock

    if model == "R32_C10":
        rnet_hr = ResNet(BasicBlock, [3, 4, 6, 3], 3, 10)
        rnet_lr = ResNet(BasicBlock, [3, 4, 6, 3], 3, 10)
        agent = ResNet(BasicBlock, [1, 1, 1, 1], 3, 16)

    elif model == "R32_C100":
        rnet_hr = ResNet(BasicBlock, [3, 4, 6, 3], 3, 100)
        rnet_lr = ResNet(BasicBlock, [3, 4, 6, 3], 3, 100)
        agent = ResNet(BasicBlock, [1, 1, 1, 1], 3, 16)

    elif model == "R50_ImgNet":
        rnet_hr = ResNet(BasicBlock, [3, 4, 6, 3], 7, 1000)
        rnet_lr = ResNet(BasicBlock, [3, 4, 6, 3], 7, 1000)
        agent = ResNet(BasicBlock, [2, 2, 2, 2], 3, 16)

    elif model == "R34_fMoW":
        rnet_hr = ResNet(BasicBlock, [3, 4, 6, 3], 7, 62)
        rnet_lr = ResNet(BasicBlock, [3, 4, 6, 3], 7, 62)
        agent = ResNet(BasicBlock, [2, 2, 2, 2], 3, 16)

    return rnet_hr, rnet_lr, agent