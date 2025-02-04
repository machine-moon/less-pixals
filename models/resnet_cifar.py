import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import orbax

class BasicBlock(nn.Module):
    in_planes: int
    planes: int
    stride: int = 1
    expansion: int = 1

    @nn.compact
    def __call__(self, x):
        conv1 = nn.Conv(features=self.planes, kernel_size=(3, 3), strides=(self.stride, self.stride), padding='SAME', use_bias=False)
        bn1 = nn.BatchNorm(use_running_average=False)
        conv2 = nn.Conv(features=self.planes, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False)
        bn2 = nn.BatchNorm(use_running_average=False)

        shortcut = x
        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            shortcut = nn.Conv(features=self.expansion * self.planes, kernel_size=(1, 1), strides=(self.stride, self.stride), use_bias=False)(x)
            shortcut = nn.BatchNorm(use_running_average=False)(shortcut)

        out = nn.relu(bn1(conv1(x)))
        out = bn2(conv2(out))
        out += shortcut
        out = nn.relu(out)
        return out

class Bottleneck(nn.Module):
    in_planes: int
    planes: int
    stride: int = 1
    expansion: int = 4

    @nn.compact
    def __call__(self, x):
        conv1 = nn.Conv(features=self.planes, kernel_size=(1, 1), use_bias=False)
        bn1 = nn.BatchNorm(use_running_average=False)
        conv2 = nn.Conv(features=self.planes, kernel_size=(3, 3), strides=(self.stride, self.stride), padding='SAME', use_bias=False)
        bn2 = nn.BatchNorm(use_running_average=False)
        conv3 = nn.Conv(features=self.planes * self.expansion, kernel_size=(1, 1), use_bias=False)
        bn3 = nn.BatchNorm(use_running_average=False)

        shortcut = x
        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            shortcut = nn.Conv(features=self.expansion * self.planes, kernel_size=(1, 1), strides=(self.stride, self.stride), use_bias=False)(x)
            shortcut = nn.BatchNorm(use_running_average=False)(shortcut)

        out = nn.relu(bn1(conv1(x)))
        out = nn.relu(bn2(conv2(out)))
        out = bn3(conv3(out))
        out += shortcut
        out = nn.relu(out)
        return out

class ResNet(nn.Module):
    block: nn.Module
    num_blocks: list
    initial_kernel_size: int
    num_classes: int

    def setup(self):
        self.in_planes = 64
        self.conv1 = nn.Conv(features=64, kernel_size=(self.initial_kernel_size, self.initial_kernel_size), strides=(1, 1), padding='SAME', use_bias=False)
        self.bn1 = nn.BatchNorm(use_running_average=False)
        self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.num_blocks[3], stride=2)
        self.linear = nn.Dense(features=self.num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_planes=self.in_planes, planes=planes, stride=stride))
            self.in_planes = planes * block.expansion
        return layers

    def __call__(self, x, dset, res):
        out = nn.relu(self.bn1(self.conv1(x)))
        for layer in self.layer1:
            out = layer(out)
        for layer in self.layer2:
            out = layer(out)
        for layer in self.layer3:
            out = layer(out)
        for layer in self.layer4:
            out = layer(out)
        if res == "lr":
            out = nn.avg_pool(out, window_shape=(1, 1))
        elif res == "lr-cl":  # lr classifier
            out = nn.avg_pool(out, window_shape=(4, 4))
        else:  # hr
            out = nn.avg_pool(out, window_shape=(4, 4))
        out = out.reshape((out.shape[0], -1))
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2], initial_kernel_size=3, num_classes=10)

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3], initial_kernel_size=3, num_classes=10)

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3], initial_kernel_size=3, num_classes=10)

def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3], initial_kernel_size=3, num_classes=10)

def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3], initial_kernel_size=3, num_classes=10)