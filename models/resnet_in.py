import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import freeze, unfreeze
from flax.training import train_state
import orbax.checkpoint

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv(
        features=out_planes,
        kernel_size=(3, 3),
        strides=(stride, stride),
        padding='SAME',
        use_bias=False,
        feature_group_count=groups,
        kernel_dilation=(dilation, dilation),
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv(
        features=out_planes,
        kernel_size=(1, 1),
        strides=(stride, stride),
        use_bias=False,
    )


class BasicBlock(nn.Module):
    inplanes: int
    planes: int
    stride: int = 1
    downsample: nn.Module = None
    groups: int = 1
    base_width: int = 64
    dilation: int = 1
    norm_layer: nn.Module = nn.BatchNorm

    @nn.compact
    def __call__(self, x):
        if self.groups != 1 or self.base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if self.dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        identity = x

        out = conv3x3(self.inplanes, self.planes, self.stride, self.groups, self.dilation)(x)
        out = self.norm_layer()(out)
        out = nn.relu(out)

        out = conv3x3(self.planes, self.planes)(out)
        out = self.norm_layer()(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = nn.relu(out)

        return out


class Bottleneck(nn.Module):
    inplanes: int
    planes: int
    stride: int = 1
    downsample: nn.Module = None
    groups: int = 1
    base_width: int = 64
    dilation: int = 1
    norm_layer: nn.Module = nn.BatchNorm

    expansion: int = 4

    @nn.compact
    def __call__(self, x):
        width = int(self.planes * (self.base_width / 64.0)) * self.groups

        identity = x

        out = conv1x1(self.inplanes, width)(x)
        out = self.norm_layer()(out)
        out = nn.relu(out)

        out = conv3x3(width, width, self.stride, self.groups, self.dilation)(out)
        out = self.norm_layer()(out)
        out = nn.relu(out)

        out = conv1x1(width, self.planes * self.expansion)(out)
        out = self.norm_layer()(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = nn.relu(out)

        return out


class ResNet(nn.Module):
    block: nn.Module
    layers: list
    num_classes: int = 1000
    zero_init_residual: bool = False
    groups: int = 1
    width_per_group: int = 64
    replace_stride_with_dilation: list = None
    norm_layer: nn.Module = nn.BatchNorm

    def setup(self):
        if self.replace_stride_with_dilation is None:
            self.replace_stride_with_dilation = [False, False, False]
        if len(self.replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(self.replace_stride_with_dilation)
            )

        self.inplanes = 64
        self.dilation = 1
        self.groups = self.groups
        self.base_width = self.width_per_group

        self.conv1 = nn.Conv(3, self.inplanes, kernel_size=(7, 7), strides=(2, 2), padding='SAME', use_bias=False)
        self.bn1 = self.norm_layer()
        self.relu = nn.relu
        self.maxpool = nn.max_pool(window_shape=(3, 3), strides=(2, 2), padding='SAME')
        self.layer1 = self._make_layer(self.block, 64, self.layers[0])
        self.layer2 = self._make_layer(self.block, 128, self.layers[1], stride=2, dilate=self.replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(self.block, 256, self.layers[2], stride=2, dilate=self.replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(self.block, 512, self.layers[3], stride=2, dilate=self.replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Dense(features=self.num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(),
            )

        layers = []
        layers.append(
            block(
                inplanes=self.inplanes, planes=planes, stride=stride, downsample=downsample, groups=self.groups, base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    inplanes=self.inplanes,
                    planes=planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = jnp.flatten(x, 1)
        x = self.fc(x)

        return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block=block, layers=layers, **kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)