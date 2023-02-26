# -*- coding: utf-8 -*-
import math
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["resnet"]


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding."
    return nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def norm2d(group_norm_num_groups, planes,track_running_stats=True):
    if group_norm_num_groups is not None and group_norm_num_groups > 0:
        # group_norm_num_groups == planes -> InstanceNorm
        # group_norm_num_groups == 1 -> LayerNorm
        group_nums = planes // group_norm_num_groups
        return nn.GroupNorm(group_nums, planes)
    else:
        return nn.BatchNorm2d(planes,track_running_stats=track_running_stats)


class BasicBlock(nn.Module):
    """
    [3 * 3, 64]
    [3 * 3, 64]
    """

    expansion = 1

    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        downsample=None,
        group_norm_num_groups=None,
        track_running_stats=True
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = norm2d(group_norm_num_groups, planes=out_planes,track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = norm2d(group_norm_num_groups, planes=out_planes,track_running_stats=track_running_stats)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    [1 * 1, x]
    [3 * 3, x]
    [1 * 1, x * 4]
    """

    expansion = 4

    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        downsample=None,
        group_norm_num_groups=None,
        track_running_stats=True
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_planes, out_channels=out_planes, kernel_size=1, bias=False
        )
        self.bn1 = norm2d(group_norm_num_groups, planes=out_planes,track_running_stats=track_running_stats)

        self.conv2 = nn.Conv2d(
            in_channels=out_planes,
            out_channels=out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = norm2d(group_norm_num_groups, planes=out_planes,track_running_stats=track_running_stats)

        self.conv3 = nn.Conv2d(
            in_channels=out_planes,
            out_channels=out_planes * 4,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = norm2d(group_norm_num_groups, planes=out_planes * 4,track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def decide_num_classes(dataset):
    if dataset == "cifar10" or dataset == "svhn":
        return 10
    elif dataset == "cifar100":
        return 100
    elif "tiny" in dataset:
        return 200
    elif "imagenet" in dataset:
        return 1000
    elif "femnist" == dataset:
        return 62

class ResNetBase(nn.Module):
    def _decide_num_classes(self):
        if self.dataset == "cifar10" or self.dataset == "svhn":
            return 10
        elif self.dataset == "cifar100":
            return 100
        elif "tiny" in self.dataset:
            return 200
        elif "imagenet" in self.dataset:
            return 1000
        elif "femnist" == self.dataset:
            return 62

    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(mean=0, std=0.01)
            #     m.bias.data.zero_()

    def _make_block(
        self, block_fn, planes, block_num, stride=1, group_norm_num_groups=None,track_running_stats=True
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_fn.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block_fn.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm2d(group_norm_num_groups, planes=planes * block_fn.expansion,track_running_stats=track_running_stats),
            )

        layers = []
        layers.append(
            block_fn(
                in_planes=self.inplanes,
                out_planes=planes,
                stride=stride,
                downsample=downsample,
                group_norm_num_groups=group_norm_num_groups,
                track_running_stats=track_running_stats
            )
        )
        self.inplanes = planes * block_fn.expansion

        for _ in range(1, block_num):
            layers.append(
                block_fn(
                    in_planes=self.inplanes,
                    out_planes=planes,
                    group_norm_num_groups=group_norm_num_groups,
                    track_running_stats=track_running_stats
                )
            )
        return nn.Sequential(*layers)

    def train(self, mode=True):
        super(ResNetBase, self).train(mode)

        # if self.freeze_bn:
        #     for m in self.modules():
        #         if isinstance(m, nn.BatchNorm2d):
        #             m.eval()
        #             if self.freeze_bn_affine:
        #                 m.weight.requires_grad = False
        #                 m.bias.requires_grad = False


class ResNet_imagenet(ResNetBase):
    def __init__(
        self,
        dataset,
        resnet_size,
        group_norm_num_groups=None,
        freeze_bn=False,
        freeze_bn_affine=False,
        projection=False,
        save_activations=False
    ):
        super(ResNet_imagenet, self).__init__()
        self.dataset = dataset
        self.freeze_bn = freeze_bn
        self.freeze_bn_affine = freeze_bn_affine
        track_running_stats = not self.freeze_bn
        # define model param.
        model_params = {
            18: {"block": BasicBlock, "layers": [2, 2, 2, 2]},
            34: {"block": BasicBlock, "layers": [3, 4, 6, 3]},
            50: {"block": Bottleneck, "layers": [3, 4, 6, 3]},
            101: {"block": Bottleneck, "layers": [3, 4, 23, 3]},
            152: {"block": Bottleneck, "layers": [3, 8, 36, 3]},
        }
        block_fn = model_params[resnet_size]["block"]
        block_nums = model_params[resnet_size]["layers"]

        # decide the num of classes.
        self.num_classes = self._decide_num_classes()

        # define layers.
        self.inplanes = 64
        # self.conv1 = nn.Conv2d(
        #     in_channels=3,
        #     out_channels=64,
        #     kernel_size=7,
        #     stride=2,
        #     padding=3,
        #     bias=False,
        # )
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm2d(group_norm_num_groups, planes=64,track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)

        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_block(
            block_fn=block_fn,
            planes=64,
            block_num=block_nums[0],
            group_norm_num_groups=group_norm_num_groups,
            track_running_stats = track_running_stats
        )
        self.layer2 = self._make_block(
            block_fn=block_fn,
            planes=128,
            block_num=block_nums[1],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
            track_running_stats=track_running_stats
        )
        self.layer3 = self._make_block(
            block_fn=block_fn,
            planes=256,
            block_num=block_nums[2],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
            track_running_stats=track_running_stats
        )
        self.layer4 = self._make_block(
            block_fn=block_fn,
            planes=512,
            block_num=block_nums[3],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
            track_running_stats=track_running_stats
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = projection

        if self.projection:
            self.projection_layer = nn.Sequential(
                nn.Linear(512 * block_fn.expansion, 512 * block_fn.expansion),
                nn.ReLU(),
                nn.Linear(512 * block_fn.expansion, 256)
            )
            self.classifier = nn.Linear(
                in_features=256,
                out_features=self.num_classes,
            )
        else:
            self.classifier = nn.Linear(
                in_features=512 * block_fn.expansion, out_features=self.num_classes
            )
        self.save_activations = save_activations
        # weight initialization based on layer type.
        self._weight_initialization()
        self.train()

    def forward(self, x,start_layer_idx = 0):
        if start_layer_idx >= 0:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            #x = self.maxpool(x)
            x = self.layer1(x)
            activation1 = x
            x = self.layer2(x)
            activation2 = x
            x = self.layer3(x)
            activation3 = x
            x = self.layer4(x)
            activation4 = x
            x = self.avgpool(x)
            feature = x.view(x.size(0), -1)
            if self.projection:
                feature = self.projection_layer(feature)

            if self.save_activations:
                self.activations = [activation1, activation2, activation3,activation4]
        else:
            feature = x
        x = self.classifier(feature)
        return F.normalize(feature, dim=1),x

class CifarResNet(ResNetBase):
    def __init__(
        self,
        dataset,
        resnet_size,
        scaling=1,
        save_activations=False,
        group_norm_num_groups=None,
        freeze_bn=False,
        freeze_bn_affine=False,
        projection = False
    ):
        super(CifarResNet, self).__init__()

        self.dataset = dataset
        self.freeze_bn = freeze_bn
        self.freeze_bn_affine = freeze_bn_affine
        track_running_stats = not self.freeze_bn

        # define model.
        if resnet_size % 6 != 2:
            raise ValueError("resnet_size must be 6n + 2:", resnet_size)
        block_nums = (resnet_size - 2) // 6
        block_fn = Bottleneck if resnet_size >= 44 else BasicBlock

        # decide the num of classes.
        self.num_classes = self._decide_num_classes()

        # define layers.
        assert int(16 * scaling) > 0
        self.inplanes = int(16 * scaling)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=(16 * scaling),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = norm2d(group_norm_num_groups, planes=int(16 * scaling),track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_block(
            block_fn=block_fn,
            planes=int(16 * scaling),
            block_num=block_nums,
            group_norm_num_groups=group_norm_num_groups,
            track_running_stats=track_running_stats
        )
        self.layer2 = self._make_block(
            block_fn=block_fn,
            planes=int(32 * scaling),
            block_num=block_nums,
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
            track_running_stats=track_running_stats
        )
        self.layer3 = self._make_block(
            block_fn=block_fn,
            planes=int(64 * scaling),
            block_num=block_nums,
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
            track_running_stats=track_running_stats
        )

        self.avgpool = nn.AvgPool2d(kernel_size=8)
        feature_dim = int(64 * scaling * block_fn.expansion)
        self.projection = projection
        if self.projection:

            self.projection_layer = nn.Sequential(
                nn.Linear(feature_dim,feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim,256)
            )
            self.classifier = nn.Linear(
                in_features=256,
                out_features=self.num_classes,
            )
        else:
            self.classifier = nn.Linear(
                in_features=feature_dim,
                out_features=self.num_classes,
            )
        # weight initialization based on layer type.
        self._weight_initialization()

        # a placeholder for activations in the intermediate layers.
        self.save_activations = save_activations
        self.activations = None

    def forward(self, x, start_layer_idx = 0):
        if start_layer_idx >= 0:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            feature = x
            if self.projection:
                feature = self.projection_layer(feature)
        else:
            feature = x
        x = self.classifier(feature)

        return F.normalize(feature, dim=1),x

def resnet(conf, arch=None):

    resnet_size = int((arch if arch is not None else conf.arch).replace("resnet", ""))
    dataset = conf.data
    save_activations = True if conf.AT_beta > 0 else False

    if "cifar" in conf.data or "svhn" in conf.data:
        # model = ResNet_cifar(
        #     dataset=dataset,
        #     resnet_size=resnet_size,
        #     freeze_bn=conf.freeze_bn,
        #     freeze_bn_affine=conf.freeze_bn_affine,
        #     group_norm_num_groups=conf.group_norm_num_groups,
        # )
        model = CifarResNet(
            dataset=dataset,
            resnet_size=resnet_size,
            freeze_bn=conf.freeze_bn,
            freeze_bn_affine=conf.freeze_bn_affine,
            group_norm_num_groups=conf.group_norm_num_groups,
            projection=conf.projection,
            save_activations = save_activations
        )
    elif "imagenet" in dataset:
        if dataset == "tiny-imagenet" or dataset == "imagenet":
            model = ResNet_imagenet(
                dataset=dataset,
                resnet_size=resnet_size,
                group_norm_num_groups=conf.group_norm_num_groups,
                freeze_bn=conf.freeze_bn,
                freeze_bn_affine=conf.freeze_bn_affine,
                projection=conf.projection,
                save_activations=save_activations
            )
        # if (
        #     "imagenet" in conf.data and len(conf.data) > 8
        # ):  # i.e., downsampled imagenet with different resolution.
        else:
            model = CifarResNet(
            dataset=dataset,
            resnet_size=resnet_size,
            freeze_bn=conf.freeze_bn,
            freeze_bn_affine=conf.freeze_bn_affine,
            group_norm_num_groups=conf.group_norm_num_groups,
            projection=conf.projection,
            scaling=4
        )
            # model = ResNet_cifar(
            #     dataset=dataset,
            #     resnet_size=resnet_size,
            #     scaling=4,
            #     group_norm_num_groups=conf.group_norm_num_groups,
            #     freeze_bn=conf.freeze_bn,
            #     freeze_bn_affine=conf.freeze_bn_affine,
            # )

    else:
        raise NotImplementedError
    return model


