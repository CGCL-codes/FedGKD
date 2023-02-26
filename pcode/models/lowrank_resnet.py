import torch
import torch.nn as nn

# for time measuring
import time
import logging

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.nn.utils.fusion import fuse_conv_bn_weights

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

__all__ = ['ResNet', 'resnet18', 'baseline_resnet18', 'lowrank_resnet18_conv1x1', 'lowrank_resnet34_conv1x1',
           'resnet34', 'resnet50',
           'lowrank_resnet50', 'lowrank_resnet50_conv1x1', 'lowrank_resresnet50', 'hybrid_resnet50',
           'hybrid_resnet50_extra_bns', 'amp_hybrid_resnet50', 'amp_resnet50', 'resnet101', 'hybrid_resnet101',
           'resnet152', 'hybrid_resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'lowrank_wide_resnet50_2', 'hybrid_wide_resnet50_2', 'wide_resnet101_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

# for now we use this as a constant for all layers
# if `CONST_RANK_DENOMINATOR` = r, it means we use a rank equals to n/r
# where n is the original dimension
CONST_RANK_DENOMINATOR = 4


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    # def forward(self, x):
    #     identity = x

    #     logger.info("@@@ Inside the Blk, x shape1: {}".format(x.size()))

    #     torch.cuda.synchronize()
    #     conv1_start = time.time()
    #     out = self.conv1(x)
    #     torch.cuda.synchronize()
    #     conv1_dur = time.time() - conv1_start
    #     logger.info("@@@ Conv 1 Cost: {}".format(conv1_dur))

    #     torch.cuda.synchronize()
    #     bn1_start = time.time()
    #     out = self.bn1(out)
    #     torch.cuda.synchronize()
    #     bn1_dur = time.time() - bn1_start
    #     logger.info("@@@ BN 1 Cost: {}".format(bn1_dur))

    #     torch.cuda.synchronize()
    #     relu1_start = time.time()
    #     out = self.relu(out)
    #     torch.cuda.synchronize()
    #     relu1_dur = time.time() - relu1_start
    #     logger.info("@@@ ReLU 1 Cost: {}".format(relu1_dur))

    #     logger.info("@@@ Inside the Blk, x shape2: {}".format(x.size()))
    #     torch.cuda.synchronize()
    #     conv2_start = time.time()
    #     out = self.conv2(out)
    #     torch.cuda.synchronize()
    #     conv2_dur = time.time() - conv2_start
    #     logger.info("@@@ Conv 2 Cost: {}".format(conv2_dur))

    #     torch.cuda.synchronize()
    #     bn2_start = time.time()
    #     out = self.bn2(out)
    #     torch.cuda.synchronize()
    #     bn2_dur = time.time() - bn2_start
    #     logger.info("@@@ BN 2 Cost: {}".format(bn2_dur))

    #     if self.downsample is not None:
    #         logger.info("@@@ Inside the Blk, x shape downsample: {}".format(x.size()))
    #         torch.cuda.synchronize()
    #         downsample_start = time.time()
    #         identity = self.downsample(x)
    #         torch.cuda.synchronize()
    #         downsample_dur = time.time() - downsample_start
    #         logger.info("@@@ Downsample Cost: {}".format(downsample_dur))

    #     torch.cuda.synchronize()
    #     res_start = time.time()
    #     out += identity
    #     torch.cuda.synchronize()
    #     res_dur = time.time() - res_start
    #     logger.info("@@@ Residual Cost: {}".format(res_dur))

    #     torch.cuda.synchronize()
    #     relu2_start = time.time()
    #     out = self.relu(out)
    #     torch.cuda.synchronize()
    #     relu2_dur = time.time() - relu2_start
    #     logger.info("@@@ ReLU 2 Cost: {}".format(relu2_dur))

    #     return out

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AMPBasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(AMPBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    @autocast()
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# class LowRankBasicBlockConv1x1(nn.Module):
#     expansion = 1
#     __constants__ = ['downsample']

#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None):
#         super(LowRankBasicBlockConv1x1, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError('BasicBlock only supports groups=1 and base_width=64')
#         if dilation > 1:
#             raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1_u = conv3x3(inplanes, int(planes/CONST_RANK_DENOMINATOR), stride)
#         self.conv1_v = conv1x1(int(planes/CONST_RANK_DENOMINATOR), planes)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2_u = conv3x3(planes, int(planes/CONST_RANK_DENOMINATOR))
#         self.conv2_v = conv1x1(int(planes/CONST_RANK_DENOMINATOR), planes)
#         self.bn2 = norm_layer(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         identity = x

#         out = self.conv1_u(x)
#         out = self.conv1_v(out)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2_u(out)
#         out = self.conv2_v(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out


# updated April 14th
class LowRankBasicBlockConv1x1(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, rank_factor=4):
        super(LowRankBasicBlockConv1x1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.convbn1_u = fuse_conv_bn_weights()
        self.conv1_u = conv3x3(inplanes, int(planes / rank_factor), stride)
        #self.bn1_u = norm_layer(int(planes / rank_factor))
        self.conv1_v = conv1x1(int(planes / rank_factor), planes)
        #self.bn1_v = norm_layer(planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2_u = conv3x3(planes, int(planes / rank_factor))
        #self.bn2_u = norm_layer(int(planes / rank_factor))
        self.conv2_v = conv1x1(int(planes / rank_factor), planes)
        #self.bn2_v = norm_layer(planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1_u(x)
        #out = self.bn1_u(out)
        out = self.conv1_v(out)
        #out = self.bn1_v(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2_u(out)
        #out = self.bn2_u(out)
        out = self.conv2_v(out)
        #out = self.bn2_v(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 rank_factor=None):  # we add a dummy parameter here to make the APIs more adaptable
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AMPBottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 rank_factor=None):  # we add a dummy parameter here to make the APIs more adaptable
        super(AMPBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    @autocast()
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class LowRankBottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(LowRankBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1_u = conv1x1(inplanes, int(width / CONST_RANK_DENOMINATOR))
        self.conv1_v = nn.Linear(int(width / CONST_RANK_DENOMINATOR), width)
        self.bn1 = norm_layer(width)

        self.conv2_u = conv3x3(width, int(width / CONST_RANK_DENOMINATOR), stride, groups, dilation)
        self.conv2_v = nn.Linear(int(width / CONST_RANK_DENOMINATOR), width)
        self.bn2 = norm_layer(width)

        self.conv3_u = conv1x1(width, int(planes * self.expansion / CONST_RANK_DENOMINATOR))
        self.conv3_v = nn.Linear(int(planes * self.expansion / CONST_RANK_DENOMINATOR), planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1_u(x)
        adj_out, o_shape1 = self._adjust_itermediate_shape(out)
        out = self.conv1_v(adj_out)
        out = out.transpose(2, 1).view(o_shape1[0], out.size()[-1], o_shape1[2], o_shape1[3])
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2_u(out)
        adj_out, o_shape1 = self._adjust_itermediate_shape(out)
        out = self.conv2_v(adj_out)
        out = out.transpose(2, 1).view(o_shape1[0], out.size()[-1], o_shape1[2], o_shape1[3])
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3_u(out)
        adj_out, o_shape1 = self._adjust_itermediate_shape(out)
        out = self.conv3_v(adj_out)
        out = out.transpose(2, 1).view(o_shape1[0], out.size()[-1], o_shape1[2], o_shape1[3])
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def _adjust_itermediate_shape(self, o):
        o_shape1 = o.size()
        o1 = o.view(o_shape1[0], o_shape1[1], o_shape1[2] * o_shape1[3])
        o2 = o1.transpose(2, 1)
        return o2, o_shape1


class LowRankBottleneckConv1x1(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, rank_factor=4):
        super(LowRankBottleneckConv1x1, self).__init__()

        ################################ previous version #####################################
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1_u = conv1x1(inplanes, int(width / rank_factor))
        self.conv1_v = conv1x1(int(width / rank_factor), width)
        self.bn1 = norm_layer(width)

        self.conv2_u = conv3x3(width, int(width / rank_factor), stride, groups, dilation)
        self.conv2_v = conv1x1(int(width / rank_factor), width)
        self.bn2 = norm_layer(width)

        # TODO(hwang): to check if this works
        self.conv3_u = conv1x1(width, int(width / rank_factor))
        self.conv3_v = conv1x1(int(width / rank_factor), planes * self.expansion)
        # self.conv3 = conv1x1(width, int(planes * self.expansion))
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        ######################################################################################

        ####################### Updated on Mar 27th ########################################
        # if norm_layer is None:
        #     norm_layer = nn.BatchNorm2d
        # width = int(planes * (base_width / 64.)) * groups
        # # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        # self.conv1_u = conv1x1(inplanes, int(width/rank_factor))
        # self.bn1_u = norm_layer(int(width/rank_factor))
        # self.conv1_v = conv1x1(int(width/rank_factor), width)
        # self.bn1_v = norm_layer(width)

        # self.conv2_u = conv3x3(width, int(width/rank_factor), stride, groups, dilation)
        # self.bn2_u = norm_layer(int(width/rank_factor))
        # self.conv2_v = conv1x1(int(width/rank_factor), width)
        # self.bn2_v = norm_layer(width)

        # # TODO(hwang): to check if this works
        # self.conv3_u = conv1x1(width, int(width/rank_factor))
        # self.bn3_u = norm_layer(int(width/rank_factor))
        # self.conv3_v = conv1x1(int(width/rank_factor), planes * self.expansion)
        # #self.conv3 = conv1x1(width, int(planes * self.expansion))
        # self.bn3_v = norm_layer(planes * self.expansion)

        # self.relu = nn.ReLU(inplace=True)
        # self.downsample = downsample
        # self.stride = stride
        ###################################################################################

    def forward(self, x):
        identity = x

        ###################### old version #########################
        out = self.conv1_u(x)
        out = self.conv1_v(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2_u(out)
        out = self.conv2_v(out)
        out = self.bn2(out)
        out = self.relu(out)

        ## TODO(hwang): check performance of this
        out = self.conv3_u(out)
        out = self.conv3_v(out)

        # out = self.conv3(out)
        out = self.bn3(out)
        ############################################################

        ################ Updated on Mar 27th #######################
        # out = self.conv1_u(x)
        # out = self.bn1_u(out)
        # out = self.conv1_v(out)
        # out = self.bn1_v(out)
        # out = self.relu(out)

        # out = self.conv2_u(out)
        # out = self.bn2_u(out)
        # out = self.conv2_v(out)
        # out = self.bn2_v(out)
        # out = self.relu(out)

        # ## TODO(hwang): check performance of this
        # out = self.conv3_u(out)
        # out = self.bn3_u(out)
        # out = self.conv3_v(out)
        # out = self.bn3_v(out)
        ############################################################

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def _adjust_itermediate_shape(self, o):
        o_shape1 = o.size()
        o1 = o.view(o_shape1[0], o_shape1[1], o_shape1[2] * o_shape1[3])
        o2 = o1.transpose(2, 1)
        return o2, o_shape1


class LowRankBottleneckConv1x1ExtraBN(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, rank_factor=4):
        super(LowRankBottleneckConv1x1ExtraBN, self).__init__()

        ################################ previous version #####################################
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1_u = conv1x1(inplanes, int(width / rank_factor))
        self.bn1_u = norm_layer(int(width / rank_factor))
        self.conv1_v = conv1x1(int(width / rank_factor), width)
        self.bn1_v = norm_layer(width)

        self.conv2_u = conv3x3(width, int(width / rank_factor), stride, groups, dilation)
        self.bn2_u = norm_layer(int(width / rank_factor))
        self.conv2_v = conv1x1(int(width / rank_factor), width)
        self.bn2_v = norm_layer(width)

        # TODO(hwang): to check if this works
        self.conv3_u = conv1x1(width, int(width / rank_factor))
        self.bn3_u = norm_layer(int(width / rank_factor))
        self.conv3_v = conv1x1(int(width / rank_factor), planes * self.expansion)
        # self.conv3 = conv1x1(width, int(planes * self.expansion))
        self.bn3_v = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        ######################################################################################

    def forward(self, x):
        identity = x

        ###################### old version #########################
        out = self.conv1_u(x)
        out = self.bn1_u(out)
        out = self.conv1_v(out)
        out = self.bn1_v(out)
        out = self.relu(out)

        out = self.conv2_u(out)
        out = self.bn2_u(out)
        out = self.conv2_v(out)
        out = self.bn2_v(out)
        out = self.relu(out)

        ## TODO(hwang): check performance of this
        out = self.conv3_u(out)
        out = self.bn3_u(out)
        out = self.conv3_v(out)
        out = self.bn3_v(out)
        ############################################################

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AMPLowRankBottleneckConv1x1(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, rank_factor=4):
        super(AMPLowRankBottleneckConv1x1, self).__init__()

        ################################ previous version #####################################
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1_u = conv1x1(inplanes, int(width / rank_factor))
        self.conv1_v = conv1x1(int(width / rank_factor), width)
        self.bn1 = norm_layer(width)

        self.conv2_u = conv3x3(width, int(width / rank_factor), stride, groups, dilation)
        self.conv2_v = conv1x1(int(width / rank_factor), width)
        self.bn2 = norm_layer(width)

        # TODO(hwang): to check if this works
        self.conv3_u = conv1x1(width, int(width / rank_factor))
        self.conv3_v = conv1x1(int(width / rank_factor), planes * self.expansion)
        # self.conv3 = conv1x1(width, int(planes * self.expansion))
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        ######################################################################################

    @autocast()
    def forward(self, x):
        identity = x

        ###################### old version #########################
        out = self.conv1_u(x)
        out = self.conv1_v(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2_u(out)
        out = self.conv2_v(out)
        out = self.bn2(out)
        out = self.relu(out)

        ## TODO(hwang): check performance of this
        out = self.conv3_u(out)
        out = self.conv3_v(out)

        # out = self.conv3(out)
        out = self.bn3(out)
        ############################################################

        ################ Updated on Mar 27th #######################
        # out = self.conv1_u(x)
        # out = self.bn1_u(out)
        # out = self.conv1_v(out)
        # out = self.bn1_v(out)
        # out = self.relu(out)

        # out = self.conv2_u(out)
        # out = self.bn2_u(out)
        # out = self.conv2_v(out)
        # out = self.bn2_v(out)
        # out = self.relu(out)

        # ## TODO(hwang): check performance of this
        # out = self.conv3_u(out)
        # out = self.bn3_u(out)
        # out = self.conv3_v(out)
        # out = self.bn3_v(out)
        ############################################################

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class LowRankResidualBottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, rank_factor=4):
        super(LowRankResidualBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1_u = conv1x1(inplanes, int(width / rank_factor))
        self.conv1_v = conv1x1(int(width / rank_factor), width)
        self.conv1_res = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        self.conv2_u = conv3x3(width, int(width / rank_factor), stride, groups, dilation)
        self.conv2_v = conv1x1(int(width / rank_factor), width)
        self.conv2_res = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)

        # self.conv3_u = conv1x1(width, int(planes * self.expansion/(2*CONST_RANK_DENOMINATOR)))
        # self.conv3_v = conv1x1(int(planes * self.expansion/(2*CONST_RANK_DENOMINATOR)), planes * self.expansion)
        self.conv3 = conv1x1(width, int(planes * self.expansion))
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # out = self.conv1_u(x)
        # out = self.conv1_v(out)
        out = self.conv1_v(self.conv1_u(x)) + self.conv1_res(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2_u(out)
        # out = self.conv2_v(out)
        out = self.conv2_v(self.conv2_u(out)) + self.conv2_res(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def _adjust_itermediate_shape(self, o):
        o_shape1 = o.size()
        o1 = o.view(o_shape1[0], o_shape1[1], o_shape1[2] * o_shape1[3])
        o2 = o1.transpose(2, 1)
        return o2, o_shape1


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

    # Allow for accessing forward method in a inherited class
    forward = _forward


class AMPResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(AMPResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    @autocast()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


class VariationRankResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(VariationRankResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

    # Allow for accessing forward method in a inherited class
    forward = _forward


class LowRankResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(LowRankResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.fc_u = nn.Linear(512 * block.expansion, int(num_classes/CONST_RANK_DENOMINATOR))
        # self.fc_v = nn.Linear(int(num_classes/CONST_RANK_DENOMINATOR), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        # x = self.fc_u(x)
        # x = self.fc_v(x)
        return x

    # Allow for accessing forward method in a inherited class
    forward = _forward


class HybridResNet(nn.Module):
    def __init__(self, lowrank_block, fullrank_block, rank_factor, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        # (lowrank_block, fullrank_block, rank_factor, layers, **kwargs)
        super(HybridResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # the first half (roughly) are low rank
        # self.layer1 = self._make_layer(lowrank_block, 64, layers[0], rank_factor=rank_factor)
        # self.layer2 = self._make_layer(lowrank_block, 128, layers[1], rank_factor=rank_factor,
        #                                 stride=2,
        #                                 dilate=replace_stride_with_dilation[0])
        # self.layer3 = self._make_layer(lowrank_block, 256, layers[2], rank_factor=rank_factor,
        #                                 stride=2,
        #                                 dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(fullrank_block, 512, layers[3], stride=2,
        #                                 dilate=replace_stride_with_dilation[2])

        # the second half (roughly) are low rank
        # ================================================================================================================
        self.layer1 = self._make_layer(fullrank_block, 64, layers[0])
        self.layer2 = self._make_layer(lowrank_block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        # TODO(hwang) check how to modify this
        # _make_layer_dual_blocks(self, fr_block, lr_block, planes, blocks, stride=1, rank_factor=None, dilate=False):
        # self.layer3 = self._make_layer_dual_blocks(fullrank_block, lowrank_block, 256, layers[2],
        #                                rank_factor=rank_factor, stride=2,
        #                                dilate=replace_stride_with_dilation[1])

        self.layer3 = self._make_layer(lowrank_block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        # self.layer3 = self._make_layer(lowrank_block, 256, layers[2], rank_factor=rank_factor, stride=2,
        #                                 dilate=replace_stride_with_dilation[1])

        self.layer4 = self._make_layer(lowrank_block, 512, layers[3], rank_factor=rank_factor, stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # ================================================================================================================

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear(512 * fullrank_block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, rank_factor=None, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            ###############################################################################
            # TODO(hwang): check performance of these changes
            if block is LowRankBottleneckConv1x1 or block is LowRankBottleneckConv1x1ExtraBN:
                downsample = nn.Sequential(
                    # conv1x1(self.inplanes, planes * block.expansion, stride),
                    conv1x1(self.inplanes, int(self.inplanes / rank_factor), stride=1),
                    # norm_layer(int(self.inplanes/rank_factor)),
                    conv1x1(int(self.inplanes / rank_factor), planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

            ####
            # downsample = nn.Sequential(
            #     conv1x1(self.inplanes, planes * block.expansion, stride),
            #     norm_layer(planes * block.expansion),
            # )
            ###############################################################################
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, rank_factor=rank_factor))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, rank_factor=rank_factor))

        return nn.Sequential(*layers)

    def _make_layer_dual_blocks(self, fr_block, lr_block, planes, blocks, stride=1, rank_factor=None, dilate=False):
        """
        Trial implementation: just for `Layer3` in resnet50
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * blocks.expansion:
            ###############################################################################
            # TODO(hwang): check performance of these changes
            # if block is LowRankBottleneckConv1x1:
            #     downsample = nn.Sequential(
            #         #conv1x1(self.inplanes, planes * block.expansion, stride),
            #         conv1x1(self.inplanes, int(planes/rank_factor), stride=1),
            #         conv1x1(int(planes/rank_factor), planes * block.expansion, stride),
            #         norm_layer(planes * block.expansion),
            #     )
            # else:
            #     downsample = nn.Sequential(
            #         conv1x1(self.inplanes, planes * block.expansion, stride),
            #         norm_layer(planes * block.expansion),
            #     )

            ####
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * fr_block.expansion, stride),
                norm_layer(planes * fr_block.expansion),
            )
            ###############################################################################
        layers = []
        layers.append(fr_block(self.inplanes, planes, stride, downsample, self.groups,
                               self.base_width, previous_dilation, norm_layer, rank_factor=rank_factor))
        self.inplanes = planes * fr_block.expansion
        for block_index in range(1, blocks):
            if block_index <= 2:
                layers.append(fr_block(self.inplanes, planes, groups=self.groups,
                                       base_width=self.base_width, dilation=self.dilation,
                                       norm_layer=norm_layer, rank_factor=rank_factor))
            else:
                layers.append(lr_block(self.inplanes, planes, groups=self.groups,
                                       base_width=self.base_width, dilation=self.dilation,
                                       norm_layer=norm_layer, rank_factor=rank_factor))
        return nn.Sequential(*layers)

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    # Allow for accessing forward method in a inherited class
    forward = _forward


class AMPHybridResNet(nn.Module):
    def __init__(self, lowrank_block, fullrank_block, rank_factor, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        # (lowrank_block, fullrank_block, rank_factor, layers, **kwargs)
        super(AMPHybridResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # the first half (roughly) are low rank
        # self.layer1 = self._make_layer(lowrank_block, 64, layers[0], rank_factor=rank_factor)
        # self.layer2 = self._make_layer(lowrank_block, 128, layers[1], rank_factor=rank_factor,
        #                                 stride=2,
        #                                 dilate=replace_stride_with_dilation[0])
        # self.layer3 = self._make_layer(lowrank_block, 256, layers[2], rank_factor=rank_factor,
        #                                 stride=2,
        #                                 dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(fullrank_block, 512, layers[3], stride=2,
        #                                 dilate=replace_stride_with_dilation[2])

        # the second half (roughly) are low rank
        # ================================================================================================================
        self.layer1 = self._make_layer(fullrank_block, 64, layers[0])
        self.layer2 = self._make_layer(fullrank_block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        # TODO(hwang) check how to modify this
        # _make_layer_dual_blocks(self, fr_block, lr_block, planes, blocks, stride=1, rank_factor=None, dilate=False):
        # self.layer3 = self._make_layer_dual_blocks(fullrank_block, lowrank_block, 256, layers[2],
        #                                rank_factor=rank_factor, stride=2,
        #                                dilate=replace_stride_with_dilation[1])

        self.layer3 = self._make_layer(fullrank_block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        # self.layer3 = self._make_layer(lowrank_block, 256, layers[2], rank_factor=rank_factor, stride=2,
        #                                 dilate=replace_stride_with_dilation[1])

        self.layer4 = self._make_layer(lowrank_block, 512, layers[3], rank_factor=rank_factor, stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # ================================================================================================================

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear(512 * fullrank_block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, rank_factor=None, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            ###############################################################################
            # TODO(hwang): check performance of these changes
            if block is AMPLowRankBottleneckConv1x1:
                downsample = nn.Sequential(
                    # conv1x1(self.inplanes, planes * block.expansion, stride),
                    conv1x1(self.inplanes, int(self.inplanes / rank_factor), stride=1),
                    # norm_layer(int(self.inplanes/rank_factor)),
                    conv1x1(int(self.inplanes / rank_factor), planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

            ####
            # downsample = nn.Sequential(
            #     conv1x1(self.inplanes, planes * block.expansion, stride),
            #     norm_layer(planes * block.expansion),
            # )
            ###############################################################################
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, rank_factor=rank_factor))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, rank_factor=rank_factor))

        return nn.Sequential(*layers)

    def _make_layer_dual_blocks(self, fr_block, lr_block, planes, blocks, stride=1, rank_factor=None, dilate=False):
        """
        Trial implementation: just for `Layer3` in resnet50
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            ###############################################################################
            # TODO(hwang): check performance of these changes
            # if block is LowRankBottleneckConv1x1:
            #     downsample = nn.Sequential(
            #         #conv1x1(self.inplanes, planes * block.expansion, stride),
            #         conv1x1(self.inplanes, int(planes/rank_factor), stride=1),
            #         conv1x1(int(planes/rank_factor), planes * block.expansion, stride),
            #         norm_layer(planes * block.expansion),
            #     )
            # else:
            #     downsample = nn.Sequential(
            #         conv1x1(self.inplanes, planes * block.expansion, stride),
            #         norm_layer(planes * block.expansion),
            #     )

            ####
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * fr_block.expansion, stride),
                norm_layer(planes * fr_block.expansion),
            )
            ###############################################################################
        layers = []
        layers.append(fr_block(self.inplanes, planes, stride, downsample, self.groups,
                               self.base_width, previous_dilation, norm_layer, rank_factor=rank_factor))
        self.inplanes = planes * fr_block.expansion
        for block_index in range(1, blocks):
            if block_index <= 2:
                layers.append(fr_block(self.inplanes, planes, groups=self.groups,
                                       base_width=self.base_width, dilation=self.dilation,
                                       norm_layer=norm_layer, rank_factor=rank_factor))
            else:
                layers.append(lr_block(self.inplanes, planes, groups=self.groups,
                                       base_width=self.base_width, dilation=self.dilation,
                                       norm_layer=norm_layer, rank_factor=rank_factor))
        return nn.Sequential(*layers)

    @autocast()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class LowRankResNetConv1x1(nn.Module):

    def __init__(self, block, rank_factor, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(LowRankResNetConv1x1, self).__init__()
        zero_init_residual = True
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1,padding=1,bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, rank_factor, 64, layers[0])
        self.layer2 = self._make_layer(block, rank_factor, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, rank_factor, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, rank_factor, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.fc_u = nn.Linear(512 * block.expansion, int(num_classes/CONST_RANK_DENOMINATOR))
        # self.fc_v = nn.Linear(int(num_classes/CONST_RANK_DENOMINATOR), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, LowRankBottleneckConv1x1):
                    nn.init.constant_(m.bn3.weight, 0)
                    # nn.init.constant_(m.bn3_v.weight, 0)
                elif isinstance(m, BasicBlock):
                    # nn.init.constant_(m.bn2.weight, 0)
                    nn.init.constant_(m.bn2_v.weight, 0)

        # for m in self.modules():
        #    if isinstance(m, LowRankBottleneckConv1x1):
        #        print(m.bn3_v.weight)
        #        print("##"*15)

    def _make_layer(self, block, rank_factor, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            ###############################################################################
            # TODO(hwang): check performance of these changes
            # if block is LowRankBottleneckConv1x1 or block is LowRankBasicBlockConv1x1:
            #if block is LowRankBottleneckConv1x1:
            downsample = nn.Sequential(
                # conv1x1(self.inplanes, planes * block.expansion, stride),
                conv1x1(self.inplanes, int(planes / rank_factor), stride=1),
                conv1x1(int(planes / rank_factor), planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
            # else:
            #     downsample = nn.Sequential(
            #         conv1x1(self.inplanes, planes * block.expansion, stride),
            #         norm_layer(planes * block.expansion),
            #     )

            ####
            # downsample = nn.Sequential(
            #    conv1x1(self.inplanes, planes * fr_block.expansion, stride),
            #    norm_layer(planes * fr_block.expansion),
            # )
            ###############################################################################

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, rank_factor=rank_factor))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, rank_factor=rank_factor))

        return nn.Sequential(*layers)

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        # x = self.fc_u(x)
        # x = self.fc_v(x)
        return x

    # Allow for accessing forward method in a inherited class
    forward = _forward


class BaselineResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(BaselineResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(64 / CONST_RANK_DENOMINATOR), layers[0])
        self.layer2 = self._make_layer(block, int(128 / CONST_RANK_DENOMINATOR), layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(256 / CONST_RANK_DENOMINATOR), layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, int(512 / CONST_RANK_DENOMINATOR), layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512 / CONST_RANK_DENOMINATOR) * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    # Allow for accessing forward method in a inherited class
    forward = _forward


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def _baseline_resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = BaselineResNet(block, layers, **kwargs)
    return model


def _lowrank_resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = LowRankResNet(block, layers, **kwargs)
    return model


def _lowrank_resnet_conv1x1(arch, rank_factor, block, layers, pretrained, progress, **kwargs):
    model = LowRankResNetConv1x1(block, rank_factor, layers, **kwargs)
    return model


def _hybrid_resnet(arch, rank_factor, lowrank_block, fullrank_block, layers, pretrained, progress, **kwargs):
    model = HybridResNet(lowrank_block, fullrank_block, rank_factor, layers, **kwargs)
    return model


def _amp_hybrid_resnet(arch, rank_factor, lowrank_block, fullrank_block, layers, pretrained, progress, **kwargs):
    model = AMPHybridResNet(lowrank_block, fullrank_block, rank_factor, layers, **kwargs)
    return model


def _vr_resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = VariationRankResNet(block, layers, **kwargs)
    return model


def _amp_resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = AMPResNet(block, layers, **kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def baseline_resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _baseline_resnet('baseline_resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                            **kwargs)


def lowrank_resnet18_conv1x1(rank_factor=4, pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _lowrank_resnet_conv1x1('lowrank_resnet18_conv1x1', rank_factor, LowRankBasicBlockConv1x1, [2, 2, 2, 2],
                                   pretrained, progress,
                                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def lowrank_resnet34_conv1x1(rank_factor=4, pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _lowrank_resnet_conv1x1('lowrank_resnet34_conv1x1', rank_factor, LowRankBasicBlockConv1x1, [3, 4, 6, 3],
                                   pretrained, progress,
                                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def amp_resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _amp_resnet('amp_resnet50', AMPBottleneck, [3, 4, 6, 3], pretrained, progress,
                       **kwargs)


def vr_resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vr_resnet('vr_resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                      **kwargs)


def lowrank_resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _lowrank_resnet('resnet50', LowRankBottleneck, [3, 4, 6, 3], pretrained, progress,
                           **kwargs)


def lowrank_resnet50_conv1x1(rank_factor=4, pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _lowrank_resnet_conv1x1('resnet50', rank_factor, LowRankBottleneckConv1x1, [3, 4, 6, 3], pretrained,
                                   progress,
                                   **kwargs)


def hybrid_resnet50(rank_factor=4, pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _hybrid_resnet('resnet50', rank_factor, LowRankBottleneckConv1x1, Bottleneck, [3, 4, 6, 3], pretrained,
                          progress,
                          **kwargs)

def hybrid_resnet18(rank_factor = 4, pretrained = False, progress = True, **kwargs):
    return _hybrid_resnet('resnet18', rank_factor, LowRankBasicBlockConv1x1, BasicBlock, [2, 2, 2, 2], pretrained,
                          progress,
                          **kwargs)

def hybrid_resnet50_extra_bns(rank_factor=4, pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _hybrid_resnet('resnet50', rank_factor, LowRankBottleneckConv1x1ExtraBN, Bottleneck, [3, 4, 6, 3],
                          pretrained, progress,
                          **kwargs)


def amp_hybrid_resnet50(rank_factor=4, pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _amp_hybrid_resnet('amp_resnet50', rank_factor, AMPLowRankBottleneckConv1x1, AMPBottleneck, [3, 4, 6, 3],
                              pretrained, progress,
                              **kwargs)


def lowrank_resresnet50(rank_factor=4, pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _lowrank_resnet_conv1x1('resnet50', rank_factor, LowRankResidualBottleneck, [3, 4, 6, 3], pretrained,
                                   progress,
                                   **kwargs)


# vanilla resnet101
def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def hybrid_resnet101(rank_factor=4, pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _hybrid_resnet('resnet101', rank_factor, LowRankBottleneckConv1x1, Bottleneck, [3, 4, 23, 3], pretrained,
                          progress,
                          **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def hybrid_resnet152(rank_factor=4, pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _hybrid_resnet('resnet152', rank_factor, LowRankBottleneckConv1x1, Bottleneck, [3, 8, 36, 3], pretrained,
                          progress,
                          **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def lowrank_wide_resnet50_2(rank_factor=4, pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2

    return _lowrank_resnet_conv1x1('lowrank_wide_resnet50_2', rank_factor,
                                   LowRankBottleneckConv1x1, [3, 4, 6, 3], pretrained, progress,
                                   **kwargs)


def hybrid_wide_resnet50_2(rank_factor=4, pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _hybrid_resnet('hybrid_resnet50_2', rank_factor, LowRankBottleneckConv1x1, Bottleneck,
                          [3, 4, 6, 3], pretrained, progress,
                          **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters2(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = hybrid_resnet50(rank_factor=4)
    # model = hybrid_resnet50_extra_bns(rank_factor=4)
    # model = hybrid_resnet152(rank_factor=4)
    # model = resnet50()
    # model = wide_resnet50_2()
    # model = lowrank_wide_resnet50_2(rank_factor=4)
    # model = hybrid_wide_resnet50_2(rank_factor=4)

    # model = wide_resnet50_2()
    # model = resnet152()
    # model = lowrank_resnet34_conv1x1(rank_factor=4)
    # model = baseline_resnet18()
    print("### Let's look at the model architecture ... ")
    # simulated_input = torch.randn(2, 3, 32, 32)
    print(model)
    print("Num params: {}, {}".format(count_parameters(model), count_parameters2(model)))
    # y = model(simulated_input)
    # print("y shape: {}".format(y.size()))
    # exit()

    parameters_counter = 0
    conv_params_counter = 0
    low_rank_paras_counter = 0
    for param_index, (k, v) in enumerate(model.state_dict().items()):
        # print("Layer index: {}, layer name: {}, layer shape: {}, Num params: {}".format(
        #    param_index, k, v.shape, v.numel()
        #    ))
        parameters_counter += v.numel()
        if len(v.size()) == 4 or 'fc' in k:
            conv_params_counter += v.numel()
        if "layer4." in k:
            low_rank_paras_counter += v.numel()
    # print("Total number of parameters: {}; Res func: {}".format(parameters_counter, count_parameters(model)))
    print("Total number of parameters in Conv+FC: {}".format(conv_params_counter))
    print("Total number of parameters in last 3 blocks: {}".format(low_rank_paras_counter))

    # model = lowrank_resnet50()
    # print("Let's look at the model architecture ... ")
    # print(model)
