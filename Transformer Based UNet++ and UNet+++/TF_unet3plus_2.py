import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.hub import load_state_dict_from_url

import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

'''
    UNet 3+
'''

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        # for _ in range(depth):
        self.layers.append(nn.ModuleList([
            PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
            PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class encoder(nn.Module):
    # def __init__(self):
    #     super(encoder,self).__init__()

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.channelTrans = nn.Conv2d(in_channels=65, out_channels=512, kernel_size=1, padding=0)

    def forward(self, x):
        x_vit = x
        x_vit = self.to_patch_embedding(x_vit)
        b, n, _ = x_vit.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x_vit = torch.cat((cls_tokens, x_vit), dim=1)
        x_vit += self.pos_embedding[:, :(n + 1)]
        x_vit = self.dropout(x_vit)

        vit_layerInfo = []
        for i in range(4):  # 设置深度的地方[6, 64+1, dim=196]
            x_vit = self.transformer(x_vit)
            vit_layerInfo.append(x_vit)

        return vit_layerInfo


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        return self.conv(x)



class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

        self.vitLayer_UpConv = nn.ConvTranspose2d(65, 65, kernel_size=2, stride=2)
        self.T_CONV10 = Conv(193, 128)
        self.T_CONV20 = Conv(321,256)
        self.T_CONV30 = Conv(577,512)
        self.T_CONV40 = Conv(1089, 1024)
        self.encoder = encoder(image_size=224, patch_size=28, num_classes=2, dim=196, depth=6, heads=16, mlp_dim=2048)

    def forward(self, x):
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        # print(self.features)
        feat1 = self.features[:4](x)
        x_original = x
        vit_layerInfo = self.encoder(x_original)
        vit_layerInfo = vit_layerInfo[::-1]
        feat2 = self.features[4:9](feat1)
        v = vit_layerInfo[0].view(4, 65, 14, 14)
        v = self.vitLayer_UpConv(v)
        v = self.vitLayer_UpConv(v)
        v = self.vitLayer_UpConv(v)
        feat2 = torch.cat([v, feat2], dim=1)
        feat2 = self.T_CONV10(feat2)
        feat3 = self.features[9:14](feat2)
        v = vit_layerInfo[1].view(4, 65, 14, 14)
        v = self.vitLayer_UpConv(v)
        v = self.vitLayer_UpConv(v)
        feat3 = torch.cat([v, feat3], dim=1)
        feat3 = self.T_CONV20(feat3)
        feat4 = self.features[14:19](feat3)
        v = vit_layerInfo[2].view(4, 65, 14, 14)
        v = self.vitLayer_UpConv(v)
        feat4 = torch.cat([v, feat4], dim=1)
        feat4 = self.T_CONV30(feat4)
        feat5 = self.features[19:-1](feat4)
        v = vit_layerInfo[3].view(4, 65, 14, 14)
        feat5 = torch.cat([v, feat5], dim=1)
        feat5 = self.T_CONV40(feat5)
        return [feat1, feat2, feat3, feat4, feat5]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, in_channels=3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# 512,512,3 -> 512,512,64 -> 256,256,64 -> 256,256,128 -> 128,128,128 -> 128,128,256 -> 64,64,256
# 64,64,512 -> 32,32,512 -> 32,32,512
cfgs = {
    # 'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 1024, 1024, 'M']
}


def VGG16(pretrained, in_channels=3, **kwargs):
    model = VGG(make_layers(cfgs["D"], batch_norm=False, in_channels=in_channels), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth",
                                              model_dir="./model_data")
        model.load_state_dict(state_dict)

    del model.avgpool
    del model.classifier
    return model

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # 利用1x1卷积下降通道数
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # 利用3x3卷积进行特征提取
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        # 利用1x1卷积上升通道数
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


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        # -----------------------------------------------------------#
        #   假设输入图像为600,600,3
        #   当我们使用resnet50的时候
        # -----------------------------------------------------------#
        self.inplanes = 64
        super(ResNet, self).__init__()
        # 600,600,3 -> 300,300,64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change
        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75,75,512 -> 38,38,1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # 38,38,1024 -> 19,19,2048
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        x = self.conv1(x)
        x = self.bn1(x)
        feat1 = self.relu(x)

        x = self.maxpool(feat1)
        feat2 = self.layer1(x)

        feat3 = self.layer2(feat2)
        feat4 = self.layer3(feat3)
        feat5 = self.layer4(feat4)
        return [feat1, feat2, feat3, feat4, feat5]


def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(
            model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth', model_dir='model_data'),
            strict=False)

    del model.avgpool
    del model.fc
    return model


class unetUp4(nn.Module):
    def __init__(self, filters, out_size):
        super(unetUp4, self).__init__()
        count = 5
        self.downsample1 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.downsample2 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.downsample3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(filters[0], out_size, 3, padding=1)
        self.conv2 = nn.Conv2d(filters[1], out_size, 3, padding=1)
        self.conv3 = nn.Conv2d(filters[2], out_size, 3, padding=1)
        self.conv4 = nn.Conv2d(filters[3], out_size, 3, padding=1)
        self.conv5 = nn.Conv2d(filters[4], out_size, 3, padding=1)
        self.conv = nn.Conv2d(filters[0]*count, out_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2, inputs3, inputs4, inputs5, bn):
        if bn:
            h1_downsample = self.downsample1(inputs1)
            h1_downsample_conv = self.conv1(h1_downsample)
            h1_downsample_bn = self.bn(h1_downsample_conv)
            h1_downsample_relu = self.relu(h1_downsample_bn)

            h2_downsample = self.downsample2(inputs2)
            h2_downsample_conv = self.conv2(h2_downsample)
            h2_downsample_bn = self.bn(h2_downsample_conv)
            h2_downsample_relu = self.relu(h2_downsample_bn)

            h3_downsample = self.downsample3(inputs3)
            h3_downsample_conv = self.conv3(h3_downsample)
            h3_downsample_bn = self.bn(h3_downsample_conv)
            h3_downsample_relu = self.relu(h3_downsample_bn)

            h5_upsample = self.upsample(inputs5)
            h5_upsample_conv = self.conv5(h5_upsample)
            h5_upsample_bn = self.bn(h5_upsample_conv)
            h5_upsample_relu = self.relu(h5_upsample_bn)

            h4_conv = self.conv4(inputs4)
            h4_bn = self.bn(h4_conv)
            h4_relu = self.relu(h4_bn)

            h_concat = torch.cat((h1_downsample_relu, h2_downsample_relu, h3_downsample_relu, h4_relu, h5_upsample_relu), 1)
            h_conv = self.conv(h_concat)
            h_bn = self.bn(h_conv)
            h_relu = self.relu(h_bn)
            return h_relu
        else:
            h1_downsample = self.downsample1(inputs1)
            h1_downsample_conv = self.conv1(h1_downsample)
            h1_downsample_relu = self.relu(h1_downsample_conv)

            h2_downsample = self.downsample2(inputs2)
            #print("h2_downsample:" + str(h2_downsample.shape))
            h2_downsample_conv = self.conv2(h2_downsample)
            h2_downsample_relu = self.relu(h2_downsample_conv)

            h3_downsample = self.downsample3(inputs3)
            h3_downsample_conv = self.conv3(h3_downsample)
            h3_downsample_relu = self.relu(h3_downsample_conv)

            h5_upsample = self.upsample(inputs5)
            h5_upsample_conv = self.conv5(h5_upsample)
            h5_upsample_relu = self.relu(h5_upsample_conv)

            h4_conv = self.conv4(inputs4)
            h4_relu = self.relu(h4_conv)

            h_concat = torch.cat(
                (h1_downsample_relu, h2_downsample_relu, h3_downsample_relu, h4_relu, h5_upsample_relu), 1)
            h_conv = self.conv(h_concat)
            h_relu = self.relu(h_conv)
            return h_relu

class unetUp3(nn.Module):
    def __init__(self, filters, out_size):
        super(unetUp3, self).__init__()
        count = 5
        self.downsample1 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.downsample2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample5 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.conv1 = nn.Conv2d(filters[0], out_size, 3, padding=1)
        self.conv2 = nn.Conv2d(filters[1], out_size, 3, padding=1)
        self.conv3 = nn.Conv2d(filters[2], out_size, 3, padding=1)
        self.conv4 = nn.Conv2d(out_size, out_size, 3, padding=1)
        self.conv5 = nn.Conv2d(filters[4], out_size, 3, padding=1)
        self.conv = nn.Conv2d(filters[0] * count, out_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2, inputs3, inputs4, inputs5, bn):
        if bn:
            h1_downsample = self.downsample1(inputs1)
            h1_downsample_conv = self.conv1(h1_downsample)
            h1_downsample_bn = self.bn(h1_downsample_conv)
            h1_downsample_relu = self.relu(h1_downsample_bn)

            h2_downsample = self.downsample2(inputs2)
            h2_downsample_conv = self.conv2(h2_downsample)
            h2_downsample_bn = self.bn(h2_downsample_conv)
            h2_downsample_relu = self.relu(h2_downsample_bn)

            h3_conv = self.conv3(inputs3)
            h3_bn = self.bn(h3_conv)
            h3_relu = self.relu(h3_bn)

            h4_upsample = self.upsample4(inputs4)
            h4_upsample_conv = self.conv4(h4_upsample)
            h4_upsample_bn = self.bn(h4_upsample_conv)
            h4_upsample_relu = self.relu(h4_upsample_bn)

            h5_upsample = self.upsample5(inputs5)
            h5_upsample_conv = self.conv5(h5_upsample)
            h5_upsample_bn = self.bn(h5_upsample_conv)
            h5_upsample_relu = self.relu(h5_upsample_bn)

            h_concat = torch.cat(
                (h1_downsample_relu, h2_downsample_relu, h3_relu, h4_upsample_relu, h5_upsample_relu), 1)
            h_conv = self.conv(h_concat)
            h_bn = self.bn(h_conv)
            h_relu = self.relu(h_bn)
            return h_relu
        else:
            h1_downsample = self.downsample1(inputs1)
            h1_downsample_conv = self.conv1(h1_downsample)
            h1_downsample_relu = self.relu(h1_downsample_conv)

            h2_downsample = self.downsample2(inputs2)
            h2_downsample_conv = self.conv2(h2_downsample)
            h2_downsample_relu = self.relu(h2_downsample_conv)

            h3_conv = self.conv3(inputs3)
            h3_relu = self.relu(h3_conv)

            h4_upsample = self.upsample4(inputs4)
            h4_upsample_conv = self.conv4(h4_upsample)
            h4_upsample_relu = self.relu(h4_upsample_conv)

            h5_upsample = self.upsample5(inputs5)
            h5_upsample_conv = self.conv5(h5_upsample)
            h5_upsample_relu = self.relu(h5_upsample_conv)

            h_concat = torch.cat(
                (h1_downsample_relu, h2_downsample_relu, h3_relu, h4_upsample_relu, h5_upsample_relu), 1)
            h_conv = self.conv(h_concat)
            h_relu = self.relu(h_conv)
            return h_relu

class unetUp2(nn.Module):
    def __init__(self, filters, out_size):
        super(unetUp2, self).__init__()
        count = 5
        self.downsample1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample5 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.conv1 = nn.Conv2d(filters[0], out_size, 3, padding=1)
        self.conv2 = nn.Conv2d(filters[1], out_size, 3, padding=1)
        self.conv3 = nn.Conv2d(out_size, out_size, 3, padding=1)
        self.conv4 = nn.Conv2d(out_size, out_size, 3, padding=1)
        self.conv5 = nn.Conv2d(filters[4], out_size, 3, padding=1)
        self.conv = nn.Conv2d(filters[0] * count, out_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2, inputs3, inputs4, inputs5, bn):
        if bn:
            h1_downsample = self.downsample1(inputs1)
            h1_downsample_conv = self.conv1(h1_downsample)
            h1_downsample_bn = self.bn(h1_downsample_conv)
            h1_downsample_relu = self.relu(h1_downsample_bn)

            h2_conv = self.conv2(inputs2)
            h2_bn = self.bn(h2_conv)
            h2_relu = self.relu(h2_bn)

            h3_upsample = self.upsample3(inputs3)
            h3_upsample_conv = self.conv3(h3_upsample)
            h3_upsample_bn = self.bn(h3_upsample_conv)
            h3_upsample_relu = self.relu(h3_upsample_bn)

            h4_upsample = self.upsample4(inputs4)
            h4_upsample_conv = self.conv4(h4_upsample)
            h4_upsample_bn = self.bn(h4_upsample_conv)
            h4_upsample_relu = self.relu(h4_upsample_bn)

            h5_upsample = self.upsample5(inputs5)
            h5_upsample_conv = self.conv4(h5_upsample)
            h5_upsample_bn = self.bn(h5_upsample_conv)
            h5_upsample_relu = self.relu(h5_upsample_bn)

            h_concat = torch.cat(
                (h1_downsample_relu, h2_relu, h3_upsample_relu, h4_upsample_relu, h5_upsample_relu), 1)
            h_conv = self.conv(h_concat)
            h_bn = self.bn(h_conv)
            h_relu = self.relu(h_bn)
            return h_relu
        else:
            h1_downsample = self.downsample1(inputs1)
            h1_downsample_conv = self.conv1(h1_downsample)
            h1_downsample_relu = self.relu(h1_downsample_conv)

            h2_conv = self.conv2(inputs2)
            h2_relu = self.relu(h2_conv)

            h3_upsample = self.upsample3(inputs3)
            h3_upsample_conv = self.conv3(h3_upsample)
            h3_upsample_relu = self.relu(h3_upsample_conv)

            h4_upsample = self.upsample4(inputs4)
            h4_upsample_conv = self.conv4(h4_upsample)
            h4_upsample_relu = self.relu(h4_upsample_conv)

            h5_upsample = self.upsample5(inputs5)
            h5_upsample_conv = self.conv5(h5_upsample)
            h5_upsample_relu = self.relu(h5_upsample_conv)

            h_concat = torch.cat(
                (h1_downsample_relu, h2_relu, h3_upsample_relu, h4_upsample_relu, h5_upsample_relu), 1)
            h_conv = self.conv(h_concat)
            h_relu = self.relu(h_conv)
            return h_relu


class unetUp1(nn.Module):
    def __init__(self, filters, out_size):
        super(unetUp1, self).__init__()
        count = 5
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.conv1 = nn.Conv2d(filters[0], out_size, 3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, 3, padding=1)
        self.conv3 = nn.Conv2d(out_size, out_size, 3, padding=1)
        self.conv4 = nn.Conv2d(out_size, out_size, 3, padding=1)
        self.conv5 = nn.Conv2d(filters[4], out_size, 3, padding=1)
        self.conv = nn.Conv2d(filters[0] * count, out_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2, inputs3, inputs4, inputs5, bn):
        if bn:
            h1_conv = self.conv1(inputs1)
            h1_bn = self.bn(h1_conv)
            h1_relu = self.relu(h1_bn)

            h2_upsample = self.upsample2(inputs2)
            h2_upsample_conv = self.conv2(h2_upsample)
            h2_upsample_bn = self.bn(h2_upsample_conv)
            h2_upsample_relu = self.relu(h2_upsample_bn)

            h3_upsample = self.upsample3(inputs3)
            h3_upsample_conv = self.conv3(h3_upsample)
            h3_upsample_bn = self.bn(h3_upsample_conv)
            h3_upsample_relu = self.relu(h3_upsample_bn)

            h4_upsample = self.upsample4(inputs4)
            h4_upsample_conv = self.conv4(h4_upsample)
            h4_upsample_bn = self.bn(h4_upsample_conv)
            h4_upsample_relu = self.relu(h4_upsample_bn)

            h5_upsample = self.upsample5(inputs5)
            h5_upsample_conv = self.conv5(h5_upsample)
            h5_upsample_bn = self.bn(h5_upsample_conv)
            h5_upsample_relu = self.relu(h5_upsample_bn)

            h_concat = torch.cat(
                (h1_relu, h2_upsample_relu, h3_upsample_relu, h4_upsample_relu, h5_upsample_relu), 1)
            h_conv = self.conv(h_concat)
            h_bn = self.bn(h_conv)
            h_relu = self.relu(h_bn)
            return h_relu
        else:
            h1_conv = self.conv1(inputs1)
            h1_relu = self.relu(h1_conv)

            h2_upsample = self.upsample2(inputs2)
            h2_upsample_conv = self.conv2(h2_upsample)
            h2_upsample_relu = self.relu(h2_upsample_conv)

            h3_upsample = self.upsample3(inputs3)
            h3_upsample_conv = self.conv3(h3_upsample)
            h3_upsample_relu = self.relu(h3_upsample_conv)

            h4_upsample = self.upsample4(inputs4)
            h4_upsample_conv = self.conv4(h4_upsample)
            h4_upsample_relu = self.relu(h4_upsample_conv)

            h5_upsample = self.upsample5(inputs5)
            h5_upsample_conv = self.conv5(h5_upsample)
            h5_upsample_relu = self.relu(h5_upsample_conv)

            h_concat = torch.cat(
                (h1_relu, h2_upsample_relu, h3_upsample_relu, h4_upsample_relu, h5_upsample_relu), 1)
            h_conv = self.conv(h_concat)
            h_relu = self.relu(h_conv)
            return h_relu


class UNet3Plus(nn.Module):
    def __init__(self, num_classes = 1, pretrained = False, backbone = 'vgg'):
        super(UNet3Plus, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            #in_filters  = [192, 384, 768, 1536]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            #in_filters  = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        if backbone == 'vgg':
            out_filters = [64, 128, 256, 512, 1024]
            out_channels = 64
        elif backbone == "resnet50":
            out_filters = [64, 256, 512, 1024, 2048]
            out_channels = 64
        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp4(out_filters, out_channels)
        # 128,128,256
        self.up_concat3 = unetUp3(out_filters, out_channels)
        # 256,256,128
        self.up_concat2 = unetUp2(out_filters, out_channels)
        # 512,512,64
        self.up_concat1 = unetUp1(out_filters, out_channels)

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs, bn=False):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        #print("feat1:"+str(feat1.shape))
        #print("feat2:" + str(feat2.shape))
        #print("feat3:" + str(feat3.shape))
        #print("feat4:" + str(feat4.shape))
        #print("feat5:" + str(feat5.shape))
        up4 = self.up_concat4(feat1, feat2, feat3, feat4, feat5, bn)
        up3 = self.up_concat3(feat1, feat2, feat3, up4, feat5, bn)
        up2 = self.up_concat2(feat1, feat2, up3, up4, feat5, bn)
        up1 = self.up_concat1(feat1, up2, up3, up4, feat5, bn)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        
        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True


class UNet3Plus_DeepSupervision(nn.Module):
    def __init__(self, num_classes=2, pretrained=False, backbone='vgg'):
        super(UNet3Plus, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1536]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            in_filters = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]
        out_channels = 64
        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp4(out_filters, out_channels)
        # 128,128,256
        self.up_concat3 = unetUp3(out_filters, out_channels)
        # 256,256,128
        self.up_concat2 = unetUp2(out_filters, out_channels)
        # 512,512,64
        self.up_concat1 = unetUp1(out_filters, out_channels)

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs, bn):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        up4 = self.up_concat4(feat1, feat2, feat3, feat4, feat5, bn)
        up3 = self.up_concat3(feat1, feat2, feat3, up4, feat5, bn)
        up2 = self.up_concat2(feat1, feat2, up3, up4, feat5, bn)
        up1 = self.up_concat1(feat1, up2, up3, up4, feat5, bn)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)

        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True


class UNet3Plus_DeepSupervision_CGM(nn.Module):
    def __init__(self, num_classes=2, pretrained=False, backbone='vgg'):
        super(UNet3Plus, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1536]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            in_filters = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]
        out_channels = 64
        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp4(out_filters, out_channels)
        # 128,128,256
        self.up_concat3 = unetUp3(out_filters, out_channels)
        # 256,256,128
        self.up_concat2 = unetUp2(out_filters, out_channels)
        # 512,512,64
        self.up_concat1 = unetUp1(out_filters, out_channels)

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs, bn):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        up4 = self.up_concat4(feat1, feat2, feat3, feat4, feat5, bn)
        up3 = self.up_concat3(feat1, feat2, feat3, up4, feat5, bn)
        up2 = self.up_concat2(feat1, feat2, up3, up4, feat5, bn)
        up1 = self.up_concat1(feat1, up2, up3, up4, feat5, bn)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)

        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True


# def test():
#     x = torch.randn((4, 3, 224, 224))
#     model = UNet3Plus()
#     preds = model(x)
#     print(preds.shape)
#     print(x.shape)



# if __name__ == "__main__":
#     test()
