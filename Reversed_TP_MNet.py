import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchvision import models as resnet_model
import torch.nn.functional as F


"""
Reversed TP-MNet

Authors: Yuefei Wang, Xiang Zhang
Chengdu University
"""



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
        x_vit = torch.cat([cls_tokens, x_vit], dim=1)
        x_vit += self.pos_embedding[:, :(n + 1)]
        x_vit = self.dropout(x_vit)

        vit_layerInfo = []
        for i in range(4):  # 设置深度的地方[6, 64+1, dim=196]
            x_vit = self.transformer(x_vit)
            vit_layerInfo.append(x_vit)

        return vit_layerInfo
    


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class soubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(soubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class BC1_ChannelChange(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BC1_ChannelChange, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class ASPP(nn.Module):
    def __init__(self, inplanes):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]


        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self._init_weight()
        self.upconv = nn.ConvTranspose2d(1280, 1280, kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)#双线性插值,与上采样类似,还原图像大小
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.upconv(x)        

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



class TFRB(nn.Module):
    def __init__(self, channels,out_channels=128,):
        super(TFRB, self,).__init__()

        self.upsample = nn.ConvTranspose2d(channels, channels // 2, kernel_size=2, stride=2)

        self.block1 = nn.Sequential(
            nn.Conv2d(channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 7, 1, 3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels),

        )
        self.block2 = nn.Sequential(
            nn.Conv2d(channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 7, 1, 3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels),

        )
        self.conv_6 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.conv_7 = nn.Sequential(
            nn.Conv2d(128, 33, kernel_size=1),
            nn.BatchNorm2d(33),
            nn.ReLU(inplace=True)
        )
        self.conv_8 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.upsample_1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.last_conv = nn.Sequential(
            nn.Conv2d(256 + channels//2, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)            
        )
                                        

    def forward(self, x, x_original):
        br1 = x         #[4,512,28,28]
        br1 = self.block1(br1)#[4,512,28,28]
        br1 = br1 + x     #残差连接，#[4,512,28,28]
        newbr = br1     #复制,用于下一次相加,[4,512,28,28]
        br1 = self.block2(br1)#br1[4,512,28,28]
        br1 = br1 + newbr #残差连接，br1[4,512,28,28]
        br1 = br1 + x #残差连接,[4,512,28,28]
        br1 = self.upsample(br1)#[4,256,56,56]



        x = x_original
        # x.shape = [4, 256, 56, 56]
        """为了与图像br1进行cat,所以x需要做出相应的该变"""
        if br1.size(2) == 112:
            x = self.upsample_1(x)
        x = torch.cat([br1, x], dim=1)
        x = self.last_conv(x)  #cat后进行一次3*3卷积，归一化，ReLU激活函数
        return x


class TFEB_Context_sensitive_Path(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TFEB_Context_sensitive_Path, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 11, dilation=11, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 9, dilation=9, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 7, dilation=7, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class TFEB_Attention_Path(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TFEB_Attention_Path, self).__init__()
        self.SE11Branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # avgpool
            nn.Conv2d(in_channels, in_channels // 16, 1, 1, 0),  # 1x1conv，替代linear
            nn.ReLU(inplace=True),  # relu
            nn.Conv2d(in_channels // 16, in_channels, 1, 1, 0),  # 1x1conv，替代linear
            nn.Sigmoid()  # sigmoid，将输出压缩到(0,1)
        )
        self.Conv3Branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.Conv35Branch3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 5, 1, 2, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.Convfinal = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

       
    def forward(self, x):
        w = self.SE11Branch1(x)
        b2 = self.Conv3Branch2(x)
        b3 = self.Conv35Branch3(x)
        b2 = w * b2
        b3 = w * b3
        bcat = torch.cat([b2, b3], dim=1)
        tbsub2 = self.Convfinal(bcat)
        return tbsub2


class Reversal_TP_MNet(nn.Module):
    def __init__(self, features=[64, 128, 256, 512], out_channels=1):
        super(Reversal_TP_MNet, self).__init__()

        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        in_channels = 3
        downfeature = [64, 128, 256, 512]

        self.down_1 = DoubleConv(3, 64)
        self.down_2 = DoubleConv(64, 128)
        self.down_3 = DoubleConv(128, 256)
        self.down_4 = DoubleConv(256, 512)
        self.down_5 = DoubleConv(512, 1024)

        self.C3To1 = BC1_ChannelChange(3, 1)

        downfeature = [64, 128, 256, 512]
        self.downs = nn.ModuleList()
        for feature in downfeature:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        self.SupplementConv = nn.Conv2d(512, 512, 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder = encoder(image_size=224, patch_size=28, num_classes=2, dim=196, depth=6, heads=16, mlp_dim=2048)

        """TFEB"""
        self.scb1_l1 = TFEB_Context_sensitive_Path(64, 128)
        self.scb2_l1 = TFEB_Attention_Path(64, 128)
        self.scb1_l2 = TFEB_Context_sensitive_Path(256, 256)
        self.scb2_l2 = TFEB_Attention_Path(256, 256)
        self.scb2_l2_up = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.scb1_l3 = TFEB_Context_sensitive_Path(512, 512)
        self.scb2_l3 = TFEB_Attention_Path(512, 512)
        self.scb2_l3_up = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.scb1_l4 = TFEB_Context_sensitive_Path(1024, 1024)
        self.scb2_l4 = TFEB_Attention_Path(1024, 1024)
        self.scb2_l4_up = nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2)
        

        self.bottleneck = DoubleConv(512, 1024)

        self.upsampleups = nn.ModuleList()  # 分枝1的上采样，融合Upsample粘贴Skipconnection
        self.upsampleups.append(nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2))  # 上采样
        self.upsampleups.append(DoubleConv(512 + 1024, 512))  # 双卷积
        self.upsampleups.append(nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2))  # 第4-第3上采样
        self.upsampleups.append(DoubleConv(256 + 65 + 512, 256))  # 双卷积(此前粘贴邻分枝上一层)
        self.upsampleups.append(nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2))  # 第3-第2上采样
        self.upsampleups.append(DoubleConv(128 + 65 + 256, 128))  # 双卷积(此前粘贴邻分枝上一层)
        self.upsampleups.append(nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2))  # 第2-第1上采样
        self.upsampleups.append(DoubleConv(64 + 128, 64))  # 双卷积(此前粘贴邻分枝上一层)

        self.skups = nn.ModuleList()
        self.skups.append(DoubleConv(512 + 1024, 512))  # 双卷积
        self.skups.append(nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2))  # 第4-第3上采样
        self.skups.append(DoubleConv(256 + 512 + 512, 256))  # 双卷积
        self.skups.append(nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2))  # 第3-第2上采样
        self.skups.append(DoubleConv(128 + 256 + 256, 128))  # 双卷积
        self.skups.append(nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2))  # 第2-第1上采样
        self.skups.append(DoubleConv(64 + 65, 64))  # 双卷积

        self.vitups = nn.ModuleList()
        self.vitups.append(DoubleConv(65 + 1024, 65))
        self.vitups.append(DoubleConv(512 + 65 + 512, 65))
        self.vitups.append(DoubleConv(256 + 65 + 256, 65))
        self.vitups.append(DoubleConv(128 + 65, 65))
        self.vitups.append(nn.ConvTranspose2d(65, 65, kernel_size=2, stride=2))
        self.vitups.append(nn.ConvTranspose2d(65, 64, kernel_size=2, stride=2))

        self.transblcok = nn.ModuleList()
        self.transblcok.append(TFRB(512))
        self.transblcok.append(TFRB(65))
        self.transblcok.append(TFRB(256))

        self.trans_conv = nn.ModuleList()  # 跳连trans所用的卷积
        self.trans_conv.append(soubleConv(65 + 65, 65))
        self.trans_conv.append(soubleConv(65 + 64, 64))

        self.Three2One = DoubleConv(64 * 3, 64)

        self.finalconv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.vitLayer_UpConv = nn.ConvTranspose2d(65, 65, kernel_size=2, stride=2)
        self.e_UpConv = nn.ModuleList()
        for f in reversed(downfeature):
            self.e_UpConv.append(nn.ConvTranspose2d(f, f, kernel_size=2, stride=2))

        self.final_conv1 = nn.ConvTranspose2d(64, 32, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, 1, 3, padding=1)
        self.aspp = ASPP(256)

        self.e1_LC = nn.Conv2d(64, 64, kernel_size=1)
        self.e2_LC = nn.Conv2d(128, 128, kernel_size=1)
        self.e3_LC = nn.Conv2d(256, 256, kernel_size=1)
        self.e4_LC = nn.Conv2d(512, 512, kernel_size=1)


    def forward(self, x):
        x_original_2 = x

        branch_conv = x
        branch_sc = []
        for down in self.downs:
            branch_conv = down(branch_conv)
            branch_sc.append(branch_conv)
            branch_conv = self.pool(branch_conv)
        branch_conv = self.SupplementConv(branch_conv)
        branch_sc.append(branch_conv)

        # decoder part
        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        x_aspp = self.aspp(e3)

        #########skip connection

        sc_B1_l1_out = self.scb1_l1(branch_sc[0])  # [128, 224, 224]
        sc_B2_l1_out = self.scb2_l1(e1)  # [128, 112, 112]
        #############第2层数据准备
        sc_B1_l2_in = torch.cat([sc_B2_l1_out, branch_sc[1]], dim=1)  # 第二层数据B1准备 [256,112,112]
        sc_B2_l2_in = self.pool(sc_B1_l1_out)  # 第二层数据B2准备：池化1次
        sc_B2_l2_in = self.pool(sc_B2_l2_in)  # 第二层数据B2准备：池化2次
        sc_B2_l2_in = torch.cat([sc_B2_l2_in, e2], dim=1)  # 第二层数据B2准备：[256, 56, 56]
        #############第二层TBlock执行
        sc_B1_l2_out = self.scb1_l2(sc_B1_l2_in)  # 第二层B1输出：[256, 112, 112]
        sc_B2_l2_out = self.scb2_l2(sc_B2_l2_in)  # 第二层B2输出：[256, 56, 56]
        sc_B2_l2_out_forsc = self.scb2_l2_up(sc_B2_l2_out)  # 第二层B2输出：[256, 112, 112]
        sc2 = sc_B1_l2_out + sc_B2_l2_out_forsc  # 跳连接sc4    [256, 112, 112]
        #############第三层数据准备
        sc_B1_l3_in = torch.cat([sc_B2_l2_out, branch_sc[2]], dim=1)  # 第三层数据B1准备 [512, 56, 56]
        sc_B2_l3_in = self.pool(sc_B1_l2_out)  # 第三层数据B2准备：池化1次
        sc_B2_l3_in = self.pool(sc_B2_l3_in)  # 第三层数据B2准备：池化2次
        sc_B2_l3_in = torch.cat([sc_B2_l3_in, e3], dim=1)  # 第三层数据B2准备：[512, 28, 28]
        #############第三层TBlock执行
        sc_B1_l3_out = self.scb1_l3(sc_B1_l3_in)  # 第三层B1输出：[512, 56, 56]
        sc_B2_l3_out = self.scb2_l3(sc_B2_l3_in)  # 第三层B2输出：[512, 28, 28]
        sc_B2_l3_out_forsc = self.scb2_l3_up(sc_B2_l3_out)  # 第三层B2输出：[512, 56, 56]
        sc3 = sc_B1_l3_out + sc_B2_l3_out_forsc  # 跳连接sc3    [512, 56, 56]
        #############第四层数据准备
        sc_B1_l4_in = torch.cat([sc_B2_l3_out, branch_sc[3]], dim=1)  # 第四层数据B1准备 [1024,28,28]
        sc_B2_l4_in = self.pool(sc_B1_l3_out)  # 第四层数据B2准备：池化1次
        sc_B2_l4_in = self.pool(sc_B2_l4_in)  # 第四层数据B2准备：池化2次[512,14,14]
        sc_B2_l4_in = torch.cat([sc_B2_l4_in, e4], dim=1)  # 第四层数据B2准备：[1024,14,14]
        #############第四层TBlock执行
        sc_B1_l4_out = self.scb1_l4(sc_B1_l4_in)  # 第四层B1输出：[1024, 28, 28]
        sc_B2_l4_out = self.scb2_l4(sc_B2_l4_in)  # 第四层B2输出：[1024, 14, 14]
        sc_B2_l4_out_forsc = self.scb2_l4_up(sc_B2_l4_out)  # 第四层B2输出：[1024, 28, 28]
        sc4 = sc_B1_l4_out + sc_B2_l4_out_forsc  # 跳连接sc4     [1024, 28, 28]

        En_x = self.bottleneck(e4 + branch_sc[4])

        vit_layerInfo = self.encoder(x_original_2)
        vit_layerInfo = vit_layerInfo[::-1]  # 翻转，呈正序。0表示第四层...3表示第一层

        # 准备阶段，三方到第四层
        # Branch1: Dec_Br1_l4
        Dec_Br1_l4 = self.upsampleups[0](En_x)  # 从bottleneck上采样到第四层 [4,512,28,28]
        # Branch2: Dec_Br2_l4:                                       skipconnection分枝
        e4_up = self.e_UpConv[0](e4)
        e3_up = self.e_UpConv[1](e3)
        e2_up = self.e_UpConv[2](e2)
        e1_up = self.e_UpConv[3](e1)
        Dec_Br2_l4 = e4_up + branch_sc[3]  # [4,512,28,28]
        # Branch3: ViT_Br3_l4:
        ViT_Br3_l4 = vit_layerInfo[0].view(4, 65, 14, 14)
        ViT_Br3_l4 = self.vitLayer_UpConv(ViT_Br3_l4)  # ViT_Br3_l4 [4, 65, 28, 28]
        # #Branch3的剩余三个（共循环四次，前三次分别为）
        ViT_Br3_l3 = vit_layerInfo[1].view(4, 65, 14, 14)
        ViT_Br3_l3 = self.vitLayer_UpConv(ViT_Br3_l3)
        ViT_Br3_l3 = self.vitLayer_UpConv(ViT_Br3_l3)  # 这个是倒数第二次也就是第三次的trans

        ViT_Br3_l2 = vit_layerInfo[2].view(4, 65, 14, 14)
        ViT_Br3_l2 = self.vitLayer_UpConv(ViT_Br3_l2)
        ViT_Br3_l2 = self.vitLayer_UpConv(ViT_Br3_l2)
        ViT_Br3_l2 = self.vitLayer_UpConv(ViT_Br3_l2)  # 这个是倒数第一次也就是第二次的trans

        ViT_Br3_l1 = vit_layerInfo[3].view(4, 65, 14, 14)
        ViT_Br3_l1 = self.vitLayer_UpConv(ViT_Br3_l1)
        ViT_Br3_l1 = self.vitLayer_UpConv(ViT_Br3_l1)
        ViT_Br3_l1 = self.vitLayer_UpConv(ViT_Br3_l1)
        ViT_Br3_l1 = self.vitLayer_UpConv(ViT_Br3_l1)  # 这个是倒数第一次也就是第二次的trans

        # Branch1 ->
        Dec_Br1_l4_cat = torch.cat([Dec_Br1_l4, sc4], dim=1)  # [4 1536 28 28]
        Dec_Br1_l_4TO3 = self.upsampleups[1](Dec_Br1_l4_cat)  # 双卷积 Dec_Br1_l_4TO3[4, 512, 28, 28]
        Dec_Br1_l3 = self.upsampleups[2](Dec_Br1_l_4TO3)  # 上采样 Dec_Br1_l3[4, 256, 56, 56]
        # Dec_Br2_l_4TransTO3 = self.e_UpConv[0](Dec_Br2_l4)   #分枝2：第四层到第三层通道不变，大小*2 Dec_Br2_l_4TransTO3[4, 512, 56, 56]
        Dec_Br2_l_4TransTO3 = self.transblcok[0](Dec_Br2_l4, x_aspp)
        ViT_Br3_l_4TransTO3 = self.transblcok[1](ViT_Br3_l4, x_aspp)
        Dec_Br1_l3 = self.e3_LC(e3_up + Dec_Br1_l3)
        Dec_Br1_l_3TO2 = torch.cat([Dec_Br1_l3, ViT_Br3_l_4TransTO3, sc3], dim=1)  # Dec_Br1_l_3TO2:[4,768,56,56]
        # Dec_Br1_l_3TO2 = torch.cat([Dec_Br1_l3, Dec_Br2_l_4TransTO3, sc3], dim=1)  # Dec_Br1_l_3TO2:[4,768,56,56]
        Dec_Br1_l_3TO2 = self.upsampleups[3](Dec_Br1_l_3TO2)  # 双卷积Dec_Br1_l_3TO2[4,256,56,56]
        Dec_Br1_l2 = self.upsampleups[4](Dec_Br1_l_3TO2)  # 上采样Dec_Br1_l2[4,128,112,112]

        # Branch2 ->
        Dec_Br2_l4 = torch.cat([Dec_Br2_l4, sc4], dim=1)
        Dec_Br2_l_4TO3 = self.skups[0](Dec_Br2_l4)  # DoubleConv Dec_Br2_l_4TO3:[4,512,28,28]
        Dec_Br2_l3 = self.skups[1](Dec_Br2_l_4TO3)  # 上采样到第三层 Dec_Br2_l3[4,256,56,56]
        # ViT_Br3_l_4TransTO3 = self.vitLayer_UpConv(ViT_Br3_l4)       #
        # ViT_Br3_l_4TransTO3 = self.transblcok[1](ViT_Br3_l4, x_aspp)
        # Dec_Br2_l_4TransTO3 = self.transblcok[0](Dec_Br2_l4, x_aspp)
        Dec_Br1_l_4TransTO3 = self.transblcok[0](Dec_Br1_l4, x_aspp)
        Dec_Br2_l3 = self.e3_LC(e3_up + Dec_Br2_l3)
        Dec_Br2_l_3TO2 = torch.cat([Dec_Br2_l3, Dec_Br1_l_4TransTO3, sc3], dim=1)  # Dec_Br2_l_3TO2[4,321,56,56]
        # Dec_Br2_l_3TO2 = torch.cat([Dec_Br2_l3, ViT_Br3_l_4TransTO3, sc3], dim=1)  # Dec_Br2_l_3TO2[4,321,56,56]
        Dec_Br2_l_3TO2 = self.skups[2](Dec_Br2_l_3TO2)  # 双卷积Dec_Br2_l_3TO2[4,256,56,56]
        Dec_Br2_l2 = self.skups[3](Dec_Br2_l_3TO2)  # 上采样Dec_Br2_l_2[4,128,112,112]

        # Branch3 ->
        ViT_Br3_l4 = torch.cat([ViT_Br3_l4, sc4], dim=1)
        Dec_Br3_l_4TO3 = self.vitups[0](ViT_Br3_l4)  # 双卷积[4,65,28,28]
        Dec_Br3_l3 = self.vitups[4](Dec_Br3_l_4TO3)  # 上采样[4,65,56,56]
        Dec_Br3_l3 = torch.cat([Dec_Br3_l3, ViT_Br3_l3], dim=1)
        Dec_Br3_l3 = self.trans_conv[0](Dec_Br3_l3)  # 卷积把通道还原为65

        # Dec_Br1_l_4TransTO3 = self.e_UpConv[0](Dec_Br1_l4)     ViT_Br3_l3   #Br1通道不变大小*2 [4,512,56,56]
        # Dec_Br1_l_4TransTO3 = self.transblcok[0](Dec_Br1_l4, x_aspp)
        # Dec_Br2_l_4TransTO3 = self.transblcok[0](Dec_Br2_l4, x_aspp)
        Dec_Br3_l_3TO2 = torch.cat([Dec_Br3_l3, Dec_Br2_l_4TransTO3, sc3], dim=1)  # [4,577,56,56]
        Dec_Br3_l_3TO2 = self.vitups[1](Dec_Br3_l_3TO2)  # 双卷积Dec_Br3_l_3TO2[4,65,56,56]
        Dec_Br3_l2 = self.vitups[4](Dec_Br3_l_3TO2)  # 上采样Dec_Br3_l2[4,65,112,112]

        # Branch1 -> 从Dec_Br1_l2，粘贴来自BR1第三层开始
        # Dec_Br2_3TransTO2 = self.e_UpConv[1](Dec_Br2_l3)      #分枝2：第三层到第而层通道不变，大小*2 [4, 256, 112, 112]
        Dec_Br3_3TransTO2 = self.transblcok[1](Dec_Br3_l3, x_aspp)
        # Dec_Br2_3TransTO2 = self.transblcok[2](Dec_Br2_l3, x_aspp)
        Dec_Br1_l2 = self.e2_LC(e2_up + Dec_Br1_l2)
        Dec_Br1_l_2TO1 = torch.cat([Dec_Br1_l2, Dec_Br3_3TransTO2, sc2], dim=1)  # Dec_Br1_l_2TO1[4,384,112,112]
        # Dec_Br1_l_2TO1 = torch.cat([Dec_Br1_l2, Dec_Br2_3TransTO2, sc2], dim=1)  # Dec_Br1_l_2TO1[4,384,112,112]
        Dec_Br1_l_2TO1 = self.upsampleups[5](Dec_Br1_l_2TO1)  # Dec_Br1_l_2TO1:[4,128,112,112]
        Dec_Br1_l1 = self.upsampleups[6](Dec_Br1_l_2TO1)  # 上采样[4,64,224,224]

        # Branch2 -> 从Dec_Br2_l2，粘贴来自Br3的第3ceng
        # Dec_Br3_3TransTO2 = self.vitLayer_UpConv(Dec_Br3_l3)     #Dec_Br3_3TO2:[4,65,112,112]
        Dec_Br1_l_3TransTO2 = self.transblcok[2](Dec_Br1_l3, x_aspp)
        # Dec_Br3_3TransTO2 = self.transblcok[1](Dec_Br3_l3, x_aspp)
        Dec_Br2_l2 = self.e2_LC(e2_up + Dec_Br2_l2)
        Dec_Br2_2TO1 = torch.cat([Dec_Br2_l2, Dec_Br1_l_3TransTO2, sc2], dim=1)  # [4,193,112,112]
        Dec_Br2_2TO1 = self.skups[4](Dec_Br2_2TO1)  # DoubConv [4,128,112,112]
        Dec_Br2_l1 = self.skups[5](Dec_Br2_2TO1)  # 上采样[4,64,224,224]

        # Branch3 -> 从Dec_Br3_l2，粘贴来自Br1第三层开始
        # Dec_Br1_l_3TransTO2 = self.e_UpConv[1](Dec_Br1_l3)     #[4,256,112,112]
        Dec_Br2_3TransTO2 = self.transblcok[2](Dec_Br2_l3, x_aspp)
        Dec_Br1_l_3TransTO2 = self.transblcok[2](Dec_Br1_l3, x_aspp)
        Dec_Br3_l2 = torch.cat([Dec_Br3_l2, ViT_Br3_l2], dim=1)  # 第二次trans的跳连拼接
        Dec_Br3_l2 = self.trans_conv[0](Dec_Br3_l2)     #通道数还原
        Dec_Br3_l_2TO1 = torch.cat([Dec_Br3_l2, Dec_Br2_3TransTO2, sc2], dim=1)  # [4,321,112,112]
        # Dec_Br3_l_2TO1 = torch.cat([Dec_Br3_l2, Dec_Br1_l_3TransTO2, sc2], dim=1)  # [4,321,112,112]
        Dec_Br3_l_2TO1 = self.vitups[2](Dec_Br3_l_2TO1)  # 双卷积[4,65,112,112]
        Dec_Br3_l1 = self.vitups[5](Dec_Br3_l_2TO1)  # 上采样[4,64,224,224]
        Dec_Br3_l1 = torch.cat([Dec_Br3_l1, ViT_Br3_l1], dim=1)
        Dec_Br3_l1 = self.trans_conv[1](Dec_Br3_l1)  # 上采样后的最后一次拼接

        Dec_Br1_l1 = self.e1_LC(e1_up + Dec_Br1_l1)
        Dec_Br2_l1 = self.e1_LC(e1_up + Dec_Br2_l1)
        ThreeBranches = torch.cat([Dec_Br1_l1, Dec_Br2_l1, Dec_Br3_l1], dim=1)  # [4,192,224,224]
        De_x = self.Three2One(ThreeBranches)  # [4,64,224,224]

        out1 = self.final_conv1(De_x)  # 1*1卷积 反卷积步长为1
        out1 = self.final_relu1(out1)  # [4, 32, 224, 224]
        out = self.final_conv2(out1)  # 3*3卷积 通道32不变
        out = self.final_relu2(out)  # [4, 32, 224, 224]
        out = self.final_conv3(out)  # 3*3卷积 通道32->1
        return out  # [4,1,224,224]                                       #En_x:      [4, 1, 224, 224]


# x = torch.randn(4, 3, 224, 224)
# model = Reversal_TP_MNet()
# preds = model(x)
# print(x.shape)
# print(preds.shape)
# print("Reversal_TP_MNet")

# x = torch.randn(4, 512, 56, 56)
# model = TFRB(512)
# preds = model(x)
