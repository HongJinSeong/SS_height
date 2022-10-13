import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import get_graph_node_names
import timm
from torchvision.models.feature_extraction import create_feature_extractor


class Scale_invariant_logloss(nn.Module):
    """Scale invariant log loss"""

    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduction='mean')
        self.lam = 0.85

    def forward(self, x, target):
        pix_num = x.shape[0] * x.shape[2] * x.shape[3]
        x = torch.log(x + 1e-7)
        target = torch.log(target + 1e-7)

        ou = target - x
        ou = torch.sum(ou)
        loss = torch.sqrt(self.mse(x, target) + ((self.lam * (ou * ou)) / (pix_num * pix_num)))

        return loss


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(True),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Double_stride(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        return self.double_conv(x)


class Res(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        return self.double_conv(x) + x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return self.conv(x)


# channel-wise attention ==> UNET용
class CA(nn.Module):
    def __init__(self, in_channel):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(4, in_channel // 4, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(in_channel // 4, in_channel // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, cls_out):
        # channel wise attention (SE-Net)
        b, c, _, _ = x.size()

        y = self.fc(cls_out).view(b, c, 1, 1)
        cout = x * y.expand_as(x)
        return cout + x


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        # norm_layer(oup),
        nlin_layer(inplace=True)
    )


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Hardsigmoid(inplace=True)
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU  # or ReLU6
        elif nl == 'HS':
            nlin_layer = nn.Hardswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = nn.Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


## classifer for 4class (외곽 영역 4가지 값 고정적으로 나오도록 set하기 위함)
class classifier(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(classifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=7, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.MBN6 = MobileBottleneck(inp=32, oup=48, exp=256, se=True, nl='HS', kernel=5, stride=2)
        self.MBN7 = MobileBottleneck(inp=48, oup=96, exp=256, se=True, nl='HS', kernel=5, stride=1)

        self.conv1x1_BN = conv_1x1_bn(inp=96, oup=256, nlin_layer=nn.Hardswish)
        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.conv1x1_NBN_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        self.hardswish1 = nn.Hardswish()

        self.FC = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.conv1(x)

        out = self.MBN6(x)
        out = self.MBN7(out)

        out = self.conv1x1_BN(out)
        out = self.GAP(out)

        out = self.conv1x1_NBN_1(out)
        out = self.hardswish1(out)

        out = nn.Flatten()(out)

        clsoutput = self.FC(out)

        return clsoutput


## U-Net
## test해볼떄 파라미터 과다가 되면 val score가 높아졌다 낮아졌다 학습이 엉망이였음
## 현재 best는 시작 channel 16일때 부터 x2 scailing
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.CA1 = CA(128)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.CA2 = CA(64)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.CA3 = CA(32)
        self.up4 = Up(32, 16, bilinear)
        self.CA4 = CA(32)
        self.outc = OutConv(16, n_classes, 5, 2)

        ## case 별로 depth map의 외곽값이 140 150 160 170이기 때문에 이를 class 정보로 주기위함 (내부는 몰라도 외부는 잘 판별할수 있도록 set)
        ## 우선 classifer로 예측한 정보를 fc거쳐서 channel-wise attention 형태로 반영
        ## 해보고 별로면 classfier값을 단순하게 이용해보기
        self.classifier = nn.Linear(3072, 4)

        self.onehot = torch.eye(4, dtype=torch.int)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        clsoutput = self.classifier(x5.flatten(1))

        x = self.up1(x5, x4)
        x = self.CA1(x, clsoutput)
        x = self.up2(x, x3)
        x = self.CA2(x, clsoutput)
        x = self.up3(x, x2)
        x = self.CA3(x, clsoutput)
        x = self.up4(x, x1)
        x = self.CA4(x, clsoutput)

        # -1 ~ 1 scailing
        # logits = nn.Tanh()(self.outc(x))
        # 0 ~ 1 scailing
        logits = nn.Sigmoid()(self.outc(x))
        return logits, clsoutput


# 기존것보다 성능안좋음
class Unet_trans(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Unet_trans, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)

        # 최초 decoder 시작부분 기준으로 positional encoding 진행
        # self.up1 = nn.Conv2d(272, 160 , kernel_size=1)
        self.up1 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1),
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.posi_enco = PositionEncodingSine(128)
        self.self_attention1 = EncoderLayer(128, 16, 'full')
        self.cross_attention1 = EncoderLayer(128, 16, 'full')

        # self.up2 = nn.ConvTranspose2d(160, 56, kernel_size=2, stride=2)
        self.up2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1),
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.self_attention2 = EncoderLayer(64, 8, 'full')
        self.cross_attention2 = EncoderLayer(64, 8, 'full')

        # self.up3 = nn.ConvTranspose2d(56, 32, kernel_size=2, stride=2)
        self.self_attention3 = EncoderLayer(32, 8, 'full')
        self.cross_attention3 = EncoderLayer(32, 8, 'full')

        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)

        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.up1(x5)
        x5 = self.posi_enco(x5).permute(0, 2, 3, 1)
        n, h, w, c = x5.shape
        x5 = x5.reshape(n, -1, c)
        x5 = self.self_attention1(x5, x5)

        x4 = x4.permute(0, 2, 3, 1)
        n, h, w, c = x4.shape
        x4 = x4.reshape(n, -1, c)
        x4 = self.cross_attention1(x5, x4)

        x4 = x4.reshape(n, h, w, c)
        x4 = x4.permute(0, 3, 1, 2)
        x4 = self.up2(x4)
        x4 = x4.permute(0, 2, 3, 1)
        n, h, w, c = x4.shape
        x4 = x4.reshape(n, -1, c)
        x4 = self.self_attention2(x4, x4)

        x3 = x3.permute(0, 2, 3, 1)
        n, h, w, c = x3.shape
        x3 = x3.reshape(n, -1, c)
        x3 = self.cross_attention2(x4, x3)

        x3 = x3.reshape(n, h, w, c)
        x3 = x3.permute(0, 3, 1, 2)

        x = self.up3(x3, x2)

        x = self.up4(x, x1)

        logits = nn.Sigmoid()(self.outc(x))

        return logits


##오히려 안좋아짐 왠진 모름 ..
class Attention_for_skipconnection(nn.Module):
    def __init__(self, channel, reduction=4):
        super(Attention_for_skipconnection, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # channel wise attention (SE-Net)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        cout = x * y.expand_as(x)

        # spatial wise attention (CBAM)
        y = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        y = self.conv(y)
        sout = x * y.expand_as(x)

        return cout + sout + x


# skip connection feature에 attention 추가하기
class UNet_attention(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_attention, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.att1 = Attention_for_skipconnection(128)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.att2 = Attention_for_skipconnection(64)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.att3 = Attention_for_skipconnection(32)
        self.up4 = Up(32, 16, bilinear)
        self.att4 = Attention_for_skipconnection(16)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x4 = self.att1(x4)
        x = self.up1(x5, x4)

        x3 = self.att2(x3)
        x = self.up2(x, x3)

        x2 = self.att3(x2)
        x = self.up3(x, x2)

        x1 = self.att4(x1)
        x = self.up4(x, x1)
        # -1 ~ 1 scailing
        # logits = nn.Tanh()(self.outc(x))
        # 0 ~ 1 scailing
        logits = nn.Sigmoid()(self.outc(x))
        return logits


def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()


# class CA(nn.Module):
#     def __init__(self, in_channel):
#         super(CA, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.embed = nn.Embedding(4,in_channel//4)
#         self.fc = nn.Sequential(
#             nn.Linear(in_channel*2, in_channel, bias=False),
#             nn.Sigmoid()
#         )
#
#
#     def forward(self, x,emb):
#         # channel wise attention (SE-Net)
#         b, c, _, _ = x.size()
#
#         embedoutput = self.embed(emb).view(b,-1)
#         y = self.avg_pool(x).view(b, c)
#         y = torch.cat((y,embedoutput),dim=1)
#         y = self.fc(y).view(b, c, 1, 1)
#         cout = x * y.expand_as(x)
#         return cout+x


# MFF (multi-scale feature fusion module)
class MFF_depth(nn.Module):
    def __init__(self):
        super(MFF_depth, self).__init__()
        base = timm.create_model('efficientnet_b1', pretrained=True)
        train_nodes, eval_nodes = get_graph_node_names(base)
        return_nodes = {
            train_nodes[10]: 'f1',
            train_nodes[64]: 'f2',
            train_nodes[176]: 'f3',
            train_nodes[288]: 'f4',
            train_nodes[438]: 'f5',
        }

        self.backbone = create_feature_extractor(base, return_nodes)

        self.decoder1 = nn.Sequential(
            nn.Conv2d(80, 72, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.BatchNorm2d(72, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU()
        )
        self.CA1 = CA(72)

        self.decoder2 = nn.Sequential(
            nn.Conv2d(72, 64, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU()
        )
        self.CA2 = CA(64)

        self.decoder3 = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.BatchNorm2d(48, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU()
        )
        self.CA3 = CA(48)

        self.decoder4 = nn.Sequential(
            nn.Conv2d(48, 40, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.BatchNorm2d(40, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU()
        )
        self.CA4 = CA(40)

        self.UP1 = nn.Sequential(nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1)
                                 , nn.Upsample(scale_factor=2, mode='bilinear')
                                 , nn.BatchNorm2d(8, eps=1e-5, momentum=0.01, affine=True)
                                 , nn.ReLU())
        self.UP2 = nn.Sequential(nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
                                 , nn.Upsample(scale_factor=2, mode='bilinear')
                                 , nn.BatchNorm2d(8, eps=1e-5, momentum=0.01, affine=True)
                                 , nn.ReLU())
        self.UP3 = nn.Sequential(nn.Conv2d(24, 8, kernel_size=3, stride=1, padding=1)
                                 , nn.Upsample(scale_factor=4, mode='bilinear')
                                 , nn.BatchNorm2d(8, eps=1e-5, momentum=0.01, affine=True)
                                 , nn.ReLU())
        self.UP4 = nn.Sequential(nn.Conv2d(40, 8, kernel_size=3, stride=1, padding=1)
                                 , nn.Upsample(scale_factor=8, mode='bilinear')
                                 , nn.BatchNorm2d(8, eps=1e-5, momentum=0.01, affine=True)
                                 , nn.ReLU())
        self.UP5 = nn.Sequential(nn.Conv2d(80, 8, kernel_size=3, stride=1, padding=1)
                                 , nn.Upsample(scale_factor=16, mode='bilinear')
                                 , nn.BatchNorm2d(8, eps=1e-5, momentum=0.01, affine=True)
                                 , nn.ReLU())

        self.Conv1 = nn.Sequential(nn.Conv2d(40, 40, kernel_size=3, stride=1, padding=1)
                                   , nn.BatchNorm2d(40, eps=1e-5, momentum=0.01, affine=True)
                                   , nn.ReLU())
        self.Conv2 = nn.Sequential(nn.Conv2d(80, 80, kernel_size=3, stride=1, padding=1)
                                   , nn.BatchNorm2d(80, eps=1e-5, momentum=0.01, affine=True)
                                   , nn.ReLU())
        self.Conv3 = nn.Sequential(nn.Conv2d(80, 80, kernel_size=3, stride=1, padding=1)
                                   , nn.BatchNorm2d(80, eps=1e-5, momentum=0.01, affine=True)
                                   , nn.ReLU())

        self.Conv4 = nn.Sequential(nn.Conv2d(80, 1, kernel_size=3, stride=1, padding=1)
                                   , nn.Sigmoid())

        ## case 별로 depth map의 외곽값이 140 150 160 170이기 때문에 이를 class 정보로 주기위함 (내부는 몰라도 외부는 잘 판별할수 있도록 set)
        ## 우선 classifer로 예측한 정보를 one-hot 형태로 바꾸고 embedding하여 channel-wise attention 형태로 decoder,up, conv에 반영해보도록 진행
        ## 해보고 별로면 classfier값을 단순하게 이용해보기
        # self.classifier = nn.Linear(1920,4)
        self.classifier = classifier()

    def forward(self, x):
        x = self.backbone(x)

        # classify_output = self.classifier(x['f5'].flatten(1))
        # onehot = torch.nn.functional.one_hot(torch.argmax(classify_output, dim=1)).detach()

        classify_output = self.classifier(x['f5'])

        y = self.decoder1(x['f5'].clone())
        # y = self.CA1(y, onehot)
        y = self.decoder2(y)
        # y = self.CA2(y, onehot)
        y = self.decoder3(y)
        # y = self.CA3(y, onehot)
        y = self.decoder4(y)
        # y = self.CA4(y, onehot)

        x1 = self.UP1(x['f1'])
        x2 = self.UP2(x['f2'])
        x3 = self.UP3(x['f3'])
        x4 = self.UP4(x['f4'])
        x5 = self.UP5(x['f5'].clone())

        x_mff = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x_mff = self.Conv1(x_mff)

        y = torch.cat((x_mff, y), dim=1)

        y = self.Conv2(y)
        y = self.Conv3(y)
        y = self.Conv4(y)

        return y, classify_output


# simple autoencoder
class Encoder_decoder(nn.Module):
    def __init__(self):
        super(Encoder_decoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class BaseModel(nn.Module):
    def __init__(self, CFG):
        super(BaseModel, self).__init__()
        self.CFG = CFG
        self.encoder = nn.Sequential(
            nn.Linear(CFG['HEIGHT'] * CFG['WIDTH'], 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, CFG['HEIGHT'] * CFG['WIDTH']),
        )

    def forward(self, x):
        x = x.view(-1, self.CFG['HEIGHT'] * self.CFG['WIDTH'])
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1, 1, self.CFG['HEIGHT'], self.CFG['WIDTH'])
        return x


## U-Net
## test해볼떄 파라미터 과다가 되면 val score가 높아졌다 낮아졌다 학습이 엉망이였음
## 현재 best는 시작 channel 16일때 부터 x2 scailing

class UNet_with_embed(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_with_embed, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Double_stride(16, 32)
        self.down2 = Double_stride(32, 64)
        self.down3 = Double_stride(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Double_stride(128, 256)

        self.res1 = Res(256, 256)
        self.res2 = Res(256, 256)
        self.res3 = Res(256, 256)
        self.res4 = Res(256, 256)

        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes, 5, 2)

        ## case 별로 depth map의 외곽값이 140 150 160 170이기 때문에 이를 class 정보로 주기위함 (내부는 몰라도 외부는 잘 판별할수 있도록 set)
        ## 우선 classifer로 예측한 정보를 fc거쳐서 channel-wise attention 형태로 반영
        ## 해보고 별로면 classfier값을 단순하게 이용해보기
        self.classifier = nn.Linear(3072, 4)

        self.embed = nn.Embedding(4, 32)
        self.embed_fc = nn.Linear(128, 6144)

    def forward(self, x, onehot):
        b, c, h, w = x.shape

        embed_output = self.embed(onehot)
        embed_output = self.embed_fc(embed_output.flatten(1))
        embed_output = embed_output.reshape(b, c, h, w)

        x = torch.cat((x, embed_output), dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.res1(x5)
        x5 = self.res2(x5)
        x5 = self.res3(x5)
        x5 = self.res4(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # -1 ~ 1 scailing
        # logits = nn.Tanh()(self.outc(x))
        # 0 ~ 1 scailing
        logits = nn.Sigmoid()(self.outc(x))
        return logits


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import get_graph_node_names
import timm
from torchvision.models.feature_extraction import create_feature_extractor


class Scale_invariant_logloss(nn.Module):
    """Scale invariant log loss"""

    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduction='mean')
        self.lam = 0.85

    def forward(self, x, target):
        pix_num = x.shape[0] * x.shape[2] * x.shape[3]
        x = torch.log(x + 1e-7)
        target = torch.log(target + 1e-7)

        ou = target - x
        ou = torch.sum(ou)
        loss = torch.sqrt(self.mse(x, target) + ((self.lam * (ou * ou)) / (pix_num * pix_num)))

        return loss


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(True),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Double_stride(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        return self.double_conv(x)


class Res(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        return self.double_conv(x) + x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return self.conv(x)


# channel-wise attention ==> UNET용
class CA(nn.Module):
    def __init__(self, in_channel):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(4, in_channel // 4, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(in_channel // 4, in_channel // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, cls_out):
        # channel wise attention (SE-Net)
        b, c, _, _ = x.size()

        y = self.fc(cls_out).view(b, c, 1, 1)
        cout = x * y.expand_as(x)
        return cout + x


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        # norm_layer(oup),
        nlin_layer(inplace=True)
    )


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Hardsigmoid(inplace=True)
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU  # or ReLU6
        elif nl == 'HS':
            nlin_layer = nn.Hardswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = nn.Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


## classifer for 4class (외곽 영역 4가지 값 고정적으로 나오도록 set하기 위함)
class classifier(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(classifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=7, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.MBN6 = MobileBottleneck(inp=32, oup=48, exp=256, se=True, nl='HS', kernel=5, stride=2)
        self.MBN7 = MobileBottleneck(inp=48, oup=96, exp=256, se=True, nl='HS', kernel=5, stride=1)

        self.conv1x1_BN = conv_1x1_bn(inp=96, oup=256, nlin_layer=nn.Hardswish)
        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.conv1x1_NBN_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        self.hardswish1 = nn.Hardswish()

        self.FC = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.conv1(x)

        out = self.MBN6(x)
        out = self.MBN7(out)

        out = self.conv1x1_BN(out)
        out = self.GAP(out)

        out = self.conv1x1_NBN_1(out)
        out = self.hardswish1(out)

        out = nn.Flatten()(out)

        clsoutput = self.FC(out)

        return clsoutput


## U-Net
## test해볼떄 파라미터 과다가 되면 val score가 높아졌다 낮아졌다 학습이 엉망이였음
## 현재 best는 시작 channel 16일때 부터 x2 scailing
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.CA1 = CA(128)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.CA2 = CA(64)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.CA3 = CA(32)
        self.up4 = Up(32, 16, bilinear)
        self.CA4 = CA(32)
        self.outc = OutConv(16, n_classes, 5, 2)

        ## case 별로 depth map의 외곽값이 140 150 160 170이기 때문에 이를 class 정보로 주기위함 (내부는 몰라도 외부는 잘 판별할수 있도록 set)
        ## 우선 classifer로 예측한 정보를 fc거쳐서 channel-wise attention 형태로 반영
        ## 해보고 별로면 classfier값을 단순하게 이용해보기
        self.classifier = nn.Linear(3072, 4)

        self.onehot = torch.eye(4, dtype=torch.int)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        clsoutput = self.classifier(x5.flatten(1))

        x = self.up1(x5, x4)
        x = self.CA1(x, clsoutput)
        x = self.up2(x, x3)
        x = self.CA2(x, clsoutput)
        x = self.up3(x, x2)
        x = self.CA3(x, clsoutput)
        x = self.up4(x, x1)
        x = self.CA4(x, clsoutput)

        # -1 ~ 1 scailing
        # logits = nn.Tanh()(self.outc(x))
        # 0 ~ 1 scailing
        logits = nn.Sigmoid()(self.outc(x))
        return logits, clsoutput


# 기존것보다 성능안좋음
class Unet_trans(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Unet_trans, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)

        # 최초 decoder 시작부분 기준으로 positional encoding 진행
        # self.up1 = nn.Conv2d(272, 160 , kernel_size=1)
        self.up1 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1),
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.posi_enco = PositionEncodingSine(128)
        self.self_attention1 = EncoderLayer(128, 16, 'full')
        self.cross_attention1 = EncoderLayer(128, 16, 'full')

        # self.up2 = nn.ConvTranspose2d(160, 56, kernel_size=2, stride=2)
        self.up2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1),
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.self_attention2 = EncoderLayer(64, 8, 'full')
        self.cross_attention2 = EncoderLayer(64, 8, 'full')

        # self.up3 = nn.ConvTranspose2d(56, 32, kernel_size=2, stride=2)
        self.self_attention3 = EncoderLayer(32, 8, 'full')
        self.cross_attention3 = EncoderLayer(32, 8, 'full')

        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)

        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.up1(x5)
        x5 = self.posi_enco(x5).permute(0, 2, 3, 1)
        n, h, w, c = x5.shape
        x5 = x5.reshape(n, -1, c)
        x5 = self.self_attention1(x5, x5)

        x4 = x4.permute(0, 2, 3, 1)
        n, h, w, c = x4.shape
        x4 = x4.reshape(n, -1, c)
        x4 = self.cross_attention1(x5, x4)

        x4 = x4.reshape(n, h, w, c)
        x4 = x4.permute(0, 3, 1, 2)
        x4 = self.up2(x4)
        x4 = x4.permute(0, 2, 3, 1)
        n, h, w, c = x4.shape
        x4 = x4.reshape(n, -1, c)
        x4 = self.self_attention2(x4, x4)

        x3 = x3.permute(0, 2, 3, 1)
        n, h, w, c = x3.shape
        x3 = x3.reshape(n, -1, c)
        x3 = self.cross_attention2(x4, x3)

        x3 = x3.reshape(n, h, w, c)
        x3 = x3.permute(0, 3, 1, 2)

        x = self.up3(x3, x2)

        x = self.up4(x, x1)

        logits = nn.Sigmoid()(self.outc(x))

        return logits


##오히려 안좋아짐 왠진 모름 ..
class Attention_for_skipconnection(nn.Module):
    def __init__(self, channel, reduction=4):
        super(Attention_for_skipconnection, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # channel wise attention (SE-Net)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        cout = x * y.expand_as(x)

        # spatial wise attention (CBAM)
        y = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        y = self.conv(y)
        sout = x * y.expand_as(x)

        return cout + sout + x


# skip connection feature에 attention 추가하기
class UNet_attention(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_attention, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.att1 = Attention_for_skipconnection(128)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.att2 = Attention_for_skipconnection(64)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.att3 = Attention_for_skipconnection(32)
        self.up4 = Up(32, 16, bilinear)
        self.att4 = Attention_for_skipconnection(16)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x4 = self.att1(x4)
        x = self.up1(x5, x4)

        x3 = self.att2(x3)
        x = self.up2(x, x3)

        x2 = self.att3(x2)
        x = self.up3(x, x2)

        x1 = self.att4(x1)
        x = self.up4(x, x1)
        # -1 ~ 1 scailing
        # logits = nn.Tanh()(self.outc(x))
        # 0 ~ 1 scailing
        logits = nn.Sigmoid()(self.outc(x))
        return logits


def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()


# class CA(nn.Module):
#     def __init__(self, in_channel):
#         super(CA, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.embed = nn.Embedding(4,in_channel//4)
#         self.fc = nn.Sequential(
#             nn.Linear(in_channel*2, in_channel, bias=False),
#             nn.Sigmoid()
#         )
#
#
#     def forward(self, x,emb):
#         # channel wise attention (SE-Net)
#         b, c, _, _ = x.size()
#
#         embedoutput = self.embed(emb).view(b,-1)
#         y = self.avg_pool(x).view(b, c)
#         y = torch.cat((y,embedoutput),dim=1)
#         y = self.fc(y).view(b, c, 1, 1)
#         cout = x * y.expand_as(x)
#         return cout+x


# MFF (multi-scale feature fusion module)
class MFF_depth(nn.Module):
    def __init__(self):
        super(MFF_depth, self).__init__()
        base = timm.create_model('efficientnet_b1', pretrained=True)
        train_nodes, eval_nodes = get_graph_node_names(base)
        return_nodes = {
            train_nodes[10]: 'f1',
            train_nodes[64]: 'f2',
            train_nodes[176]: 'f3',
            train_nodes[288]: 'f4',
            train_nodes[438]: 'f5',
        }

        self.backbone = create_feature_extractor(base, return_nodes)

        self.decoder1 = nn.Sequential(
            nn.Conv2d(80, 72, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.BatchNorm2d(72, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU()
        )
        self.CA1 = CA(72)

        self.decoder2 = nn.Sequential(
            nn.Conv2d(72, 64, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU()
        )
        self.CA2 = CA(64)

        self.decoder3 = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.BatchNorm2d(48, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU()
        )
        self.CA3 = CA(48)

        self.decoder4 = nn.Sequential(
            nn.Conv2d(48, 40, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.BatchNorm2d(40, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU()
        )
        self.CA4 = CA(40)

        self.UP1 = nn.Sequential(nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1)
                                 , nn.Upsample(scale_factor=2, mode='bilinear')
                                 , nn.BatchNorm2d(8, eps=1e-5, momentum=0.01, affine=True)
                                 , nn.ReLU())
        self.UP2 = nn.Sequential(nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
                                 , nn.Upsample(scale_factor=2, mode='bilinear')
                                 , nn.BatchNorm2d(8, eps=1e-5, momentum=0.01, affine=True)
                                 , nn.ReLU())
        self.UP3 = nn.Sequential(nn.Conv2d(24, 8, kernel_size=3, stride=1, padding=1)
                                 , nn.Upsample(scale_factor=4, mode='bilinear')
                                 , nn.BatchNorm2d(8, eps=1e-5, momentum=0.01, affine=True)
                                 , nn.ReLU())
        self.UP4 = nn.Sequential(nn.Conv2d(40, 8, kernel_size=3, stride=1, padding=1)
                                 , nn.Upsample(scale_factor=8, mode='bilinear')
                                 , nn.BatchNorm2d(8, eps=1e-5, momentum=0.01, affine=True)
                                 , nn.ReLU())
        self.UP5 = nn.Sequential(nn.Conv2d(80, 8, kernel_size=3, stride=1, padding=1)
                                 , nn.Upsample(scale_factor=16, mode='bilinear')
                                 , nn.BatchNorm2d(8, eps=1e-5, momentum=0.01, affine=True)
                                 , nn.ReLU())

        self.Conv1 = nn.Sequential(nn.Conv2d(40, 40, kernel_size=3, stride=1, padding=1)
                                   , nn.BatchNorm2d(40, eps=1e-5, momentum=0.01, affine=True)
                                   , nn.ReLU())
        self.Conv2 = nn.Sequential(nn.Conv2d(80, 80, kernel_size=3, stride=1, padding=1)
                                   , nn.BatchNorm2d(80, eps=1e-5, momentum=0.01, affine=True)
                                   , nn.ReLU())
        self.Conv3 = nn.Sequential(nn.Conv2d(80, 80, kernel_size=3, stride=1, padding=1)
                                   , nn.BatchNorm2d(80, eps=1e-5, momentum=0.01, affine=True)
                                   , nn.ReLU())

        self.Conv4 = nn.Sequential(nn.Conv2d(80, 1, kernel_size=3, stride=1, padding=1)
                                   , nn.Sigmoid())

        ## case 별로 depth map의 외곽값이 140 150 160 170이기 때문에 이를 class 정보로 주기위함 (내부는 몰라도 외부는 잘 판별할수 있도록 set)
        ## 우선 classifer로 예측한 정보를 one-hot 형태로 바꾸고 embedding하여 channel-wise attention 형태로 decoder,up, conv에 반영해보도록 진행
        ## 해보고 별로면 classfier값을 단순하게 이용해보기
        # self.classifier = nn.Linear(1920,4)
        self.classifier = classifier()

    def forward(self, x):
        x = self.backbone(x)

        # classify_output = self.classifier(x['f5'].flatten(1))
        # onehot = torch.nn.functional.one_hot(torch.argmax(classify_output, dim=1)).detach()

        classify_output = self.classifier(x['f5'])

        y = self.decoder1(x['f5'].clone())
        # y = self.CA1(y, onehot)
        y = self.decoder2(y)
        # y = self.CA2(y, onehot)
        y = self.decoder3(y)
        # y = self.CA3(y, onehot)
        y = self.decoder4(y)
        # y = self.CA4(y, onehot)

        x1 = self.UP1(x['f1'])
        x2 = self.UP2(x['f2'])
        x3 = self.UP3(x['f3'])
        x4 = self.UP4(x['f4'])
        x5 = self.UP5(x['f5'].clone())

        x_mff = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x_mff = self.Conv1(x_mff)

        y = torch.cat((x_mff, y), dim=1)

        y = self.Conv2(y)
        y = self.Conv3(y)
        y = self.Conv4(y)

        return y, classify_output


# simple autoencoder
class Encoder_decoder(nn.Module):
    def __init__(self):
        super(Encoder_decoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class BaseModel(nn.Module):
    def __init__(self, CFG):
        super(BaseModel, self).__init__()
        self.CFG = CFG
        self.encoder = nn.Sequential(
            nn.Linear(CFG['HEIGHT'] * CFG['WIDTH'], 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, CFG['HEIGHT'] * CFG['WIDTH']),
        )

    def forward(self, x):
        x = x.view(-1, self.CFG['HEIGHT'] * self.CFG['WIDTH'])
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1, 1, self.CFG['HEIGHT'], self.CFG['WIDTH'])
        return x


## U-Net
## test해볼떄 파라미터 과다가 되면 val score가 높아졌다 낮아졌다 학습이 엉망이였음
## 현재 best는 시작 channel 16일때 부터 x2 scailing

class UNet_with_embed(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_with_embed, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Double_stride(16, 32)
        self.down2 = Double_stride(32, 64)
        self.down3 = Double_stride(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Double_stride(128, 256)

        self.res1 = Res(256, 256)
        self.res2 = Res(256, 256)
        self.res3 = Res(256, 256)
        self.res4 = Res(256, 256)

        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes, 5, 2)

        ## case 별로 depth map의 외곽값이 140 150 160 170이기 때문에 이를 class 정보로 주기위함 (내부는 몰라도 외부는 잘 판별할수 있도록 set)
        ## 우선 classifer로 예측한 정보를 fc거쳐서 channel-wise attention 형태로 반영
        ## 해보고 별로면 classfier값을 단순하게 이용해보기
        self.classifier = nn.Linear(3072, 4)

        self.embed = nn.Embedding(4, 32)
        self.embed_fc = nn.Linear(128, 6144)

    def forward(self, x, onehot):
        b, c, h, w = x.shape

        embed_output = self.embed(onehot)
        embed_output = self.embed_fc(embed_output.flatten(1))
        embed_output = embed_output.reshape(b, c, h, w)

        x = torch.cat((x, embed_output), dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.res1(x5)
        x5 = self.res2(x5)
        x5 = self.res3(x5)
        x5 = self.res4(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # -1 ~ 1 scailing
        # logits = nn.Tanh()(self.outc(x))
        # 0 ~ 1 scailing
        logits = nn.Sigmoid()(self.outc(x))
        return logits
