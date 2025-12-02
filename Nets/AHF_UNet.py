# -*- conding: utf-8 -*- #

# -------------------------------------------------------
# @Name:         AHF_UNet
# @Author:       ZhangJiapeng@ZY-301
# @Data:         2024/12/22
# @Time:         17:14
# -------------------------------------------------------
# @Even:         
# @Description:  
# https://github.com/AfsanaAhmedMunia/AHF-Fusion-U-Net/blob/main/AHF_U_Net_(Github).ipynb

# -------------------------------------------------------
import torch
from torch import nn
import torch.nn.functional as F


class AFF(nn.Module):
    def __init__(self, channels, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, outI, feature_Level):
        xa = x + outI + feature_Level # x = I_out4, outI = x5,  feature_Level = x4
        #print("Inside AFF xa: ", xa.size())
        xl = self.local_att(xa)
        #print("Inside AFF xl: ", xl.size())
        xg = self.global_att(xa)
        #print("Inside AFF xg: ", xg.size())
        xlg = xl + xg
        #print("Inside AFF xlg: ", xlg.size())
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * outI * (1 - wei)
        #print("Inside AFF xo: ", wei.size())
        return xo


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
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


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, AFF_channel, bilinear=False):
    #def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.aff = AFF(AFF_channel)
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels) #AFF
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels//2, out_channels)

    def forward(self, x1, x2, x3):
        # print("inside forward up( initial call) x1 and x2: ", x1.size(),x2.size())
        x1 = self.up(x1) # during first call x1 = x5, x2 = x4 (x1, x2, x3 == x5, I_out_4, x4)
        # print("inside after calling up, x1=x5: ", x1.size())
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        #x = self.model1(x2,x1,x3)
        x = self.aff(x2,x1,x3)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
       return self.conv(x)


# class CLSCA(nn.Module):
class UAFF(nn.Module):
    def __init__(self, n_classes=3, n_channels=3, bilinear=False, dropout_rate=0.0):
        # super(CLSCA, self).__init__()
        super(UAFF, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.spatial_avgpool = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.trans_conv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1,
                                              output_padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.trans_conv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1,
                                              output_padding=1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.trans_conv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1,
                                              output_padding=1)
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.trans_conv4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1,
                                              output_padding=1)
        """ For AFF and iAFF"""
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        # factor = 2 if bilinear else 1
        factor = 2
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, 512)
        self.up2 = Up(512, 256, 256)
        self.up3 = Up(256, 128, 128)
        self.up4 = Up(128, 64, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)  # x2: [B 64 288 512]
        s1 = self.spatial_avgpool(x1)
        c1 = torch.mean(x1, dim=1, keepdim=True)
        mul1 = s1 * c1
        # print("Inside UNET Forward: mul1 = ", mul1.size())
        x2 = self.down1(x1)  # x2: [B 128 144 256]
        s2 = self.spatial_avgpool(x2)
        c2 = torch.mean(x2, dim=1, keepdim=True)
        mul2 = s2 * c2
        # print("Inside UNET Forward: mul2  = ", mul2.size())
        Trans_conv1 = self.trans_conv1(mul2)
        I_out_1 = self.conv1(mul1 + Trans_conv1)
        I_out_1 = I_out_1 * x1
        # print("Inside UNET Forward: I_out_1 = ", I_out_1.size())
        x3 = self.down2(x2)  # x3: [B 256 72 128]
        s3 = self.spatial_avgpool(x3)
        c3 = torch.mean(x3, dim=1, keepdim=True)
        mul3 = s3 * c3
        # print("Inside UNET Forward: mul3  = ", mul3.size())
        Trans_conv2 = self.trans_conv2(mul3)
        I_out_2 = self.conv2(mul2 + Trans_conv2)
        I_out_2 = I_out_2 * x2
        # print("Inside UNET Forward: I_out_2 = ", I_out_2.size())
        x4 = self.down3(x3)   # x4: [B 512 36 64]
        # print("Inside UNET Forward: X4 = ", x4.size())
        s4 = self.spatial_avgpool(x4)
        c4 = torch.mean(x4, dim=1, keepdim=True)
        mul4 = s4 * c4
        # print("Inside UNET Forward: mul4  = ", mul4.size())
        Trans_conv3 = self.trans_conv3(mul4)
        # print("Inside UNET Forward: Trans_conv3  = ", Trans_conv3.size())
        I_out_3 = self.conv3(mul3 + Trans_conv3)
        I_out_3 = I_out_3 * x3
        # print("Inside UNET Forward: I_out_3 = ", I_out_3.size())
        x5 = self.down4(x4) # x5: [B 1024 18 32]
        # print("Inside UNET Forward: X5 = ", x5.size())
        s5 = self.spatial_avgpool(x5)
        c5 = torch.mean(x5, dim=1, keepdim=True)
        mul5 = s5 * c5
        # print("Inside UNET Forward: mul5  = ", mul5.size())
        Trans_conv4 = self.trans_conv4(mul5)
        I_out_4 = self.conv4(mul4 + Trans_conv4)
        I_out_4 = I_out_4 * x4
        # print("Inside UNET Forward: I_out_4 = ", I_out_4.size())
        x = self.up1(x5, I_out_4, x4)
        # print("x after up1: up1(x5, x4) = ", x.size())
        x = self.up2(x, I_out_3, x3)
        # print("x: up2(x, x3) = ", x.size())
        x = self.up3(x, I_out_2, x2)
        # print("x: up3(x, x2) = ", x.size())
        x = self.up4(x, I_out_1, x1)
        # print("x: up4(x, x1) = ", x.size())
        logits = self.outc(x)
        # print("outc(x) = ", logits.size())

        return logits

if __name__ == '__main__':
    # cal the Params and the FLOPs
    from torchsummary import summary
    from thop import profile

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UAFF(n_classes=3, n_channels=3).to(device)

    summary(model, (3, 288, 512))
    input = torch.randn(1, 3, 288, 512).to(device)
    flops, params = profile(model, inputs=(input,))
    flops_in_mb = flops / 1e6
    flops_in_gb = flops / 1e9
    params_in_mb = params / 1e6
    print(f"FLOPs: {flops} ({flops_in_mb:.2f} MFLOPs, {flops_in_gb:.2f} GFLOPs)")  #
    print(f"Parameters: {params}({params_in_mb:.2f} M)")  #
    end = 'end'
