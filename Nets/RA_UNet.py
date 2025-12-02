# -*- conding: utf-8 -*- #

# -------------------------------------------------------
# @Name:         RA_UNet
# @Author:       ZhangJiapeng@ZY-301
# @Data:         2025/1/21
# @Time:         14:07
# -------------------------------------------------------
# @Even:         
# @Description:  
# https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2023.1201942/full
# -------------------------------------------------------
import torch
from torch import nn
from torchvision.models import resnet50


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
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


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn5 = nn.BatchNorm2d(out_channels)

        self.conv6 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        self.bn6 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x)))
        x3 = self.relu(self.bn3(self.conv3(x)))
        x4 = self.relu(self.bn4(self.conv4(x)))

        x5 = self.global_avg_pool(x)
        x5 = self.relu(self.bn5(self.conv5(x5)))
        x5 = nn.functional.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.relu(self.bn6(self.conv6(x)))

        return x


class RA_UNet(nn.Module):
    def __init__(self, num_classes):
        super(RA_UNet, self).__init__()

        self.layer0_main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1_main = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            nn.Sequential(
                Bottleneck(64, 64, 1,
                           nn.Sequential(
                                nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(256)
                           ),
                           1, 64, 1, nn.BatchNorm2d),
                Bottleneck(256, 64, 1, None, 1, 64, 1, nn.BatchNorm2d),
                Bottleneck(256, 64, 1, None, 1, 64, 1, nn.BatchNorm2d),
            )
        )

        self.layer2_main = nn.Sequential(
            Bottleneck(256, 128, 2,
                          nn.Sequential(
                            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
                            nn.BatchNorm2d(512)
                          ),
                          1, 64, 1, nn.BatchNorm2d),
            Bottleneck(512, 128, 1, None, 1, 64, 1, nn.BatchNorm2d),
            Bottleneck(512, 128, 1, None, 1, 64, 1, nn.BatchNorm2d),
            Bottleneck(512, 128, 1, None, 1, 64, 1, nn.BatchNorm2d),
        )

        self.layer3_main = nn.Sequential(
            Bottleneck(512, 256, 2,
                          nn.Sequential(
                            nn.Conv2d(512, 1024, kernel_size=1, stride=2, bias=False),
                            nn.BatchNorm2d(1024)
                          ),
                          1, 64, 1, nn.BatchNorm2d),
            Bottleneck(1024, 256, 1, None, 1, 64, 1, nn.BatchNorm2d),
            Bottleneck(1024, 256, 1, None, 1, 64, 1, nn.BatchNorm2d),
            Bottleneck(1024, 256, 1, None, 1, 64, 1, nn.BatchNorm2d),
            Bottleneck(1024, 256, 1, None, 1, 64, 1, nn.BatchNorm2d),
            Bottleneck(1024, 256, 1, None, 1, 64, 1, nn.BatchNorm2d),
        )

        self.layer4_main = nn.Sequential(
            Bottleneck(1024, 512, 2,
                            nn.Sequential(
                                nn.Conv2d(1024, 2048, kernel_size=1, stride=2, bias=False),
                                nn.BatchNorm2d(2048)
                            ),
                            1, 64, 1, nn.BatchNorm2d),
            Bottleneck(2048, 512, 1, None, 1, 64, 1, nn.BatchNorm2d),
            Bottleneck(2048, 512, 1, None, 1, 64, 1, nn.BatchNorm2d),
        )

        self.decoder1 = nn.Sequential(
            nn.Conv2d(2048 + 1024, 2048, kernel_size=3, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, bias=False),
            nn.ReLU(inplace=True),
            ASPP(1024, 1024),
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(1024 + 512, 1024, kernel_size=3, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, bias=False),
            nn.ReLU(inplace=True),
            ASPP(512, 512),
        )

        self.decoder3 = nn.Sequential(
            nn.Conv2d(512 + 256, 512, kernel_size=3, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, bias=False),
            nn.ReLU(inplace=True),
            ASPP(256, 256),
        )

        self.decoder4 = nn.Sequential(
            nn.Conv2d(256 + 64, 256, kernel_size=3, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, stride=1, bias=False),
            nn.ReLU(inplace=True),
            ASPP(64, 64),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, rbg):
        rgb_x0 = self.layer0_main(rbg)
        rgb_x1 = self.layer1_main(rgb_x0)
        rgb_x2 = self.layer2_main(rgb_x1)
        rgb_x3 = self.layer3_main(rgb_x2)
        rgb_x4 = self.layer4_main(rgb_x3)

        rgb_x41 = nn.functional.interpolate(rgb_x4, size=rgb_x3.size()[2:], mode='bilinear', align_corners=True)
        rgb_x41 = self.decoder1(torch.cat((rgb_x41, rgb_x3), dim=1))

        rgb_x31 = nn.functional.interpolate(rgb_x41, size=rgb_x2.size()[2:], mode='bilinear', align_corners=True)
        rgb_x31 = self.decoder2(torch.cat((rgb_x31, rgb_x2), dim=1))

        rgb_x21 = nn.functional.interpolate(rgb_x31, size=rgb_x1.size()[2:], mode='bilinear', align_corners=True)
        rgb_x21 = self.decoder3(torch.cat((rgb_x21, rgb_x1), dim=1))

        rgb_x11 = nn.functional.interpolate(rgb_x21, size=rgb_x0.size()[2:], mode='bilinear', align_corners=True)
        rgb_x11 = self.decoder4(torch.cat((rgb_x11, rgb_x0), dim=1))

        rgb_x01 = nn.functional.interpolate(rgb_x11, size=rbg.size()[2:], mode='bilinear', align_corners=True)
        return rgb_x01


if __name__=='__main__':
    # cal the Params and the FLOPs
    from torchsummary import summary
    from thop import profile

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RA_UNet(num_classes=3).to(device)

    summary(model, (3, 288, 512))
    input = torch.randn(1, 3, 288, 512).to(device)
    flops, params = profile(model, inputs=(input, ))
    flops_in_mb = flops / 1e6
    flops_in_gb = flops / 1e9
    params_in_mb = params / 1e6
    print(f"FLOPs: {flops} ({flops_in_mb:.2f} MFLOPs, {flops_in_gb:.2f} GFLOPs)")  #
    print(f"Parameters: {params}({params_in_mb:.2f} M)")  #
    end = 'end'