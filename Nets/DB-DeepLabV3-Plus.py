# -*- conding: utf-8 -*- #

# -------------------------------------------------------
# @Name:         DB_DeepLabV3p_L4.py
# @Author:       ZhangJiapeng@ZY-301
# @Data:         2024/12/29
# @Time:         18:04
# -------------------------------------------------------
# @Even:         
# @Description:  

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


class DB_DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, co_branch_rate=1.0, co_aspp=True):
        super(DB_DeepLabV3Plus, self).__init__()

        self.co_branch_rate = co_branch_rate

        self.layer0_main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
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
            Bottleneck(256 + int(256 * self.co_branch_rate), 128, 2,
                          nn.Sequential(
                            nn.Conv2d(256 + int(256 * self.co_branch_rate), 512, kernel_size=1, stride=2, bias=False),
                            nn.BatchNorm2d(512)
                          ),
                          1, 64, 1, nn.BatchNorm2d),
            Bottleneck(512, 128, 1, None, 1, 64, 1, nn.BatchNorm2d),
            Bottleneck(512, 128, 1, None, 1, 64, 1, nn.BatchNorm2d),
            Bottleneck(512, 128, 1, None, 1, 64, 1, nn.BatchNorm2d),
        )

        self.layer3_main = nn.Sequential(
            Bottleneck(512, 256, 1,
                          nn.Sequential(
                            nn.Conv2d(512, 1024, kernel_size=1, stride=1, bias=False),
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
            Bottleneck(1024, 512, 1,
                            nn.Sequential(
                                nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(2048)
                            ),
                            1, 64, 1, nn.BatchNorm2d),
            Bottleneck(2048, 512, 1, None, 1, 64, 1, nn.BatchNorm2d),
            Bottleneck(2048, 512, 1, None, 1, 64, 1, nn.BatchNorm2d),
        )

        self.aspp_main = ASPP(2048 + int(2048 * self.co_branch_rate), 256)

        self.decoder_main = nn.Sequential(
            nn.Conv2d(256 * 2 + int(256 * 2 * self.co_branch_rate), 256 + int(256 * self.co_branch_rate), kernel_size=3, padding=1),
            nn.BatchNorm2d(256 + int(256 * self.co_branch_rate)),
            nn.ReLU(),
            nn.Conv2d(256 + int(256 * self.co_branch_rate), 256 + int(256 * self.co_branch_rate), kernel_size=3, padding=1),
            nn.BatchNorm2d(256 + int(256 * self.co_branch_rate)),
            nn.ReLU(),
            nn.Conv2d(256 + int(256 * self.co_branch_rate), num_classes, kernel_size=1)
        )

        self.layer0_co = nn.Sequential(
            nn.Conv2d(3, int(64 * self.co_branch_rate), kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(int(64 * self.co_branch_rate)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1_co = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            nn.Sequential(
                Bottleneck(int(64 * self.co_branch_rate), int(64 * self.co_branch_rate), 1,
                           nn.Sequential(
                               nn.Conv2d(int(64 * self.co_branch_rate), int(256 * self.co_branch_rate), kernel_size=1, stride=1, bias=False),
                               nn.BatchNorm2d(int(256 * self.co_branch_rate))
                           ),
                           1, 64, 1, nn.BatchNorm2d),
                Bottleneck(int(256 * self.co_branch_rate), int(64 * self.co_branch_rate), 1, None, 1, 64, 1, nn.BatchNorm2d),
                Bottleneck(int(256 * self.co_branch_rate), int(64 * self.co_branch_rate), 1, None, 1, 64, 1, nn.BatchNorm2d),
            )
        )

        self.layer2_co = nn.Sequential(
            Bottleneck(int(256 * self.co_branch_rate), int(128 * self.co_branch_rate), 2,
                       nn.Sequential(
                           nn.Conv2d(int(256 * self.co_branch_rate), int(512 * self.co_branch_rate), kernel_size=1, stride=2, bias=False),
                           nn.BatchNorm2d(int(512 * self.co_branch_rate))
                       ),
                       1, 64, 1, nn.BatchNorm2d),
            Bottleneck(int(512 * self.co_branch_rate), int(128 * self.co_branch_rate), 1, None, 1, 64, 1, nn.BatchNorm2d),
            Bottleneck(int(512 * self.co_branch_rate), int(128 * self.co_branch_rate), 1, None, 1, 64, 1, nn.BatchNorm2d),
            Bottleneck(int(512 * self.co_branch_rate), int(128 * self.co_branch_rate), 1, None, 1, 64, 1, nn.BatchNorm2d),
        )

        self.layer3_co = nn.Sequential(
            Bottleneck(int(512 * self.co_branch_rate), int(256 * self.co_branch_rate), 1,
                       nn.Sequential(
                           nn.Conv2d(int(512 * self.co_branch_rate), int(1024 * self.co_branch_rate), kernel_size=1, stride=1, bias=False),
                           nn.BatchNorm2d(int(1024 * self.co_branch_rate))
                       ),
                       1, 64, 1, nn.BatchNorm2d),
            Bottleneck(int(1024 * self.co_branch_rate), int(256 * self.co_branch_rate), 1, None, 1, 64, 1, nn.BatchNorm2d),
            Bottleneck(int(1024 * self.co_branch_rate), int(256 * self.co_branch_rate), 1, None, 1, 64, 1, nn.BatchNorm2d),
            Bottleneck(int(1024 * self.co_branch_rate), int(256 * self.co_branch_rate), 1, None, 1, 64, 1, nn.BatchNorm2d),
            Bottleneck(int(1024 * self.co_branch_rate), int(256 * self.co_branch_rate), 1, None, 1, 64, 1, nn.BatchNorm2d),
            Bottleneck(int(1024 * self.co_branch_rate), int(256 * self.co_branch_rate), 1, None, 1, 64, 1, nn.BatchNorm2d),
        )

        self.layer4_co = nn.Sequential(
            Bottleneck(int(1024 * self.co_branch_rate), int(512 * self.co_branch_rate), 1,
                       nn.Sequential(
                           nn.Conv2d(int(1024 * self.co_branch_rate), int(2048 * self.co_branch_rate), kernel_size=1, stride=1, bias=False),
                           nn.BatchNorm2d(int(2048 * self.co_branch_rate))
                       ),
                       1, 64, 1, nn.BatchNorm2d),
            Bottleneck(int(2048 * self.co_branch_rate), int(512 * self.co_branch_rate), 1, None, 1, 64, 1, nn.BatchNorm2d),
            Bottleneck(int(2048 * self.co_branch_rate), int(512 * self.co_branch_rate), 1, None, 1, 64, 1, nn.BatchNorm2d),
        )

        self.aspp_co = ASPP(int(2048 * self.co_branch_rate), int(256 * self.co_branch_rate))

        self.decoder_co = nn.Sequential(
            nn.Conv2d(int((256 + 256) * self.co_branch_rate), int(256 * self.co_branch_rate) , kernel_size=3, padding=1),
            nn.BatchNorm2d(int(256 * self.co_branch_rate)),
            nn.ReLU(),
            nn.Conv2d(int(256 * self.co_branch_rate), int(256 * self.co_branch_rate), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(256 * self.co_branch_rate)),
            nn.ReLU(),
            nn.Conv2d(int(256 * self.co_branch_rate), 1, kernel_size=1)
        )

    def forward(self, rgb, edge):
        if self.co_branch_rate > 0:
            edge_x0 = self.layer0_co(edge)
            rgb_x0 = self.layer0_main(rgb)

            edge_x1 = self.layer1_co(edge_x0)
            rgb_x1 = self.layer1_main(rgb_x0)

            edge_x2 = self.layer2_co(edge_x1)
            rgb_x2 = self.layer2_main(torch.cat((rgb_x1, edge_x1), dim=1))  # 2 512 18 32

            edge_x3 = self.layer3_co(edge_x2)
            rgb_x3 = self.layer3_main(rgb_x2)

            edge_x4 = self.layer4_co(edge_x3)
            rgb_x4 = self.layer4_main(rgb_x3)

            edge_aspp = self.aspp_co(edge_x4)
            edge_aspp = nn.functional.interpolate(edge_aspp, size=edge_x1.size()[2:], mode='bilinear', align_corners=True)

            main_aspp = self.aspp_main(torch.cat((rgb_x4, edge_x4), dim=1))
            main_aspp = nn.functional.interpolate(main_aspp, size=rgb_x1.size()[2:], mode='bilinear', align_corners=True)

            edge_x21 = self.decoder_co(torch.cat((edge_x1, edge_aspp), dim=1))
            edge_x21 = nn.functional.interpolate(edge_x21, size=edge.size()[2:], mode='bilinear', align_corners=True)

            rgb_x21 = self.decoder_main(torch.cat((rgb_x1, main_aspp, edge_x1, edge_aspp), dim=1))
            rgb_x21 = nn.functional.interpolate(rgb_x21, size=rgb.size()[2:], mode='bilinear', align_corners=True)

            return rgb_x21, edge_x21

        else:
            rgb_x0 = self.layer0_main(rgb)
            rgb_x1 = self.layer1_main(rgb_x0)
            rgb_x2 = self.layer2_main(rgb_x1)
            rgb_x3 = self.layer3_main(rgb_x2)
            rgb_x4 = self.layer4_main(rgb_x3)
            main_aspp = self.aspp_main(rgb_x4)
            main_aspp = nn.functional.interpolate(main_aspp, size=rgb_x1.size()[2:], mode='bilinear', align_corners=True)
            rgb_x21 = self.decoder_main(torch.cat((rgb_x1, main_aspp), dim=1))
            rgb_x21 = nn.functional.interpolate(rgb_x21, size=rgb.size()[2:], mode='bilinear', align_corners=True)

            return rgb_x21, None

if __name__ == '__main__':
    from torchsummary import summary
    from thop import profile

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DB_DeepLabV3Plus(num_classes=3, co_branch_rate=1).to(device)
    summary(model, [(3, 288, 512), (3, 288, 512)])
    input = torch.randn(1, 3, 288, 512).to(device)
    flops, params = profile(model, inputs=(input,input))
    flops_in_mb = flops / 1e6
    flops_in_gb = flops / 1e9
    params_in_mb = params / 1e6
    print(f"FLOPs: {flops} ({flops_in_mb:.2f} MFLOPs, {flops_in_gb:.2f} GFLOPs)")  #
    print(f"Parameters: {params}({params_in_mb:.2f} M)")  #
    end = 'end'