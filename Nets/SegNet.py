# -*- conding: utf-8 -*- #

# -------------------------------------------------------
# @Name:         SegNet.py
# @Author:       ZhangJiapeng@ZY-301
# @Data:         2025/5/26
# @Time:         20:48
# -------------------------------------------------------
# @Even:         
# @Description:  

# -------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super().__init__()
        layers = []
        for i in range(num_convs):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
            else:
                layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SegNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        # Encoder (VGG-like structure)
        self.enc_conv1 = VGGBlock(input_channels, 64, 2)
        self.enc_conv2 = VGGBlock(64, 128, 2)
        self.enc_conv3 = VGGBlock(128, 256, 3)
        self.enc_conv4 = VGGBlock(256, 512, 3)
        self.enc_conv5 = VGGBlock(512, 512, 3)

        # Decoder (mirror of encoder)
        self.dec_conv5 = VGGBlock(512, 512, 3)
        self.dec_conv4 = VGGBlock(512, 256, 3)
        self.dec_conv3 = VGGBlock(256, 128, 3)
        self.dec_conv2 = VGGBlock(128, 64, 2)
        self.dec_conv1 = VGGBlock(64, 64, 2)

        # Final classification layer
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        # Encoder with max pooling and index storage
        x1 = self.enc_conv1(input)
        x1_pooled, idx1 = F.max_pool2d(x1, 2, 2, return_indices=True)

        x2 = self.enc_conv2(x1_pooled)
        x2_pooled, idx2 = F.max_pool2d(x2, 2, 2, return_indices=True)

        x3 = self.enc_conv3(x2_pooled)
        x3_pooled, idx3 = F.max_pool2d(x3, 2, 2, return_indices=True)

        x4 = self.enc_conv4(x3_pooled)
        x4_pooled, idx4 = F.max_pool2d(x4, 2, 2, return_indices=True)

        x5 = self.enc_conv5(x4_pooled)
        x5_pooled, idx5 = F.max_pool2d(x5, 2, 2, return_indices=True)

        # Decoder with max unpooling using stored indices
        d5 = F.max_unpool2d(x5_pooled, idx5, 2, 2)
        d5 = self.dec_conv5(d5)
        d4 = F.max_unpool2d(d5, idx4, 2, 2)
        d4 = self.dec_conv4(d4)
        d3 = F.max_unpool2d(d4, idx3, 2, 2)
        d3 = self.dec_conv3(d3)
        d2 = F.max_unpool2d(d3, idx2, 2, 2)
        d2 = self.dec_conv2(d2)
        d1 = F.max_unpool2d(d2, idx1, 2, 2)
        d1 = self.dec_conv1(d1)

        # Final output
        output = self.final(d1)  # output: [B, num_classes, H, W]

        return output


if __name__ == '__main__':
    # cal the Params and the FLOPs
    from torchsummary import summary
    from thop import profile

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SegNet(num_classes=3, input_channels=3).to(device)

    summary(model, (3, 288, 512))
    input = torch.randn(1, 3, 288, 512).to(device)
    flops, params = profile(model, inputs=(input,))
    flops_in_mb = flops / 1e6
    flops_in_gb = flops / 1e9
    params_in_mb = params / 1e6
    print(f"FLOPs: {flops} ({flops_in_mb:.2f} MFLOPs, {flops_in_gb:.2f} GFLOPs)")
    print(f"Parameters: {params}({params_in_mb:.2f} M)")
    end = 'end'