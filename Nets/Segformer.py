# -*- conding: utf-8 -*- #

# -------------------------------------------------------
# @Name:         Unet
# @Author:       ZhangJiapeng@ZY-301
# @Data:         2025/01/20
# @Time:         13:58
# -------------------------------------------------------
# @Even:         
# @Description:  
# https://github.com/lucidrains/segformer-pytorch

# -------------------------------------------------------
import torch
from segformer_pytorch import Segformer


if __name__=='__main__':
    # cal the Params and the FLOPs
    from torchsummary import summary
    from thop import profile

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = SegFormer(num_classes=3).to(device)

    model = Segformer(
        dims=(32, 64, 160, 256),
        heads=(1, 2, 5, 8),
        ff_expansion=(8, 8, 4, 4),
        reduction_ratio=(8, 4, 2, 1),
        num_layers=2,
        decoder_dim=256,
        num_classes=3
    ).to(device)

    summary(model, (3, 288, 512))
    input = torch.randn(1, 3, 288, 512).to(device)
    flops, params = profile(model, inputs=(input, ))
    flops_in_mb = flops / 1e6
    flops_in_gb = flops / 1e9
    params_in_mb = params / 1e6
    print(f"FLOPs: {flops} ({flops_in_mb:.2f} MFLOPs, {flops_in_gb:.2f} GFLOPs)")  #
    print(f"Parameters: {params}({params_in_mb:.2f} M)")  #
    end = 'end'
