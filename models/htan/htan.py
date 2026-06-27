import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transattunet.TransAttUnet import (
    ConvBlock, ResidualDecoderBlock, SAA_Module
)
from models.htan.mhc import ManifoldConstrainedHyperConnection, ReshapingSAA


# ---------------------------------------------------------------------------
# HTAN_1 — TransAttUNet_R + 1 mHC block wrapping the SAA at x5 (1024ch)
# ---------------------------------------------------------------------------
class HTAN_1(nn.Module):
    """
    TransAttUNet_R with mHC wrapping the bottleneck SAA at x5.
    Encoder and decoder are identical to TransAttUNet_R.
    Only the bottleneck is modified — SAA is wrapped in mHC.

    Args:
        expansion_n (int): mHC stream width n (2 or 4)
        hres_only (bool):  Ablation — only constrain H_res
        img_size (int):    Input image size (default 256)
        num_heads (int):   Attention heads in SAA (default 8)
    """
    def __init__(self, n_channels=3, n_classes=1, expansion_n=4,
                 hres_only=False, img_size=256, num_heads=8):
        super().__init__()
        self.expansion_n = expansion_n

        # Spatial dims at x5: after 4 maxpools
        sp_x5   = img_size // 16   # 256->16
        ch_x5   = 1024
        flat_x5 = ch_x5 * sp_x5 * sp_x5

        # --- Encoder (identical to TransAttUNet_R) ---
        self.inc   = ConvBlock(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(512, 1024))

        # --- mHC Bottleneck ---
        # Stream initiator: expand 1024ch → 1024*n channels
        self.stream_init = nn.Conv2d(ch_x5, ch_x5 * expansion_n, 1)

        # SAA wrapped in ReshapingSAA for mHC compatibility
        saa_module    = SAA_Module(ch_x5, num_heads)
        reshaping_saa = ReshapingSAA(saa_module, (ch_x5, sp_x5, sp_x5))

        self.mhc_x5   = ManifoldConstrainedHyperConnection(
            dim_C=flat_x5,
            expansion_n=expansion_n,
            sub_layer_module=reshaping_saa,
            hres_only=hres_only,
        )

        # Aggregate n streams back to 1024ch
        self.aggregator = nn.Conv2d(ch_x5 * expansion_n, ch_x5, 1)

        # Store SAA ref for epoch updates in trainer
        self.saa_x5 = saa_module

        # --- Decoder (identical to TransAttUNet_R) ---
        self.up1  = ResidualDecoderBlock(1024, 512, 0,    512)
        self.up2  = ResidualDecoderBlock(512,  256, 1024, 256)
        self.up3  = ResidualDecoderBlock(256,  128, 512,  128)
        self.up4  = ResidualDecoderBlock(128,  64,  256,  64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x, epoch=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        B, C, H, W = x5.shape

        # Update SAA warmup epoch
        if epoch is not None:
            self.saa_x5.current_epoch = epoch if hasattr(self.saa_x5, 'current_epoch') else None

        # mHC at x5
        x5_exp  = self.stream_init(x5)
        x5_str  = x5_exp.view(B, self.expansion_n, -1)
        out_str = self.mhc_x5(x5_str)
        bridge  = self.aggregator(out_str.view(B, -1, H, W))

        # Decode — identical to TransAttUNet_R
        d1 = self.up1(bridge, x4)
        d2 = self.up2(d1, x3, bridge)
        d3 = self.up3(d2, x2, d1)
        d4 = self.up4(d3, x1, d2)

        return self.outc(d4)


# ---------------------------------------------------------------------------
# HTAN_2 — TransAttUNet_R + 2 mHC blocks (x5 + x4)
# ---------------------------------------------------------------------------
class HTAN_2(nn.Module):
    """
    TransAttUNet_R with mHC at both x5 (1024ch, 16x16)
    and x4 (512ch, 32x32). Everything else identical to TransAttUNet_R.
    """
    def __init__(self, n_channels=3, n_classes=1, expansion_n=4,
                 hres_only=False, img_size=256, num_heads=8):
        super().__init__()
        self.expansion_n = expansion_n

        sp_x5   = img_size // 16
        sp_x4   = img_size // 8
        ch_x5   = 1024
        ch_x4   = 512
        flat_x5 = ch_x5 * sp_x5 * sp_x5
        flat_x4 = ch_x4 * sp_x4 * sp_x4

        # --- Encoder ---
        self.inc   = ConvBlock(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(512, 1024))

        # --- mHC at x5 ---
        self.stream_init_x5 = nn.Conv2d(ch_x5, ch_x5 * expansion_n, 1)
        saa_x5              = SAA_Module(ch_x5, num_heads)
        self.mhc_x5         = ManifoldConstrainedHyperConnection(
            dim_C=flat_x5, expansion_n=expansion_n,
            sub_layer_module=ReshapingSAA(saa_x5, (ch_x5, sp_x5, sp_x5)),
            hres_only=hres_only
        )
        self.aggregator_x5  = nn.Conv2d(ch_x5 * expansion_n, ch_x5, 1)
        self.saa_x5         = saa_x5

        # --- mHC at x4 ---
        self.stream_init_x4 = nn.Conv2d(ch_x4, ch_x4 * expansion_n, 1)
        saa_x4              = SAA_Module(ch_x4, num_heads)
        self.mhc_x4         = ManifoldConstrainedHyperConnection(
            dim_C=flat_x4, expansion_n=expansion_n,
            sub_layer_module=ReshapingSAA(saa_x4, (ch_x4, sp_x4, sp_x4)),
            hres_only=hres_only
        )
        self.aggregator_x4  = nn.Conv2d(ch_x4 * expansion_n, ch_x4, 1)
        self.saa_x4         = saa_x4

        # --- Decoder ---
        self.up1  = ResidualDecoderBlock(1024, 512, 0,    512)
        self.up2  = ResidualDecoderBlock(512,  256, 1024, 256)
        self.up3  = ResidualDecoderBlock(256,  128, 512,  128)
        self.up4  = ResidualDecoderBlock(128,  64,  256,  64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x, epoch=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        B, C5, H5, W5 = x5.shape
        B, C4, H4, W4 = x4.shape

        # mHC at x5
        x5_exp = self.stream_init_x5(x5)
        x5_str = x5_exp.view(B, self.expansion_n, -1)
        bridge = self.aggregator_x5(self.mhc_x5(x5_str).view(B, -1, H5, W5))

        # mHC at x4
        x4_exp = self.stream_init_x4(x4)
        x4_str = x4_exp.view(B, self.expansion_n, -1)
        x4     = self.aggregator_x4(self.mhc_x4(x4_str).view(B, -1, H4, W4))

        # Decode
        d1 = self.up1(bridge, x4)
        d2 = self.up2(d1, x3, bridge)
        d3 = self.up3(d2, x2, d1)
        d4 = self.up4(d3, x1, d2)

        return self.outc(d4)


# ---------------------------------------------------------------------------
# HTAN_1_Hres_only — ablation
# ---------------------------------------------------------------------------
class HTAN_1_Hres_only(HTAN_1):
    def __init__(self, n_channels=3, n_classes=1, expansion_n=4,
                 img_size=256, num_heads=8):
        super().__init__(n_channels=n_channels, n_classes=n_classes,
                         expansion_n=expansion_n, hres_only=True,
                         img_size=img_size, num_heads=num_heads)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
HTAN_MODELS = {
    "htan_1_n2":        lambda: HTAN_1(expansion_n=2),
    "htan_1_n4":        lambda: HTAN_1(expansion_n=4),
    "htan_2_n2":        lambda: HTAN_2(expansion_n=2),
    "htan_2_n4":        lambda: HTAN_2(expansion_n=4),
    "htan_1_hres_only": lambda: HTAN_1_Hres_only(expansion_n=4),
}