"""
Single- and Multi-task U-Net architecture implementation.

The U-Net structure is relatively unaltered from S. Katakol's implementation.
Documentation and type hinting by T. Baker.

The U-Net is built out of the following blocks:
    - TransitionDown: 3D max pool layer.
    - TransitionUp: 3D transposed convolution.
    - BasicBlock: 3D convolution -> 3D Batch norm -> ReLU -> 3D Dropout (Dropout is optional)
    - StageDown: TransitionDown -> BasicBlock -> BasicBlock
    - StageUp: TransitionUp -> BasicBlock -> BasicBlock

The current (July 17, 2023) U-Net stem structure is:
    - start_stage: BasicBlock -> BasicBlock
    - dstage1: StageDown
    - dstage2: StageDown
    - dstage3: TransitionDown -> BasicBlock -> BasicBlock
        (dstage3 is similar to a StageDown block, but with a different channel size for its 2nd BasicBlock)
    - upstage3: TransitionUp -> BasicBlock -> BasicBlock
        (ustage3 is similar to a StageUp block, but with a different channel size for its 2nd BasicBlock)
    - upstage2: StageUp
    - upstage1: StageUp
    - stem_end: BasicBlock
The U-Net stem then branches into two heads - one for segmentation task and the other for landmark localization task.
    - seg_head: BasicBlock -> 3D conv. Output is the segmentation logits.
    - ll_head: BasicBlock -> 3D conv. Output is the heatmap for landmark localization.
The U-Net also supports single task via two Boolean parameters: do_seg, and do_ll.
    - do_seg: When False, the seg_head is not created and the network does not perform segmentation.
    - do_ll: When False, the ll_head is not created and the network does not perform landmark localization.

Edit and run this file's __main__ function to print out a summary of the network with various settings.
"""
from typing import Tuple
import torch
import torch.nn as nn


class TransitionDown(nn.Module):
    def __init__(self):
        super(TransitionDown, self).__init__()
        self.max_pool3d = nn.MaxPool3d((2, 2, 2), padding=1)

    def forward(self, x):
        return self.max_pool3d(x)


class TransitionUp(nn.Module):
    def __init__(self, n_in_channels, n_out_channels):
        super(TransitionUp, self).__init__()
        self.transconv1 = nn.ConvTranspose3d(n_in_channels, n_out_channels, kernel_size=(3, 3, 3), stride=2,
                                             padding=(1, 1, 1), output_padding=1)

    def forward(self, x):
        return self.transconv1(x)


class BasicBlock(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, drop_rate=0.0, kernel_size=(3, 3, 3)):
        super(BasicBlock, self).__init__()

        padding = tuple([int((k-1)/2) for k in kernel_size])
        self.conv1 = nn.Conv3d(n_in_channels, n_out_channels, kernel_size=kernel_size, stride=1, padding=padding)

        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm3d(n_out_channels)
        self.drop_rate = drop_rate
        self.n_out_channels = n_out_channels

        self.drop = nn.Dropout3d(p=self.drop_rate)  # Strong regularization, removes out an entire feature volume.

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.drop_rate > 0:
            x = self.drop(x)
        return x


class StageDown(nn.Module):
    def __init__(self, in_channels, drop_rate=0.0, kernel_size=(3, 3, 3), channel_limit=64):
        super(StageDown, self).__init__()
        self.seq = nn.Sequential(
            TransitionDown(),
            BasicBlock(in_channels, min(2*in_channels, channel_limit), kernel_size=kernel_size, drop_rate=drop_rate),
            # BasicBlock(in_channels, 2*in_channels, kernel_size=kernel_size, drop_rate=drop_rate),
            BasicBlock(min(2*in_channels, channel_limit), min(4*in_channels, channel_limit),
                       kernel_size=kernel_size, drop_rate=drop_rate),
            # BasicBlock(2*in_channels, 4*in_channels, kernel_size=kernel_size, drop_rate=drop_rate),
        )
    
    def forward(self, x):
        return self.seq(x)


class StageUp(nn.Module):
    def __init__(self, in_channels, drop_rate=0.0, kernel_size=(3, 3, 3)):
        super(StageUp, self).__init__()
        self.seq = nn.Sequential(
            TransitionUp(in_channels, in_channels),
            BasicBlock(in_channels, in_channels//2, kernel_size=kernel_size, drop_rate=drop_rate),
            BasicBlock(in_channels//2, in_channels//4, kernel_size=kernel_size, drop_rate=drop_rate),
        )
    
    def forward(self, x):
        return self.seq(x)


class UNet(nn.Module):
    def __init__(self,
                 do_seg: bool = True,
                 do_ll: bool = True,
                 n_seg_classes: int = 1,
                 n_landmarks: int = 6,
                 in_channels: int = 1,
                 channel_size_scale: int = 1,
                 kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 drop_rate: float = 0.0):
        """Single- and Multi-task U-Net implementation.

        See UNetLightningModule in lightning_module.py for information about __init__ parameters.
        """
        super(UNet, self).__init__()
        assert do_seg or do_ll, f"At least one of do_seg:{do_seg} or do_ll:{do_ll} must be True."

        self.do_seg = do_seg
        self.do_ll = do_ll
        self.n_seg_classes = n_seg_classes
        self.n_landmarks = n_landmarks
        self.drop_rate = drop_rate

        self.start_stage = nn.Sequential(
            BasicBlock(in_channels, 4*channel_size_scale, drop_rate=drop_rate, kernel_size=kernel_size),
            BasicBlock(4 * channel_size_scale, 8*channel_size_scale, drop_rate=drop_rate, kernel_size=kernel_size)
        )
        self.dstage1 = StageDown(8*channel_size_scale, drop_rate=drop_rate, kernel_size=kernel_size,
                                 channel_limit=64*channel_size_scale)
        self.dstage2 = StageDown(32 * channel_size_scale, drop_rate=drop_rate, kernel_size=kernel_size,
                                 channel_limit=64*channel_size_scale)
        self.dstage3 = nn.Sequential(
            TransitionDown(),
            BasicBlock(64 * channel_size_scale, 128*channel_size_scale, drop_rate=drop_rate, kernel_size=kernel_size),
            BasicBlock(128 * channel_size_scale, 128*channel_size_scale, drop_rate=drop_rate, kernel_size=kernel_size),
        )

        self.upstage3 = nn.Sequential(
            TransitionUp(128 * channel_size_scale, 128*channel_size_scale),
            BasicBlock(128 * channel_size_scale, 64*channel_size_scale, kernel_size=kernel_size, drop_rate=drop_rate),
            BasicBlock(64 * channel_size_scale, 64*channel_size_scale, kernel_size=kernel_size, drop_rate=drop_rate),
        )
        self.upstage2 = StageUp(128*channel_size_scale, drop_rate=drop_rate, kernel_size=kernel_size)
        self.upstage1 = StageUp(64*channel_size_scale, drop_rate=drop_rate, kernel_size=kernel_size)
        self.stem_end = BasicBlock(24*channel_size_scale, 16*channel_size_scale, kernel_size=kernel_size,
                                   drop_rate=drop_rate)

        if do_seg:
            self.seg_head = nn.Sequential(
                BasicBlock(16*channel_size_scale, 16*channel_size_scale, drop_rate=drop_rate, kernel_size=kernel_size),
                nn.Conv3d(16*channel_size_scale, n_seg_classes, kernel_size=1)
            )

        if do_ll:
            self.ll_head = nn.Sequential(
                BasicBlock(16*channel_size_scale, 16*channel_size_scale, drop_rate=drop_rate, kernel_size=kernel_size),
                nn.Conv3d(16*channel_size_scale, n_landmarks, kernel_size=1)
            )
        
        if n_seg_classes > 1:
            raise NotImplementedError("Need to use NLL loss")

    def forward(self, x):
        # initial stage
        x1 = self.start_stage(x)

        # U-Net stem down stages
        x2 = self.dstage1(x1)
        x3 = self.dstage2(x2)
        x = self.dstage3(x3)

        # U-Net stem up stages
        x = self.upstage3(x)
        x = torch.cat([x[:, :, :x3.shape[2], :x3.shape[3], :x3.shape[4]], x3], 1)
        x = self.upstage2(x)
        x = torch.cat([x[:, :, :x2.shape[2], :x2.shape[3], :x2.shape[4]], x2], 1)
        x = self.upstage1(x)
        x = torch.cat([x[:, :, :x1.shape[2], :x1.shape[3], :x1.shape[4]], x1], 1)

        # final stem processing layer
        x = self.stem_end(x)

        # segmentation and landmark localization heads
        seg_logits = self.seg_head(x) if self.do_seg else torch.full_like(x, -1)
        heatmap = self.ll_head(x) if self.do_ll else torch.full_like(x, -1)

        return seg_logits, heatmap


if __name__ == '__main__':
    # Analyze different network settings
    import torchinfo
    unet = UNet(kernel_size=(3, 3, 3))
    torchinfo.summary(unet, input_size=(16, 1, 96, 96, 160))
    unet = UNet(kernel_size=(5, 5, 5))
    torchinfo.summary(unet, input_size=(16, 1, 96, 96, 160))
    unet = UNet(kernel_size=(3, 3, 3), channel_size_scale=2)
    torchinfo.summary(unet, input_size=(16, 1, 96, 96, 160))

