import numpy as np
import torch
import torch.nn as nn

try:
    from .backbone import Multi_Concat_Block, Conv, SiLU, Transition_Block, autopad
    from .mar_backbone import MARBackbone
except ImportError:
    from backbone import Multi_Concat_Block, Conv, SiLU, Transition_Block, autopad
    from mar_backbone import MARBackbone

class SPPCSPC(nn.Module):
    """Spatial pyramid pooling block used before the detection neck. yolov7"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        """Build parallel pooling branches and projection layers."""
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        """Apply pyramid pooling and concatenate with the shortcut branch."""
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

class RepConv(nn.Module):
    """Re-parameterizable convolution block with train and deploy branches."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=SiLU(), deploy=False):
        """Create a RepConv block."""
        super(RepConv, self).__init__()
        self.deploy         = deploy
        self.groups         = g
        self.in_channels    = c1
        self.out_channels   = c2
        
        assert k == 3
        assert autopad(k, p) == 1

        padding_11  = autopad(k, p) - k // 2
        self.act    = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam    = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        else:
            self.rbr_identity   = (nn.BatchNorm2d(num_features=c1, eps=0.001, momentum=0.03) if c2 == c1 and s == 1 else None)
            self.rbr_dense      = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )
            self.rbr_1x1        = nn.Sequential(
                nn.Conv2d( c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )

    def forward(self, inputs):
        """Run either the deploy convolution or the training branches."""
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
    
    def get_equivalent_kernel_bias(self):
        """Fuse all training branches into one equivalent kernel and bias."""
        kernel3x3, bias3x3  = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1  = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid    = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pad a 1x1 kernel to the center of a 3x3 kernel."""
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Fold a branch's batch normalization into its convolution kernel."""
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel      = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma       = branch[1].weight
            beta        = branch[1].bias
            eps         = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel      = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma       = branch.weight
            beta        = branch.bias
            eps         = branch.eps
        std = (running_var + eps).sqrt()
        t   = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        """Return the fused RepConv kernel and bias as NumPy arrays."""
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

    def fuse_conv_bn(self, conv, bn):
        """Return a convolution layer with batch normalization fused in."""
        std     = (bn.running_var + bn.eps).sqrt()
        bias    = bn.bias - bn.running_mean * bn.weight / std

        t       = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn      = nn.Identity()
        conv    = nn.Conv2d(in_channels = conv.in_channels,
                              out_channels = conv.out_channels,
                              kernel_size = conv.kernel_size,
                              stride=conv.stride,
                              padding = conv.padding,
                              dilation = conv.dilation,
                              groups = conv.groups,
                              bias = True,
                              padding_mode = conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias   = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):    
        """Convert the training branches into a single deploy convolution."""
        if self.deploy:
            return
        print(f"RepConv.fuse_repvgg_block")
        self.rbr_dense  = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])
        
        self.rbr_1x1    = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias    = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])
        
        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity, nn.modules.batchnorm.SyncBatchNorm)):
            identity_conv_1x1 = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=self.groups, 
                    bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)

            identity_conv_1x1           = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded      = identity_conv_1x1.bias
            weight_identity_expanded    = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])            
        else:
            bias_identity_expanded      = torch.nn.Parameter( torch.zeros_like(rbr_1x1_bias) )
            weight_identity_expanded    = torch.nn.Parameter( torch.zeros_like(weight_1x1_expanded) )            
        
        self.rbr_dense.weight   = torch.nn.Parameter(self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias     = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)
                
        self.rbr_reparam    = self.rbr_dense
        self.deploy         = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None
            
def fuse_conv_and_bn(conv, bn):
    """Fuse a Conv2d and BatchNorm2d pair for inference."""
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv  = conv.weight.clone().view(conv.out_channels, -1)
    w_bn    = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    # fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape).detach())

    b_conv  = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn    = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    # fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    fusedconv.bias.copy_((torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn).detach())
    return fusedconv

class RaShipWakeDet(nn.Module):
    """Radiometric Ship and Wake Detection Net."""

    def __init__(
        self,
        anchors_mask,
        num_classes,
        variant="s2",
        input_shape=(640, 640),
        pretrained=False,
        pretrained_path="",
        neck_channels=(128, 256, 512),
        ablation_mode="full",
    ):
        """Initialize Stem/MAR Backbone/PAN Neck/Head detection modules."""
        super().__init__()
        resolution = input_shape[0] if isinstance(input_shape, (list, tuple)) else input_shape
        backbone_pretrained_path = pretrained_path if pretrained else ""
        self.mar_backbone = MARBackbone(
            variant=variant,
            resolution=resolution,
            pretrained_path=backbone_pretrained_path,
            out_indices=(0, 1, 2, 3),
            ablation_mode=ablation_mode,
        )
        _, c8, c16, c32 = self.mar_backbone.out_channels

        p3_c, p4_c, p5_c = neck_channels
        ids = [-1, -2, -3, -4, -5, -6]
        n = 4
        e = 2

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.neck_sppcspc = SPPCSPC(c32, p5_c)
        self.neck_p5_reduce = Conv(p5_c, p4_c)
        self.neck_c4_proj = Conv(c16, p4_c)
        self.neck_eelan_p4_td = Multi_Concat_Block(p4_c * 2, p4_c, p4_c, e=e, n=n, ids=ids)

        self.neck_p4_reduce = Conv(p4_c, p3_c)
        self.neck_c3_proj = Conv(c8, p3_c)
        self.neck_eelan_p3 = Multi_Concat_Block(p3_c * 2, p3_c, p3_c, e=e, n=n, ids=ids)

        self.neck_p3_down = Transition_Block(p3_c, p3_c)
        self.neck_eelan_p4_bu = Multi_Concat_Block(p4_c * 2, p4_c, p4_c, e=e, n=n, ids=ids)

        self.neck_p4_down = Transition_Block(p4_c, p4_c)
        self.neck_eelan_p5 = Multi_Concat_Block(p5_c * 2, p5_c, p5_c, e=e, n=n, ids=ids)

        self.head_repconv_p3 = RepConv(p3_c, p3_c * 2, 3, 1)
        self.head_repconv_p4 = RepConv(p4_c, p4_c * 2, 3, 1)
        self.head_repconv_p5 = RepConv(p5_c, p5_c * 2, 3, 1)

        self.head_pred_p3 = nn.Conv2d(p3_c * 2, len(anchors_mask[2]) * (5 + num_classes), 1)
        self.head_pred_p4 = nn.Conv2d(p4_c * 2, len(anchors_mask[1]) * (5 + num_classes), 1)
        self.head_pred_p5 = nn.Conv2d(p5_c * 2, len(anchors_mask[0]) * (5 + num_classes), 1)

    def fuse(self):
        """Fuse supported convolution and normalization layers for inference."""
        print("Fusing layers... ")
        for m in self.modules():
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()
            elif type(m) is Conv and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, "bn")
                m.forward = m.fuseforward
        return self

    def forward(self, x):
        """Run MAR Backbone, PAN Neck, and three-scale detection heads."""
        _, c3, c4, c5 = self.mar_backbone(x)

        P5 = self.neck_sppcspc(c5)
        P5_conv = self.neck_p5_reduce(P5)
        P5_upsample = self.upsample(P5_conv)

        P4 = torch.cat([self.neck_c4_proj(c4), P5_upsample], 1)
        P4 = self.neck_eelan_p4_td(P4)

        P4_conv = self.neck_p4_reduce(P4)
        P4_upsample = self.upsample(P4_conv)

        P3 = torch.cat([self.neck_c3_proj(c3), P4_upsample], 1)
        P3 = self.neck_eelan_p3(P3)

        P3_downsample = self.neck_p3_down(P3)
        P4 = torch.cat([P3_downsample, P4], 1)
        P4 = self.neck_eelan_p4_bu(P4)

        P4_downsample = self.neck_p4_down(P4)
        P5 = torch.cat([P4_downsample, P5], 1)
        P5 = self.neck_eelan_p5(P5)

        P3 = self.head_repconv_p3(P3)
        P4 = self.head_repconv_p4(P4)
        P5 = self.head_repconv_p5(P5)

        out2 = self.head_pred_p3(P3)
        out1 = self.head_pred_p4(P4)
        out0 = self.head_pred_p5(P5)
        return [out0, out1, out2]
       





