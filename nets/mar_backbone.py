import itertools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


MAR_WIDTHS = {
    "s2": [32, 64, 144, 288],
}

MAR_DEPTHS = {
    "s2": [4, 4, 12, 8],
}

EXPANSION_RATIOS = {
    "s2": {
        "0": [4, 4, 4, 4],
        "1": [4, 4, 4, 4],
        "2": [4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
        "3": [4, 4, 3, 3, 3, 3, 4, 4],
    },
}

VIT_NUMS = { "s2": 4, }
DROP_PATH_RATES = {"s2": 0.02, }

ABLATION_CONFIGS = {
    "A": {
        "use_bmc": False,
        "use_gca": False,
        "use_gbc": False,
        "use_gca_down": False,
        "use_arca": False,
        "bmc_use_arca": False,
    },
    "B": {
        "use_bmc": True,
        "use_gca": False,
        "use_gbc": False,
        "use_gca_down": False,
        "use_arca": False,
        "bmc_use_arca": False,
    },
    "C": {
        "use_bmc": True,
        "use_gca": True,
        "use_gbc": False,
        "use_gca_down": False,
        "use_arca": False,
        "bmc_use_arca": False,
    },
    "D": {
        "use_bmc": True,
        "use_gca": True,
        "use_gbc": True,
        "use_gca_down": False,
        "use_arca": False,
        "bmc_use_arca": False,
    },
    "E": {
        "use_bmc": True,
        "use_gca": True,
        "use_gbc": True,
        "use_gca_down": True,
        "use_arca": False,
        "bmc_use_arca": False,
    },
    "full": {
        "use_bmc": True,
        "use_gca": True,
        "use_gbc": True,
        "use_gca_down": True,
        "use_arca": True,
        "bmc_use_arca": True,
    },
}


def resolve_ablation_config(ablation_mode):
    """Return the fixed module switches for one ablation network."""
    if ablation_mode is None:
        ablation_mode = "full"
    mode = str(ablation_mode)
    key = mode.upper() if mode.upper() in ABLATION_CONFIGS else mode.lower()
    if key not in ABLATION_CONFIGS:
        valid = ", ".join(ABLATION_CONFIGS.keys())
        raise ValueError(f"Unsupported ablation_mode={ablation_mode}. Expected one of: {valid}.")
    return key, ABLATION_CONFIGS[key].copy()


def drop_path(x, drop_prob=0.0, training=False):
    """Apply stochastic depth to a residual branch."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Module wrapper for stochastic depth."""

    def __init__(self, drop_prob=0.0):
        """Store the stochastic-depth drop probability."""
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """Apply drop path during training."""
        return drop_path(x, self.drop_prob, self.training)


class Stem(nn.Sequential):
    """Initial two-step feature embedding stem."""

    def __init__(self, in_chs, out_chs, act_layer=nn.ReLU):
        """Build two stride-2 convolutions for H/4 spatial compression."""
        super().__init__(
            nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_chs // 2),
            act_layer(),
            nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_chs),
            act_layer(),
        )


class LGQuery(nn.Module):
    """Local-global query projection used by downsampling attention."""

    def __init__(self, in_dim, out_dim):
        """Create pooled, local, and projection branches."""
        super().__init__()
        self.pool = nn.AvgPool2d(1, 2, 0)
        self.local = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=2, padding=1, groups=in_dim)
        self.proj = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1), nn.BatchNorm2d(out_dim))

    def forward(self, x):
        """Combine average-pooled and local-query features."""
        return self.proj(self.pool(x) + self.local(x))


class GCA(nn.Module):
    """Global Context Attention with relative position bias."""

    def __init__(self, dim, key_dim=32, num_heads=8, attn_ratio=4, resolution=7, act_layer=nn.ReLU, stride=None):
        """Initialize attention projections and relative-bias lookup tables."""
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = self.d * num_heads

        if stride is not None:
            self.resolution = math.ceil(resolution / stride)
            self.stride_conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim),
                nn.BatchNorm2d(dim),
            )
            self.upsample = nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=False)
        else:
            self.resolution = resolution
            self.stride_conv = None
            self.upsample = None

        self.q = nn.Sequential(nn.Conv2d(dim, num_heads * key_dim, 1), nn.BatchNorm2d(num_heads * key_dim))
        self.k = nn.Sequential(nn.Conv2d(dim, num_heads * key_dim, 1), nn.BatchNorm2d(num_heads * key_dim))
        self.v = nn.Sequential(nn.Conv2d(dim, self.dh, 1), nn.BatchNorm2d(self.dh))
        self.v_local = nn.Sequential(
            nn.Conv2d(self.dh, self.dh, kernel_size=3, stride=1, padding=1, groups=self.dh),
            nn.BatchNorm2d(self.dh),
        )
        self.talking_head1 = nn.Conv2d(num_heads, num_heads, 1)
        self.talking_head2 = nn.Conv2d(num_heads, num_heads, 1)
        self.proj = nn.Sequential(act_layer(), nn.Conv2d(self.dh, dim, 1), nn.BatchNorm2d(dim))

        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in offsets:
                    offsets[offset] = len(offsets)
                idxs.append(offsets[offset])
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, len(offsets)))
        self.register_buffer("attention_bias_idxs", torch.LongTensor(idxs).view(len(points), len(points)))

    @torch.no_grad()
    def train(self, mode=True):
        """Cache relative attention bias for evaluation mode."""
        super().train(mode)
        if mode and hasattr(self, "ab"):
            del self.ab
        elif not mode:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        """Apply spatial attention and local value enhancement."""
        if self.stride_conv is not None:
            x = self.stride_conv(x)

        b, _, h, w = x.shape
        n = h * w
        q = self.q(x).flatten(2).reshape(b, self.num_heads, self.key_dim, n).permute(0, 1, 3, 2)
        k = self.k(x).flatten(2).reshape(b, self.num_heads, self.key_dim, n)
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape(b, self.num_heads, self.d, n).permute(0, 1, 3, 2)

        bias = self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab
        attn = (q @ k) * self.scale + bias
        attn = self.talking_head1(attn)
        attn = attn.softmax(dim=-1)
        attn = self.talking_head2(attn)

        x = (attn @ v).transpose(2, 3).reshape(b, self.dh, h, w) + v_local
        if self.upsample is not None:
            x = self.upsample(x)
        return self.proj(x)


class GCADownBlock(nn.Module):
    """Global Context Attention downsampling block."""

    def __init__(self, dim, out_dim=None, key_dim=16, num_heads=8, attn_ratio=4, resolution=7, act_layer=nn.ReLU):
        """Initialize query downsampling and attention projections."""
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = self.d * num_heads
        self.out_dim = out_dim or dim
        self.resolution = resolution
        self.resolution2 = math.ceil(resolution / 2)

        self.q = LGQuery(dim, num_heads * key_dim)
        self.k = nn.Sequential(nn.Conv2d(dim, num_heads * key_dim, 1), nn.BatchNorm2d(num_heads * key_dim))
        self.v = nn.Sequential(nn.Conv2d(dim, self.dh, 1), nn.BatchNorm2d(self.dh))
        self.v_local = nn.Sequential(
            nn.Conv2d(self.dh, self.dh, kernel_size=3, stride=2, padding=1, groups=self.dh),
            nn.BatchNorm2d(self.dh),
        )
        self.proj = nn.Sequential(act_layer(), nn.Conv2d(self.dh, self.out_dim, 1), nn.BatchNorm2d(self.out_dim))

        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        points_ = list(itertools.product(range(self.resolution2), range(self.resolution2)))
        offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                step = math.ceil(self.resolution / self.resolution2)
                offset = (abs(p1[0] * step - p2[0]), abs(p1[1] * step - p2[1]))
                if offset not in offsets:
                    offsets[offset] = len(offsets)
                idxs.append(offsets[offset])
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, len(offsets)))
        self.register_buffer("attention_bias_idxs", torch.LongTensor(idxs).view(len(points_), len(points)))

    @torch.no_grad()
    def train(self, mode=True):
        """Cache relative attention bias for evaluation mode."""
        super().train(mode)
        if mode and hasattr(self, "ab"):
            del self.ab
        elif not mode:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        """Apply attention while reducing spatial resolution by two."""
        b, _, h, w = x.shape
        n = h * w
        h2, w2 = math.ceil(h / 2), math.ceil(w / 2)
        n2 = h2 * w2

        q = self.q(x).flatten(2).reshape(b, self.num_heads, self.key_dim, n2).permute(0, 1, 3, 2)
        k = self.k(x).flatten(2).reshape(b, self.num_heads, self.key_dim, n)
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape(b, self.num_heads, self.d, n).permute(0, 1, 3, 2)

        bias = self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab
        attn = ((q @ k) * self.scale + bias).softmax(dim=-1)
        x = (attn @ v).transpose(2, 3).reshape(b, self.dh, h2, w2) + v_local
        return self.proj(x)


class StageDownsample(nn.Module):
    """Stage transition block with optional GCA-assisted downsampling."""

    def __init__(self, in_chans, embed_dim, patch_size=3, stride=2, padding=1, norm_layer=nn.BatchNorm2d,
                 use_gca_down=False, resolution=None, act_layer=nn.ReLU):
        """Create a convolutional or attention-assisted embedding path."""
        super().__init__()
        self.use_gca_down = use_gca_down
        if use_gca_down:
            self.gca_down = GCADownBlock(in_chans, out_dim=embed_dim, resolution=resolution, act_layer=act_layer)
            self.conv = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
            self.bn = norm_layer(embed_dim)
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
            self.norm = norm_layer(embed_dim)

    def forward(self, x):
        """Project features to the next stage resolution and channel width."""
        if self.use_gca_down:
            return self.gca_down(x) + self.bn(self.conv(x))
        return self.norm(self.proj(x))


class CFFNCore(nn.Module):
    """Convolutional feed-forward projection used inside C-FFN blocks."""

    def __init__(self, in_features, hidden_features, out_features=None, act_layer=nn.GELU, drop=0.0, mid_conv=False):
        """Create pointwise and optional depthwise middle convolution layers."""
        super().__init__()  # hidden_features=4*in_features
        out_features = out_features or in_features
        self.mid_conv = mid_conv
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        if mid_conv:
            self.mid = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features)
            self.mid_norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.norm2 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Apply convolutional MLP projection."""
        x = self.act(self.norm1(self.fc1(x)))
        if self.mid_conv:
            x = self.act(self.mid_norm(self.mid(x)))
        x = self.drop(x)
        x = self.norm2(self.fc2(x))
        return self.drop(x)


class CFFN(nn.Module):
    """Conv Feed-Forward Network residual block."""

    def __init__(self, dim, mlp_ratio=4.0, act_layer=nn.GELU, drop=0.0, drop_path=0.0,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
        """Initialize the residual MLP and optional layer scale."""
        super().__init__()
        self.cffn_core = CFFNCore(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop, mid_conv=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim).view(1, dim, 1, 1))

    def forward(self, x):
        """Apply residual feed-forward processing."""
        if self.use_layer_scale:
            return x + self.drop_path(self.layer_scale_2 * self.cffn_core(x))
        return x + self.drop_path(self.cffn_core(x))


class GCAFFNBlock(nn.Module):
    """GCA Module followed by a C-FFN residual block."""

    def __init__(self, dim, mlp_ratio=4.0, act_layer=nn.ReLU, drop=0.0, drop_path=0.0,
                 use_layer_scale=True, layer_scale_init_value=1e-5, resolution=7, stride=None):
        """Initialize token mixing attention and MLP branches."""
        super().__init__()
        self.gca = GCA(dim, resolution=resolution, act_layer=act_layer, stride=stride)
        self.cffn_core = CFFNCore(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop, mid_conv=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim).view(1, dim, 1, 1))
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim).view(1, dim, 1, 1))

    def forward(self, x):
        """Apply attention token mixing followed by MLP refinement."""
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.gca(x))
            x = x + self.drop_path(self.layer_scale_2 * self.cffn_core(x))
            return x
        x = x + self.drop_path(self.gca(x))
        return x + self.drop_path(self.cffn_core(x))


class ARCA(nn.Module):
    """Axis-wise Residual Coordinate Attention."""

    def __init__(self, channels, reduction=32, beta_init=0.0, act_layer=nn.GELU):
        """Create axis-wise attention projections and residual scaling."""
        super().__init__()
        mid_channels = max(8, channels // reduction)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            act_layer(),
        )
        self.conv_h = nn.Conv2d(mid_channels, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mid_channels, channels, kernel_size=1)
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, x):
        """Apply height and width coordinate attention as a residual calibration."""
        identity = x
        _, _, h, w = x.shape
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.shared(y)
        a_h, a_w = torch.split(y, [h, w], dim=2)
        a_w = a_w.transpose(2, 3)
        a_h = self.conv_h(a_h).sigmoid()
        a_w = self.conv_w(a_w).sigmoid()
        return identity + self.beta * identity * a_h * a_w


class BMC(nn.Module):
    """Blur-aware Multi-scale Context module."""

    def __init__(
        self,
        dim,
        act_layer=nn.GELU,
        drop=0.0,
        drop_path=0.0,
        ca_reduction=32,
        ca_beta_init=0.0,
        branch4_kernel=7,
        use_arca=False,
    ):
        """Create multi-scale depthwise branches with optional ARCA."""
        super().__init__()
        hidden_dim = dim * 2
        if hidden_dim % 4 != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by 4.")
        branch_dim = hidden_dim // 4

        self.expand = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            act_layer(),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(branch_dim, branch_dim, kernel_size=3, padding=1, groups=branch_dim, bias=False),
            nn.BatchNorm2d(branch_dim),
            act_layer(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(branch_dim, branch_dim, kernel_size=5, padding=2, groups=branch_dim, bias=False),
            nn.BatchNorm2d(branch_dim),
            act_layer(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(branch_dim, branch_dim, kernel_size=3, padding=2, dilation=2, groups=branch_dim, bias=False),
            nn.BatchNorm2d(branch_dim),
            act_layer(),
        )
        if branch4_kernel == 7:
            self.branch4 = nn.Sequential(
                nn.Conv2d(branch_dim, branch_dim, kernel_size=7, padding=3, groups=branch_dim, bias=False),
                nn.BatchNorm2d(branch_dim),
                act_layer(),
            )
        else:
            self.branch4 = nn.Sequential(
                nn.Conv2d(branch_dim, branch_dim, kernel_size=3, padding=3, dilation=3, groups=branch_dim, bias=False),
                nn.BatchNorm2d(branch_dim),
                act_layer(),
            )
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.Dropout(drop),
        )
        self.arca = (
            ARCA(dim, reduction=ca_reduction, beta_init=ca_beta_init, act_layer=act_layer)
            if use_arca
            else nn.Identity()
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def context(self, x):
        """Return the blur-aware multi-scale context branch output."""
        x1, x2, x3, x4 = torch.chunk(self.expand(x), 4, dim=1)
        enhanced = torch.cat(
            [self.branch1(x1), self.branch2(x2), self.branch3(x3), self.branch4(x4)],
            dim=1,
        )
        enhanced = self.project(enhanced)
        return self.arca(enhanced)

    def forward(self, x):
        """Apply multi-scale local context as a residual refinement."""
        enhanced = self.context(x)
        return x + self.drop_path(enhanced)


class SeparateGCABMCBlock(nn.Module):
    """Non-GBC block that keeps GCA-FFN and BMC as separate operations."""

    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        drop_path=0.0,
        resolution=7,
        stride=None,
        ca_beta_init=0.0,
        bmc_use_arca=False,
    ):
        """Create a BMC module followed by a standalone GCA-FFN block."""
        super().__init__()
        self.bmc = BMC(
            dim,
            act_layer=act_layer,
            drop_path=drop_path,
            ca_beta_init=ca_beta_init,
            use_arca=bmc_use_arca,
        )
        self.gca_ffn = GCAFFNBlock(
            dim,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            drop_path=drop_path,
            resolution=resolution,
            stride=stride,
        )

    def forward(self, x):
        """Apply BMC and GCA-FFN without using the GBC cascade."""
        return self.gca_ffn(self.bmc(x))


class GBCBlock(nn.Module):
    """Global-Blur Context Block with BMC local enhancement."""

    def __init__(
        self,
        dim,
        act_layer=nn.GELU,
        drop_path=0.0,
        resolution=7,
        stride=None,
        ca_beta_init=0.0,
        bmc_use_arca=False,
    ):
        """Create global attention and blur-aware context branches."""
        super().__init__()
        self.gca = GCA(dim, resolution=resolution, act_layer=act_layer, stride=stride)
        self.bmc = BMC(
            dim,
            act_layer=act_layer,
            drop_path=drop_path,
            ca_beta_init=ca_beta_init,
            use_arca=bmc_use_arca,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_scale_1 = nn.Parameter(1e-5 * torch.ones(dim).view(1, dim, 1, 1))

    def forward(self, x):
        """Mix global context, then refine local blur-aware context."""
        x = x + self.drop_path(self.layer_scale_1 * self.gca(x))
        return self.bmc(x)


def build_mar_stage(
    dim,
    index,
    layers,
    drop_path_rate,
    vit_num,
    resolution,
    e_ratios,
    act_layer=nn.GELU,
    use_bmc=True,
    use_gca=True,
    use_gbc=True,
    bmc_use_arca=False,
    gbc_stages=(1, 2),
    gbc_stage_ratio=0.5,
    ca_beta_init=0.0,
):
    """Build one MAR Backbone stage from C-FFN, GCA, and GBC blocks."""
    blocks = []
    total_depth = sum(layers)
    gbc_start = int(math.floor(layers[index] * (1.0 - gbc_stage_ratio)))
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / max(total_depth - 1, 1)
        mlp_ratio = e_ratios[str(index)][block_idx]
        use_bmc_block = use_bmc and index in gbc_stages and block_idx >= gbc_start
        use_gca_block = use_gca and index >= 2 and block_idx > layers[index] - 1 - vit_num
        if use_bmc_block and use_gca_block and use_gbc:
            blocks.append(
                GBCBlock(
                    dim,
                    act_layer=act_layer,
                    drop_path=block_dpr,
                    resolution=resolution,
                    stride=2 if index == 2 else None,
                    ca_beta_init=ca_beta_init,
                    bmc_use_arca=bmc_use_arca,
                )
            )
        elif use_bmc_block and use_gca_block:
            blocks.append(
                SeparateGCABMCBlock(
                    dim,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    drop_path=block_dpr,
                    resolution=resolution,
                    stride=2 if index == 2 else None,
                    ca_beta_init=ca_beta_init,
                    bmc_use_arca=bmc_use_arca,
                )
            )
        elif use_bmc_block:
            blocks.append(
                BMC(
                    dim,
                    act_layer=act_layer,
                    drop_path=block_dpr,
                    ca_beta_init=ca_beta_init,
                    use_arca=bmc_use_arca,
                )
            )
        elif use_gca_block:
            blocks.append(
                GCAFFNBlock(
                    dim,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    drop_path=block_dpr,
                    resolution=resolution,
                    stride=2 if index == 2 else None,
                )
            )
        else:
            blocks.append(CFFN(dim, mlp_ratio=mlp_ratio, act_layer=act_layer, drop_path=block_dpr))
    return nn.Sequential(*blocks)


class MARBackbone(nn.Module):
    """Multi-level Artifact-Robust Feature Extraction Backbone."""

    def __init__(
        self,
        variant="s2",
        resolution=640,
        pretrained_path="",
        out_indices=(0, 1, 2, 3),
        ablation_mode="full",
        gbc_stages=(1, 2),
        gbc_stage_ratio=0.5,
        ca_beta_init=0.0,
    ):
        """Initialize the MAR feature extractor and shallow/deep ARCA calibrations."""
        super().__init__()
        self.ablation_mode, ablation_config = resolve_ablation_config(ablation_mode)
        variant = variant.lower()
        if variant not in MAR_WIDTHS:
            raise ValueError(f"Unsupported MAR Backbone variant: {variant}")

        self.variant = variant
        self.embed_dims = MAR_WIDTHS[variant]
        self.out_channels = [self.embed_dims[i] for i in out_indices]
        self.out_indices = set(out_indices)

        layers = MAR_DEPTHS[variant]
        e_ratios = EXPANSION_RATIOS[variant]
        vit_num = VIT_NUMS[variant]
        drop_path_rate = DROP_PATH_RATES[variant]

        self.stem = Stem(3, self.embed_dims[0], act_layer=nn.GELU)
        stages = []
        for i in range(len(layers)):
            stage_resolution = math.ceil(resolution / (2 ** (i + 2)))
            stages.append(
                build_mar_stage(
                    self.embed_dims[i],
                    i,
                    layers,
                    drop_path_rate=drop_path_rate,
                    vit_num=vit_num,
                    resolution=stage_resolution,
                    e_ratios=e_ratios,
                    act_layer=nn.GELU,
                    use_bmc=ablation_config["use_bmc"],
                    use_gca=ablation_config["use_gca"],
                    use_gbc=ablation_config["use_gbc"],
                    bmc_use_arca=ablation_config["bmc_use_arca"],
                    gbc_stages=gbc_stages,
                    gbc_stage_ratio=gbc_stage_ratio,
                    ca_beta_init=ca_beta_init,
                )
            )
            if i < len(layers) - 1:
                stages.append(
                    StageDownsample(
                        self.embed_dims[i],
                        self.embed_dims[i + 1],
                        resolution=stage_resolution,
                        use_gca_down=ablation_config["use_gca_down"] and i >= 2,
                        act_layer=nn.GELU,
                    )
                )
        self.stages = nn.ModuleList(stages)

        self.shallow_arca = (
            ARCA(self.embed_dims[0], beta_init=ca_beta_init, act_layer=nn.GELU)
            if ablation_config["use_arca"]
            else nn.Identity()
        )
        self.deep_arca = (
            ARCA(self.embed_dims[3], beta_init=ca_beta_init, act_layer=nn.GELU)
            if ablation_config["use_arca"]
            else nn.Identity()
        )

        if pretrained_path:
            self.load_pretrained(pretrained_path)

    def load_pretrained(self, pretrained_path):
        """Load compatible pretrained weights while skipping unmatched heads."""
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith("head.") and not k.startswith("dist_head.") and not k.startswith("norm.")
        }
        model_dict = self.state_dict()
        matched_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(matched_dict)
        self.load_state_dict(model_dict, strict=False)
        print(
            f"Loaded MAR Backbone compatible weights from {pretrained_path}. "
            f"Matched keys: {len(matched_dict)}, skipped keys: {len(state_dict) - len(matched_dict)}"
        )

    def forward(self, x):
        """Return selected multi-scale feature maps for the detection neck."""
        x = self.stem(x)   #[B,3,640,640]->[B,32,160,160]
        outs = []
        stage_idx = 0
        for block_idx, block in enumerate(self.stages):
            x = block(x)
            if block_idx % 2 == 0:
                if stage_idx == 0:
                    x = self.shallow_arca(x)
                elif stage_idx == 3:
                    x = self.deep_arca(x)
                if stage_idx in self.out_indices:
                    outs.append(x)
                stage_idx += 1
        return tuple(outs)




