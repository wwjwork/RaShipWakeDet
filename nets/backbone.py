import torch
import torch.nn as nn


def autopad(k, p=None):
    """Return same-padding for a convolution kernel."""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p

class SiLU(nn.Module):  
    """SiLU activation implemented without relying on framework aliases."""

    @staticmethod
    def forward(x):
        """Apply the SiLU activation."""
        return x * torch.sigmoid(x)
    
class Conv(nn.Module):
    """Convolution, batch normalization, and activation block."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=SiLU()):
        """Create the convolution block."""
        super(Conv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        """Run convolution, batch normalization, and activation."""
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        """Forward path used after batch normalization fusion."""
        return self.act(self.conv(x))
    
class Multi_Concat_Block(nn.Module):
    """Stack convolutional branches and concatenate selected intermediate outputs."""

    def __init__(self, c1, c2, c3, n=4, e=1, ids=[0]):
        """Create the multi-branch concatenation block."""
        super(Multi_Concat_Block, self).__init__()
        c_ = int(c2 * e)
        
        self.ids = ids
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = nn.ModuleList(
            [Conv(c_ if i ==0 else c2, c2, 3, 1) for i in range(n)]
        )
        self.cv4 = Conv(c_ * 2 + c2 * (len(ids) - 2), c3, 1, 1)

    def forward(self, x):
        """Return concatenated branch features after projection."""
        x_1 = self.cv1(x)
        x_2 = self.cv2(x)
        
        x_all = [x_1, x_2]
        for i in range(len(self.cv3)):
            x_2 = self.cv3[i](x_2)
            x_all.append(x_2)
            
        out = self.cv4(torch.cat([x_all[id] for id in self.ids], 1))
        return out

class MP(nn.Module):
    """Max-pooling wrapper used by transition blocks."""

    def __init__(self, k=2):
        """Create a max-pooling layer with stride equal to kernel size."""
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        """Apply max pooling."""
        return self.m(x)
    
class Transition_Block(nn.Module):
    """Downsample and concatenate pooled and strided-convolution branches."""

    def __init__(self, c1, c2):
        """Create the two-branch downsampling block."""
        super(Transition_Block, self).__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.cv3 = Conv(c2, c2, 3, 2)
        
        self.mp  = MP()

    def forward(self, x):
        """Downsample both branches and concatenate them."""
        x_1 = self.mp(x)
        x_1 = self.cv1(x_1)
        
        x_2 = self.cv2(x)
        x_2 = self.cv3(x_2)
        
        return torch.cat([x_2, x_1], 1)
