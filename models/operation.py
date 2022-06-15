import torch.nn as nn
OPS = {
    "none": lambda C_in, C_out, stride, affine, dila,track_running_stats: Zero(
        C_in, C_out, stride
    ),
    "avg_pool_3": lambda C_in, C_out, stride, affine, dila,track_running_stats: POOLING(
        C_in, C_out, stride, "avg", affine, track_running_stats
    ),
    "max_pool_3": lambda C_in, C_out, stride, affine,dila, track_running_stats: POOLING(
        C_in, C_out, stride, "max", affine, track_running_stats
    ),
    "nor_conv_3": lambda C_in, C_out, stride, affine,dila, track_running_stats: ReLUConvBN(
        C_in,
        C_out,
        (3,1),
        (stride, stride),
        "same",
        (1, 1),
        affine,
        track_running_stats,
    ),
    "nor_conv_5": lambda C_in, C_out, stride, affine,dila, track_running_stats: ReLUConvBN(
        C_in,
        C_out,
        (5,1),
        (stride, stride),
        "same",
        (1, 1),
        affine,
        track_running_stats,
    ),
    "nor_conv_7": lambda C_in, C_out, stride, affine,dila, track_running_stats: ReLUConvBN(
        C_in,
        C_out,
        (7,1),
        (stride, stride),
        "same",
        (1, 1),
        affine,
        track_running_stats,
    ),
    "dil_conv_3": lambda C_in, C_out, stride, affine,dila, track_running_stats: ReLUConvBN(
        C_in,
        C_out,
        (3,1),
        (stride, stride),
        "same",
        (dila, 1),
        affine,
        track_running_stats,
    ),
    "dil_conv_5": lambda C_in, C_out, stride, affine, dila, track_running_stats: ReLUConvBN(
        C_in,
        C_out,
        (5,1),
        (stride, stride),
        "same",
        (dila, 1),
        affine,
        track_running_stats,
    ),
    "dil_conv_7": lambda C_in, C_out, stride, affine, dila,track_running_stats: ReLUConvBN(
        C_in,
        C_out,
        (7,1),
        (stride, stride),
        "same",
        (dila, 1),
        affine,
        track_running_stats,
    ),
    "skip_connect": lambda C_in, C_out, stride,dila, affine, track_running_stats: Identity()
    # if stride == 1 and C_in == C_out
    # else FactorizedReduce(C_in, C_out, stride, affine, track_running_stats),
}

class ReLUConvBN(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation,
        affine,
        track_running_stats=True,
    ):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_out,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=not affine,
            ),
            nn.BatchNorm2d(
                C_out, affine=affine, track_running_stats=track_running_stats
            ),
        )

    def forward(self, x):
        return self.op(x)

class POOLING(nn.Module):
    def __init__(
        self, C_in, C_out, stride, mode, affine=True, track_running_stats=True
    ):
        super(POOLING, self).__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(
                C_in, C_out, 1, 1, 0, 1, affine, track_running_stats
            )
        if mode == "avg":
            # self.op = nn.AvgPool2d(( 3,1 ), stride=stride, padding=1, count_include_pad=False)
            self.op = nn.AdaptiveAvgPool2d((None,None))
            
        elif mode == "max":
            self.op = nn.AdaptiveMaxPool2d((None,None))
            # self.op = nn.MaxPool2d(( 3 ,1), stride=stride, padding=1)
        else:
            raise ValueError("Invalid mode={:} in POOLING".format(mode))

    def forward(self, inputs):
        if self.preprocess:
            x = self.preprocess(inputs)
        else:
            x = inputs
        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super(Zero, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x):
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.0)
            else:
                return x[:, :, :: self.stride, :: self.stride].mul(0.0)
        else:
            shape = list(x.shape)
            shape[1] = self.C_out
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros

    def extra_repr(self):
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(**self.__dict__)

class SepConv(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation,
        affine,
        track_running_stats=True,
    ):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=not affine),
            nn.BatchNorm2d(
                C_out, affine=affine, track_running_stats=track_running_stats
            ),
        )

    def forward(self, x):
        return self.op(x)

class SepConv(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation,
        affine,
        track_running_stats=True,
    ):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=not affine),
            nn.BatchNorm2d(
                C_out, affine=affine, track_running_stats=track_running_stats
            ),
        )

    def forward(self, x):
        return self.op(x)