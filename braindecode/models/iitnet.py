# Authors: Seo et al. 2020 (original IITNet architecture)
#          braindecode developers (braindecode adaptation)
#
# License: BSD (3-clause)
# Adapted from https://github.com/gist-ailab/IITNet-official (MIT); see
# NOTICE.txt.
import torch
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import _disable_batch_norm_training_if_batch_size_one

# Number of residual blocks per stage for the supported ResNet depths, as in
# the reference implementation (gist-ailab/IITNet-official, models/resnet.py).
_LAYER_CONFIG = {
    18: (2, 2, 2, 2),
    34: (3, 4, 6, 3),
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
}


class IITNet(EEGModuleMixin, nn.Module, license="mit"):
    r"""IITNet from Seo et al. (2020) [Seo2020IITNet]_.

    :bdg-success:`Convolution` :bdg-secondary:`Recurrent`

    IITNet (Intra- and inter-epoch temporal context network) scores raw
    single-channel EEG for sleep staging in two parts:

    1. **Intra-epoch features** — a 1-D ResNet (a convolutional stem and four
       residual stages with max-pooling in between) compresses the window
       into a short sequence of feature vectors, one per remaining time
       position.
    2. **Temporal context** — a two-layer bidirectional LSTM reads that
       feature sequence; the last forward state and the first backward state
       are concatenated and mapped to per-window sleep-stage logits by
       ``final_layer``.

    `License <https://github.com/gist-ailab/IITNet-official/blob/master/LICENSE>`_

    .. rubric:: Single-window adaptation

    .. note::

       The reference implementation consumes an explicit **sequence of
       epochs**: the ResNet runs per epoch and the BiLSTM spans epochs. Like
       :class:`~braindecode.models.DeepSleepNet`, this port maps braindecode's
       single-window interface onto the architecture: the ResNet feature
       sequence **of the window itself** (one feature vector per time
       position left by the strided stem and stages) is the BiLSTM input
       sequence, so the BiLSTM models the within-window temporal context.
       To score sequences of consecutive epochs, batch them externally.

       **Channels.** The reference reads a single EEG channel through
       :class:`~torch.nn.Conv1d`. This port follows DeepSleepNet and
       convolves each of the ``n_chans`` channels with shared weights
       (temporal kernels of extent one over channels); the per-position
       feature vectors of all channels are concatenated into the BiLSTM
       input.

       **Block forward.** The reference's ``BasicBlock`` applies its first
       convolution and batch norm twice and never uses ``conv2``/``bn2``
       (parameters without gradients). This port applies
       conv1 → conv2 as intended.

       **BiLSTM input size.** The reference hard-codes a 128-dim BiLSTM
       input, which only matches its default 50-layer configuration. This
       port sizes the BiLSTM from the actual ResNet output width
       (``stage_planes[-1] * block expansion * n_chans``).

       **Device.** The reference initializes its LSTM hidden state with
       ``torch.zeros(...).cuda()``. This port lets :class:`torch.nn.LSTM`
       default-initialize its hidden state, so the model is device-agnostic.

    Parameters
    ----------
    activation : type[nn.Module], default=nn.ReLU
        Activation class of the ResNet stages.
    return_feats : bool, default=False
        If True, return the concatenated BiLSTM states (features before the
        final linear layer).
    drop_prob : float, default=0.5
        Dropout probability on the ResNet feature sequence.
    resnet_num_layers : int, default=50
        Depth of the 1-D ResNet feature extractor; 18 and 34 use basic
        blocks, 50, 101 and 152 bottleneck blocks. The released
        configuration uses 50.
    stage_planes : tuple of int, default=(16, 16, 32, 32)
        Width (before the block expansion) of the stem and of the four
        residual stages, as in the reference implementation.
    bilstm_hidden_size : int, default=128
        Hidden size of the BiLSTM (per direction).
    bilstm_num_layers : int, default=2
        Number of stacked BiLSTM layers.
    bidirectional : bool, default=True
        If True, use a bidirectional LSTM and concatenate the last forward
        and first backward states; otherwise use the last state of a
        unidirectional LSTM.

    References
    ----------
    .. [Seo2020IITNet] Seo, H., Kang, H., Choi, H., & Park, H. (2020).
       Intra- and inter-epoch temporal context network (IITNet) for automatic
       sleep scoring on raw single-channel EEG. Biomedical Signal Processing
       and Control, 61, 103797.
       https://arxiv.org/abs/1902.06562
    """

    def __init__(
        self,
        n_outputs=5,
        return_feats=False,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        activation: type[nn.Module] = nn.ReLU,
        drop_prob: float = 0.5,
        resnet_num_layers: int = 50,
        stage_planes: tuple[int, ...] = (16, 16, 32, 32),
        bilstm_hidden_size: int = 128,
        bilstm_num_layers: int = 2,
        bidirectional: bool = True,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        self.return_feats = return_feats
        self.bidirectional = bidirectional
        self.bilstm_hidden_size = bilstm_hidden_size

        if resnet_num_layers not in _LAYER_CONFIG:
            raise ValueError(
                f"resnet_num_layers={resnet_num_layers} is not one of the "
                f"supported ResNet depths {sorted(_LAYER_CONFIG)}."
            )
        stage_planes = tuple(stage_planes)
        if len(stage_planes) != 4:
            raise ValueError(
                "stage_planes must have one width per residual stage, got "
                f"{stage_planes}."
            )

        # The reference's deeper configurations use bottleneck blocks.
        block = _BasicBlock1d if resnet_num_layers in (18, 34) else _Bottleneck1d
        n_blocks = _LAYER_CONFIG[resnet_num_layers]

        self.initial_layer = nn.Sequential(
            nn.Conv2d(
                1,
                stage_planes[0],
                kernel_size=(1, 7),
                stride=(1, 2),
                padding=(0, 3),
                bias=False,
            ),
            nn.BatchNorm2d(stage_planes[0]),
            activation(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
        )
        inplanes = stage_planes[0]
        self.layer1, inplanes = self._make_stage(
            block, inplanes, stage_planes[0], n_blocks[0], 1, activation
        )
        self.layer2, inplanes = self._make_stage(
            block, inplanes, stage_planes[1], n_blocks[1], 2, activation
        )
        # The reference max-pools between its second and third stages.
        self.mid_pool = nn.MaxPool2d(
            kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)
        )
        self.layer3, inplanes = self._make_stage(
            block, inplanes, stage_planes[2], n_blocks[2], 2, activation
        )
        self.layer4, inplanes = self._make_stage(
            block, inplanes, stage_planes[3], n_blocks[3], 2, activation
        )
        self.dropout = nn.Dropout(drop_prob)

        if _feature_seq_len(self.n_times) <= 0:
            raise ValueError(
                f"n_times={self.n_times} is too small for the configured "
                "ResNet stages. Increase n_times or reduce the stage strides."
            )

        self.bilstm = nn.LSTM(
            input_size=inplanes * self.n_chans,
            hidden_size=bilstm_hidden_size,
            num_layers=bilstm_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.to_sequence = Rearrange(
            "batch features nchans ntimes -> batch ntimes (features nchans)"
        )

        self.features_extractor = nn.Identity()
        self.len_last_layer = bilstm_hidden_size * (2 if bidirectional else 1)
        self.final_layer = (
            nn.Identity()
            if return_feats
            else nn.Linear(self.len_last_layer, self.n_outputs)
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    @staticmethod
    def _make_stage(block, inplanes, planes, n_blocks, stride, activation):
        """Build one residual stage and return it with its output width."""
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=(1, 1),
                    stride=(1, stride),
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(inplanes, planes, stride, downsample, activation)]
        inplanes = planes * block.expansion
        layers.extend(
            block(inplanes, planes, activation=activation)
            for _ in range(n_blocks - 1)
        )
        return nn.Sequential(*layers), inplanes

    @_disable_batch_norm_training_if_batch_size_one
    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)

        f = self.initial_layer(x)
        f = self.layer1(f)
        f = self.layer2(f)
        f = self.mid_pool(f)
        f = self.layer3(f)
        f = self.layer4(f)
        f = self.dropout(f)
        f = self.to_sequence(f)

        out, _ = self.bilstm(f)
        if self.bidirectional:
            out = torch.cat(
                (
                    out[:, -1, : self.bilstm_hidden_size],
                    out[:, 0, self.bilstm_hidden_size :],
                ),
                dim=1,
            )
        else:
            out = out[:, -1, :]

        return self.final_layer(self.features_extractor(out))


def _conv_out(width, kernel, stride, padding):
    """Output width after one conv/pool layer: ``(w + 2p - k) // s + 1``."""
    return (width + 2 * padding - kernel) // stride + 1


def _feature_seq_len(n_times):
    """Length of the per-position feature sequence left by the ResNet.

    Strided stages and pools halve the width: the stem convolution and its
    max-pool, then layer2, the mid-pool and layer3/4. The stride-2 block
    convolutions (kernel 3, padding 1) and their 1-sample downsample
    convolutions yield the same width.
    """
    width = _conv_out(_conv_out(n_times, 7, 2, 3), 3, 2, 1)  # stem conv + pool
    width = _conv_out(width, 3, 2, 1)  # layer2 (layer1 keeps the width)
    width = _conv_out(width, 3, 2, 1)  # mid-pool
    width = _conv_out(width, 3, 2, 1)  # layer3
    width = _conv_out(width, 3, 2, 1)  # layer4
    return width


def _conv3x1(in_planes, out_planes, stride=1):
    """Convolution over time only (kernel extent one over channels)."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=(1, 3),
        stride=(1, stride),
        padding=(0, 1),
        bias=False,
    )


class _BasicBlock1d(nn.Module):
    """Temporal BasicBlock: two 3-sample convolutions and a shortcut.

    Parameters
    ----------
    inplanes : int
        Input channels.
    planes : int
        Output channels (before ``expansion``).
    stride : int
        Temporal stride of the first convolution and of the shortcut.
    downsample : nn.Module or None
        Shortcut projection when the input cannot be added as is.
    activation : type[nn.Module]
        Activation function class.
    """

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        activation: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.conv1 = _conv3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x1(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.activation = activation()
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        return self.activation(out + identity)


class _Bottleneck1d(nn.Module):
    """Temporal Bottleneck: 1x1 → 3x3 → 1x1 convolutions and a shortcut.

    Parameters
    ----------
    inplanes : int
        Input channels.
    planes : int
        Intermediate channels; the block outputs ``expansion * planes``.
    stride : int
        Temporal stride of the 3-sample convolution and of the shortcut.
    downsample : nn.Module or None
        Shortcut projection when the input cannot be added as is.
    activation : type[nn.Module]
        Activation function class.
    """

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        activation: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x1(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=(1, 1), bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.activation = activation()
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        return self.activation(out + identity)
