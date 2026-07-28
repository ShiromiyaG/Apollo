###
# Author: Kai Li
# Date: 2021-06-18 16:53:49
###
"""Utility exports.

Only the handful of helpers the training and inference entry points actually use
are imported eagerly. The rest of this package is legacy ESPnet-derived code that
nothing in the repo calls but that pulls in `torch_complex`, `librosa` and friends
at import time -- keeping it lazy is what lets a plain
``pip install -r requirements.txt`` be enough to run the model.
"""

import importlib

from .lightning_utils import (
    BatchesProcessedColumn,
    MyMetricsTextColumn,
    MyRichProgressBar,
    RichProgressBarTheme,
    print_only,
)
from .parser_utils import (
    instantiate,
    isfloat,
    isint,
    parse_args_as_dict,
    prepare_parser_from_dict,
    str2bool,
    str2bool_arg,
    str_int_float,
)
from .pylogger import RankedLogger

# name -> module it lives in, resolved on first access
_LAZY = {
    "STFT": "stft",
    "is_complex": "complex_utils",
    "is_torch_complex_tensor": "complex_utils",
    "new_complex_like": "complex_utils",
    "get_layer": "get_layer_from_string",
    "InversibleInterface": "inversible_interface",
    "make_pad_mask": "nets_utils",
    "pad_x_to_y": "torch_utils",
    "shape_reconstructed": "torch_utils",
    "tensors_to_device": "torch_utils",
    "wav_chunk_inference": "separator",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(list(globals()) + list(_LAZY))


__all__ = [
    "RankedLogger",
    "instantiate",
    "print_only",
    "RichProgressBarTheme",
    "MyRichProgressBar",
    "BatchesProcessedColumn",
    "MyMetricsTextColumn",
    "prepare_parser_from_dict",
    "parse_args_as_dict",
    "str_int_float",
    "str2bool",
    "str2bool_arg",
    "isfloat",
    "isint",
    *_LAZY,
]
