###
# Author: Kai Li
# Date: 2021-06-03 18:29:46
###
"""Data modules.

Imported lazily: the original MUSDB/MoisesDB pipeline needs `h5py` and
`pytorch_lightning`, which nothing else in the package requires. Touching
`look2hear.datas.codec` should not drag those in.
"""

import importlib

_LAZY = {
    "MusdbMoisesdbDataModule": "musdb_moisesdb_datamodule",
    "MusdbMoisesdbDataset": "musdb_moisesdb_datamodule",
    "MusdbMoisesdbEval": "musdb_moisesdb_datamodule",
    "VocalDataModule": "vocal_datamodule",
    "VocalRestorationDataset": "vocal_datamodule",
    "PairedVocalEval": "vocal_datamodule",
    "VocalDegradation": "vocal_datamodule",
    "codec_simu": "codec",
    "encode_decode": "codec",
    "lowpass_resample": "codec",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(list(globals()) + list(_LAZY))


__all__ = list(_LAZY)
