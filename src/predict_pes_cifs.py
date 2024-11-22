from __future__ import annotations

import os
import shutil
import warnings

import numpy as np
import pytorch_lightning as pl
from dgl.data.utils import split_dataset
from mp_api.client import MPRester
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import diep
from diep.ext.pymatgen import Structure2Graph, get_element_list
from diep.graph.data import MGLDataset, MGLDataLoader, collate_fn_efs
from diep.models import M3GNet
from diep.utils.training import PotentialLightningModule
import argparse
from dataset import get_dataset
import glob
from pymatgen.core import Structure

# To suppress warnings for clearer output
warnings.simplefilter("ignore")
import torch

torch.set_default_device("cpu")
print("Started prediction..")

model = diep.load_model("my_models/m3gnet_pes")

element_types = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
]
model.calc_stresses = True
converter = Structure2Graph(element_types=element_types, cutoff=5.0)

# with torch.no_grad():
g = glob.glob("test_structures/*.cif")
for gg in g:
    s = Structure.from_file(gg)
    graph, state_feats_default, state_attr = converter.get_graph(s)
    result, f, st, h = model(graph, state_feats_default)
    print("Energy", result, "Force", f, "Stress", st, "H", h)
