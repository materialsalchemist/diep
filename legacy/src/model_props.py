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
from diep.ext.ase import Relaxer

# To suppress warnings for clearer output
warnings.simplefilter("ignore")
import torch
import ntpath

torch.set_default_device("cpu")

model = diep.load_model("my_models/diep_pes")
model = diep.load_model("../pretrained_models/M3GNet-MP-2021.2.8-PES")

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(pytorch_total_params)