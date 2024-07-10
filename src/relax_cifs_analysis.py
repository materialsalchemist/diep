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

import matgl
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_efs
from matgl.models import M3GNet
from matgl.utils.training import PotentialLightningModule
import argparse
from dataset import get_dataset
import glob
from pymatgen.core import Structure
from matgl.ext.ase import Relaxer
from pymatgen.analysis.structure_matcher import StructureMatcher

# To suppress warnings for clearer output
warnings.simplefilter("ignore")
import ntpath

sm = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)

folder = "test_structures/"
materials_folders = glob.glob(folder + "/*")
f = open("test_structures/results.csv", "w")
for materials_folder in materials_folders:
    output_folder_mym3gnet = materials_folder + "/"

    cifs = glob.glob(materials_folder + "/CIFs/*.cif")
    for cif in cifs:
        try:
            cif_file_name = ntpath.basename(cif)
            structure = Structure.from_file(cif)
            structure_diep = Structure.from_file(output_folder_mym3gnet + "diep/" + cif_file_name)
            d_diep = sm.get_rms_dist(structure, structure_diep)
            structure_mym3gnet = Structure.from_file(output_folder_mym3gnet + "mym3gnet/" + cif_file_name)
            d_mym3gnet = sm.get_rms_dist(structure, structure_mym3gnet)
            if d_mym3gnet:
                d_mym3gnet = d_mym3gnet[0]
            if d_diep:
                d_diep = d_diep[0]
            f.write(materials_folder + "," + cif_file_name + "," + str(d_diep) + "," + str(d_mym3gnet) + "\n")
        except:
            continue
f.close()
