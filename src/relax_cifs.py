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

# To suppress warnings for clearer output
warnings.simplefilter("ignore")
import torch
import ntpath

torch.set_default_device("cpu")
print("Started prediction..")
pot = matgl.load_model("my_models/m3gnet_pes")
pot.calc_stresses = True
pot_m3gnet = matgl.load_model("../pretrained_models/M3GNet-MP-2021.2.8-PES")
pot_m3gnet.calc_stresses = True

# with torch.no_grad():
folder = 'test_structures/'
materials_folders = glob.glob(folder+"/*")
for materials_folder in materials_folders[3:]:
    print(materials_folder)
    output_folder_mym3gnet = materials_folder +'/mym3gnet'
    if not os.path.isdir(output_folder_mym3gnet):
        os.mkdir(output_folder_mym3gnet)
    output_folder_m3gnet = materials_folder +'/m3gnet'
    if not os.path.isdir(output_folder_m3gnet):
        os.mkdir(output_folder_m3gnet)
    
    cifs = glob.glob(materials_folder+'/CIFs/*.cif')
    for cif in cifs:
        structure = Structure.from_file(cif)
        outfn = ntpath.basename(cif).replace(".cif", "")
        print(f"Initial structure\n{structure}")
        print("Relaxing...")
        relaxer = Relaxer(potential=pot)
        relax_results = relaxer.relax(structure, fmax=0.01)
        final_structure = relax_results["final_structure"]
        outfn_mym3gnet = output_folder_mym3gnet + "/" + outfn + ".cif"
        final_structure.to(filename=outfn_mym3gnet)
        print(f"Structure written to {outfn_mym3gnet}!")

        # relaxer = Relaxer(potential=pot_m3gnet)
        # relax_results = relaxer.relax(structure, fmax=0.01)
        # final_structure = relax_results["final_structure"]
        # outfn_m3gnet = output_folder_m3gnet + "/" + outfn + ".cif"
        # final_structure.to(filename=outfn_m3gnet)
        # print(f"Structure written to {outfn_m3gnet}!")
