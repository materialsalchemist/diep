from __future__ import annotations

import os
import shutil
import warnings

import numpy as np
import pytorch_lightning as pl
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

# To suppress warnings for clearer output
warnings.simplefilter("ignore")
import torch

# torch.set_default_device('cuda')

random_state = 42

parser = argparse.ArgumentParser(description="Train ")
parser.add_argument("--lg", action=argparse.BooleanOptionalAction, help="Create line graph before training")
parser.add_argument("--uses", action=argparse.BooleanOptionalAction, help="Include stress")
parser.add_argument("--sw", type=float, default=0.1, help="Stress weight loss")
parser.add_argument("--fw", type=float, default=1, help="Stress weight loss")
parser.add_argument("--full_dataset", action=argparse.BooleanOptionalAction, help="full dataset")
parser.add_argument("--exclude_force_outliers", action=argparse.BooleanOptionalAction, help="exclude_force_outliers")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--max_n", type=int, default=3, help="max_n")
parser.add_argument("--max_l", type=int, default=3, help="max_l")
parser.add_argument("--model", type=str, default="dft", help="Model")
parser.add_argument("--dataset", type=str, default="mp_pes", help="Dataset")
parser.add_argument("--forcelimit", type=float, default=10, help="Limit of forces to include in training")
args = parser.parse_args()

if args.uses:
    swnanme = "_sw_" + str(args.sw)
else:
    swnanme = ""

if args.full_dataset:
    full_dataset = True
else:
    full_dataset = False
if args.exclude_force_outliers:
    exclude_force_outliers = True
else:
    exclude_force_outliers = False

epochs = 500

if args.lg:
    bs = 1
    lg_name = "_lg"
else:
    bs = 1
    lg_name = "_no_lg"

if args.model == "dft":
    model_name = "DFT"
else:
    model_name = "M3GNet"

if args.dataset == "jarvis":
    args.uses = False
    args.fw = 0

logname = (
    "PES_2_"
    + model_name
    + "_training"
    + lg_name
    + swnanme
    + "_"
    + args.dataset
    + "_fw_"
    + str(args.fw)
    + "_lr_"
    + str(args.lr)
    + "_full_dataset_"
    + str(full_dataset)
    + "_exclude_force_outliers_"
    + str(exclude_force_outliers)
    + "_epochs_"
    + str(epochs)
    + "_max_l_"
    + str(args.max_l)
    + "_max_n_"
    + str(args.max_n)
    + "_forcelimit_"
    + str(args.forcelimit)
)

print(logname, bs, epochs)

structures, energies, forces, stresses = get_dataset(
    args=args, full_dataset=full_dataset, exclude_force_outliers=exclude_force_outliers
)
structures = [s.as_dict() for s in structures]
print("Read dataset")

from pymatgen.core import Structure,PeriodicSite

# Same shuffle and split code in DGL:
def split(dataset, frac_list):
    print('Splitting dataset of size',len(dataset))
    frac_list = np.asarray(frac_list)
    num_data = len(dataset)
    indices = np.random.RandomState(seed=random_state).permutation(num_data)
    lengths = (num_data * frac_list).astype(int)
    lengths[-1] = num_data - np.sum(lengths[:-1])
    train = []
    val = []
    test = []
    print('Splitting data to lengths',lengths)
    for i in range(0, lengths[0]):
        train += [dataset[indices[i]]]
    for i in range(lengths[0], lengths[0] + lengths[1]):
        val += [dataset[indices[i]]]
    for i in range(lengths[0] + lengths[1], num_data):
        test += [dataset[indices[i]]]
    return train, val, test


train_energies, val_energies, test_energies = split(
    energies,
    frac_list=[0.9, 0.05, 0.05],
)
train_forces, val_forces, test_forces = split(
    forces,
    frac_list=[0.9, 0.05, 0.05],
)
train_stresses, val_stresses, test_stresses = split(
    stresses,
    frac_list=[0.9, 0.05, 0.05],
)
train_structures, val_structures, test_structures = split(
    structures,
    frac_list=[0.9, 0.05, 0.05],
)
train = {"structures": train_structures, "energies": train_energies, "forces": train_forces, "stresses": train_stresses}
val = {"structures": val_structures, "energies": val_energies, "forces": val_forces, "stresses": val_stresses}
test = {"structures": test_structures, "energies": test_energies, "forces": test_forces, "stresses": test_stresses}
logname = "./"
import json

f = open(logname + "/train_data.json", "w")
json.dump(train, f)
f.close()
f = open(logname + "/val_data.json", "w")
json.dump(val, f)
f.close()
f = open(logname + "/test_data.json", "w")
json.dump(test, f)
f.close()
