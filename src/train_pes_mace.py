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
import ase
import ase.data
import ase.io
import numpy as np
from diep.ext.pymatgen import Structure2Graph, get_element_list
import torch
from pymatgen.io.ase import AseAtomsAdaptor
from diep.graph.data import MGLDataset, MGLDataLoader, collate_fn_efs
from torch.utils.data import ConcatDataset, Dataset
from dgl.data.utils import split_dataset
from tqdm import tqdm

# To suppress warnings for clearer output
warnings.simplefilter("ignore")
import torch

devicestr = "cpu"
devicestr = "cuda"
torch.set_default_device(devicestr)
print("Started training..")

parser = argparse.ArgumentParser(description="Train ")
parser.add_argument("--lg", action=argparse.BooleanOptionalAction, help="Create line graph before training")
parser.add_argument("--uses", action=argparse.BooleanOptionalAction, help="Include stress")
parser.add_argument("--ew", type=float, default=1, help="Energy weight loss")
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

epochs = 10000

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
    + "_ew_"
    + str(args.ew)
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
try:
    os.mkdir(logname)
except OSError as error:
    print("Path " + logname + " exists.")


print(logname, bs, epochs)
gendevicestr = devicestr
# devicestr = "gpu"

file_path = "MACE_training_data/training_data.xyz"
atoms_list = ase.io.read(file_path, index=":")
atoms_list_id = np.arange(len(atoms_list))
train_spit_id = np.array_split(atoms_list_id, 32)
element_types = (
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
)
converter = Structure2Graph(element_types=element_types, cutoff=5.0)
datasets = []
for id, data_split in tqdm(enumerate(train_spit_id)):
    structures = []
    energies = []
    forces = []
    stresses = []
    print("Converting to Pymatgen structure")
    for aseatom_id in tqdm(data_split):
        aseatom = atoms_list[aseatom_id]
        structure = AseAtomsAdaptor.get_structure(aseatom)
        structures.extend([structure])
        energies.append(aseatom.get_potential_energy())
        forces.append(aseatom.get_forces().tolist())
        stresses.append(aseatom.get_stress(voigt=False).tolist())
    labels = {
        "energies": energies,
        "forces": forces,
        "stresses": stresses,
    }
    converter = Structure2Graph(element_types=element_types, cutoff=5.0)
    logname = "MACE_data_small/"
    dataset = MGLDataset(
        threebody_cutoff=4.0,
        structures=structures,
        converter=converter,
        labels=labels,
        filename=logname + f"dgl_graph_{id}.bin",
        filename_lattice=logname + f"lattice_{id}.pt",
        filename_line_graph=logname + f"dgl_line_graph_{id}.bin",
        filename_state_attr=logname + f"state_attr_{id}.pt",
        filename_labels=logname + f"labels_{id}.json",
        # save_dir = '/home/minhtrin/Code/Generative/diep/MACE_data/',
        name=f"MACE_{id}",
    )
    datasets.append(dataset)
datasets = ConcatDataset(datasets)
train_data, val_data, test_data = split_dataset(
    datasets,
    frac_list=[0.9, 0.05, 0.05],
    shuffle=True,
    random_state=42,
)
train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=collate_fn_efs,
    batch_size=32,
    num_workers=0,
    generator=torch.Generator(device=gendevicestr),
)
print("lengths:", len(train_loader), len(train_data))

model = M3GNet(
    element_types=element_types, is_intensive=True, basis_expansion_type=args.model, max_l=args.max_l, max_n=args.max_n
)
lit_module = PotentialLightningModule(
    model=model, energy_weight=args.ew, force_weight=args.fw, stress_weight=args.sw, lr=args.lr, include_line_graph=True
)
# If you wish to disable GPU or MPS (M1 mac) training, use the accelerator="cpu" kwarg.
logger = CSVLogger("logs", name=logname)
# Inference mode = False is required for calculating forces, stress in test mode and prediction mode
checkpoint_callback = ModelCheckpoint(dirpath="logs/" + logname, save_top_k=5, monitor="val_Total_Loss")
early_stop_callback = EarlyStopping(monitor="val_Total_Loss", patience=200, mode="min")
trainer = pl.Trainer(
    max_epochs=epochs, accelerator=devicestr, logger=logger, inference_mode=False, callbacks=[checkpoint_callback]
)
trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
# test the model, remember to set inference_mode=False in trainer (see above)
trainer.test(dataloaders=test_loader)
# save trained model
model_export_path = "./trained_model/"
model.save(model_export_path)

# load trained model
model = diep.load_model(path=model_export_path)
