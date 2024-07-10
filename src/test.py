import json
import matgl
from matgl.ext.pymatgen import Structure2Graph, get_element_list

from pymatgen.core import Structure
import gc
import torch
import ntpath
import torchmetrics
import sys
import numpy as np
model_name = sys.argv[1]

torch.set_default_device("cuda")
if model_name == "mym3gnet":
    model = matgl.load_model("my_models/m3gnet_pes_final", map_location=torch.device("cuda"))

# model = matgl.load_model("/home/abshe/matgl/pretrained_models/M3GNet-MP-2021.2.8-DIRECT-PES", map_location=torch.device("cpu"))
elif model_name == "diep":
    model = matgl.load_model("my_models/diep_pes")
elif model_name == "m3gnet":
    model = matgl.load_model("../pretrained_models/M3GNet-MP-2021.2.8-PES")

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

f = open("test_data.json")
data = json.load(f)
f.close()

ff = open("test_data_" + model_name, "w")
print(data["structures"][0])
s = Structure.from_dict(data["structures"][0])
e_MAE = 0
f_MAE = 0
l = len(data["structures"])

j = 11

mae = torchmetrics.MeanAbsoluteError()

if 800 * (j + 1) > l:
    max_limit = l
else:
    max_limit = 800 * (j + 1)


force_counter = 0
for i in range(800 * j, max_limit):
    # for i in range(l):
    s = Structure.from_dict(data["structures"][i])
    force_counter += 3 * len(s)
    graph, state_feats_default, state_attr = converter.get_graph(s)
    if model_name == "diep":
        e, f, st, h = model(graph, state_feats_default)
    elif model_name == "m3gnet":
        e, f, st, h = model(graph, state_feats_default)
    elif model_name == "mym3gnet":
        e = model.predict_structure(s)
    delta_f = sum(sum([abs(x) for x in (f.detach().cpu().numpy() - np.array(data["forces"][i]))]))
    delta_e = abs(e - data["energies"][i]) / len(s)
    print(i, "Energy", data["energies"][i] / len(s), e / len(s))
    e_MAE += delta_e
    f_MAE += delta_f
    print(i, e_MAE / (i + 1))
    ff.write(str(delta_e) + "," + str(delta_f) + "\n")
    ff.flush()
e_MAE = e_MAE / len(data["structures"])
f_MAE = f_MAE / force_counter
print("e_MAE=", e_MAE)
ff.write("e_MAE=" + str(e_MAE) + ",f_MAE=" + str(f_MAE) + "\n")
ff.close()
