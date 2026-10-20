import torch

torch.set_default_device("cpu")

import numpy as np
from dataset import get_dataset
from diep.ext.pymatgen import get_element_list

structures, energies, forces, stresses = get_dataset(full_dataset=True, exclude_force_outliers=True)
labels = {
    "energies": energies,
    "forces": forces,
    "stresses": stresses,
}
element_types = get_element_list(structures)

import json

d = {'element_types':element_types,'energies':energies}
json.dump(d,open('energies_element_types.json','w'))