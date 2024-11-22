import json
import diep
from diep.ext.pymatgen import Structure2Graph, get_element_list

from pymatgen.core import Structure
import gc
import torch
import ntpath
import torchmetrics
import sys
import numpy as np

f = open("test_data.json")
data = json.load(f)
f.close()
ff = open('test_data_atomnumbers','w')
for i in range(len(data["structures"])):
    s = Structure.from_dict(data["structures"][i])
    
    ff.write(str(len(s)) + "\n")
    ff.flush()
ff.close()
