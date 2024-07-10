import torch

torch.set_default_device("cpu")

import numpy as np
from dataset import get_dataset
from matgl.ext.pymatgen import get_element_list
from matgl.models import M3GNet
from matgl.utils.training import PotentialLightningModule
import json

d = json.load(open("energies_element_types.json"))
element_types = d["element_types"]
energies = d["energies"]

# model = ModelLightningModule.load_from_checkpoint('logs/trainfolder_DFT_summation__maxn_1_maxl_1_lr_0.001/epoch=1870-step=288134.ckpt',model=M3GNet(element_types=element_types,basis_expansion_type = 'dft',max_n=1,max_l=1,is_intensive=True),map_location=torch.device('cpu'))
# model.model.save('my_models/diip_pristine_mp_1to10')


model = PotentialLightningModule.load_from_checkpoint(
    "/home/abshe/matgl/src/logs/PES_2_DFT_training_lg_mp_pes_fw_1.0_lr_0.0001_full_dataset_True_exclude_force_outliers_True_epochs_500_max_l_3_max_n_3_forcelimit_10/epoch=230-step=1204203.ckpt",
    model=M3GNet(
        element_types=element_types,
        basis_expansion_type="dft",
        max_n=3,
        max_l=3,
        is_intensive=True,
        data_mean=np.mean(np.array(energies)),
        data_std=np.std(np.array(energies)),
    ),
    map_location=torch.device("cpu"),
)
model.model.save("my_models/diep_pes")

model = PotentialLightningModule.load_from_checkpoint(
    "logs/PES_2_M3GNet_training_lg_mp_pes_fw_1.0_lr_0.0001_full_dataset_True_exclude_force_outliers_True_epochs_500_max_l_3_max_n_3_forcelimit_10/epoch=135-step=708968.ckpt",
    model=M3GNet(
        element_types=element_types,
        basis_expansion_type="m3gnet",
        max_n=3,
        max_l=3,
        is_intensive=True,
        data_mean=np.mean(np.array(energies)),
        data_std=np.std(np.array(energies)),
    ),
    map_location=torch.device("cpu"),
)
model.model.save("my_models/m3gnet_pes")
