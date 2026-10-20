from jarvis.db.figshare import data as jdata
from jarvis.core.atoms import Atoms
import numpy as np
import pickle
import pandas as pd
import ase.data
import ase.io
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm
import json, glob


def get_jarvis_forme_dataset(args):
    dft_3d = jdata("dft_3d")
    prop = "formation_energy_peratom"
    max_samples = 50
    # f = open("id_prop.csv", "w")
    count = 0
    structures = []
    energies = []
    forces = []
    stresses = []
    cnt = 0
    for i in dft_3d:
        target = i[prop]
        if target != "na":
            # if cnt > 10000:
            #     break
            # cnt +=1
            s = Atoms.from_dict(i["atoms"]).pymatgen_converter()
            structures += [s]
            energies += [target]
            forces += [np.zeros((len(s), 3)).tolist()]
            stresses += [np.zeros((3, 3)).tolist()]
    print("Number of samples in Jarvis dataset: ", len(structures))
    return structures, energies, forces, stresses


def get_mp_pes_dataset(args=None, full_dataset=False, exclude_force_outliers=False, forcelimit=10):
    print("full_dataset", full_dataset, "exclude_force_outliers", exclude_force_outliers)
    with open("MPF.2021.2.8/19470599/block_0.p", "rb") as f:
        data = pickle.load(f)
    with open("MPF.2021.2.8/19470599/block_1.p", "rb") as f:
        data.update(pickle.load(f))
    df = pd.DataFrame.from_dict(data)

    cnt = 0
    dataset_train = []

    for idx, item in df.items():
        for iid in range(len(item["energy"])):
            dataset_train.append(
                {
                    "atoms": item["structure"][iid],
                    "energy": item["energy"][iid],
                    "force": np.array(item["force"][iid]),
                    "stress": np.array(item["stress"][iid]),
                }
            )

    structures = []
    energies = []
    forces = []
    stresses = []
    for item in dataset_train:

        if not full_dataset:
            if cnt > 10000:
                break
        if exclude_force_outliers:
            n = np.abs(item["force"]).max()
        else:
            n = 0

        if n < forcelimit:
            cnt += 1
            structures += [item["atoms"]]
            # print(item['atoms'])
            energies += [item["energy"]]
            forces += [item["force"].tolist()]
            stresses += [item["stress"].tolist()]
    print("Number of samples in MP-PES dataset: ", len(structures))
    return structures, energies, forces, stresses


def get_mp_pes_dataset_json(args=None):
    import json

    with open("MPF.2021.2.8/19470599/block_0.p", "rb") as f:
        data = pickle.load(f)

    with open("MPF.2021.2.8/19470599/block_1.p", "rb") as f:
        data.update(pickle.load(f))

    cnt = 0
    dataset_train = []

    for id in data:
        # print(id)
        item = data[id]
        for iid in range(len(item["energy"])):
            dataset_train.append(
                {
                    "atoms": item["structure"][iid],
                    "energy": item["energy"][iid],
                    "force": np.array(item["force"][iid]),
                    "stress": np.array(item["stress"][iid]),
                }
            )

    structures = []
    energies = []
    forces = []
    stresses = []
    for item in dataset_train:
        # if args.lg:
        #     if cnt > 30000:
        #         break
        #     cnt +=1
        structures += [item["atoms"]]
        # print(item['atoms'])
        energies += [item["energy"]]
        forces += [item["force"].tolist()]
        stresses += [item["stress"].tolist()]

    print("Number of samples in MP-PES dataset: ", len(structures))
    return structures, energies, forces, stresses


def get_mptrj_dataset(args=None):
    print("Start loading MPTrj dataset")
    atoms_list = ase.io.read("MACE_training_data/training_data.xyz", index=":")
    print("Finished loading MPTrj dataset")
    structures = []
    energies = []
    forces = []
    stresses = []
    n = 0
    for struct in tqdm(atoms_list):
        n += 1
        print("Added", n, "structures")
        structures += [AseAtomsAdaptor.get_structure(struct)]
        energies += [struct.get_total_energy()]
        forces += [struct.get_forces().tolist()]
        stresses += [struct.get_stress(voigt=False).tolist()]

    print("Number of samples in MP-Trj dataset: ", len(structures))
    return structures, energies, forces, stresses


def get_aimdtraj_json(args=None, full_dataset=False, exclude_force_outliers=False, forcelimit=10):
    structures = []
    energies = []
    forces = []
    stresses = []
    g = glob.glob("AIMD_dataset/split/*.json")

    cnt = 0
    i = 0
    structures = []
    energies = []
    forces = []
    stresses = []
    for gg in g:
        i += 1
        print("Opened", i, "items")
        f = open(gg)
        data = json.load(f)
        f.close()

        if not full_dataset:
            if cnt > 10000:
                break
        n = 0
        # if exclude_force_outliers:
        #     n = np.abs(item["force"]).max()
        # else:
        #     n = 0

        if n < forcelimit:
            cnt += 1
            structures += data["structures"]
            # print(item['atoms'])
            energies += data["energies"]
            forces += data["forces"]
            stresses += data["stresses"]
            print("structures len:", len(data["structures"]), len(structures))
        cnt += 100
    print("Number of samples in dataset: ", len(structures))
    return structures, energies, forces, stresses


def get_megnet_forme_dataset(args):
    structures = []
    energies = []
    forces = []
    stresses = []
    return structures, energies, forces, stresses


def get_dataset(args=None, full_dataset=False, exclude_force_outliers=False, forcelimit=10):
    if args is None:
        return get_mp_pes_dataset_json()
    if args.dataset == "jarvis":
        return get_jarvis_forme_dataset(args)
    if args.dataset == "mp_pes":
        return get_mp_pes_dataset(args, full_dataset, exclude_force_outliers, forcelimit=forcelimit)
    if args.dataset == "mptraj":
        return get_mptrj_dataset(args)
    if args.dataset == "aimdtraj":
        return get_aimdtraj_json(args, full_dataset, exclude_force_outliers, forcelimit=forcelimit)


# def get_props_dataset(args):
#     dft_3d = jdata(args.dataset)
#     prop = args.props
#     structures = []
#     targets = []
#     forces = []
#     stresses = []

#     for i in dft_3d:
#         target = i[prop]
#         if target != "na":
#             s = Atoms.from_dict(i["atoms"]).pymatgen_converter()
#             structures += [s]
#             targets += [target]
#             forces += [np.zeros((len(s), 3)).tolist()]
#             stresses += [np.zeros((3, 3)).tolist()]

#     print('Number of samples in '+ args.dataset +' dataset: ', len(structures))

#     return structures, targets, forces, stresses


def get_props_dataset(args):
    dft_3d = jdata(args.dataset)
    prop = args.props
    structures = []
    targets = []
    ct = 0
    for i in dft_3d:
        # if ct > 1000:
        #     break
        # ct +=1
        target = i[prop]
        if target != "na":
            s = Atoms.from_dict(i["atoms"]).pymatgen_converter()
            structures += [s]
            targets += [target]

    print("Number of samples in " + args.dataset + " dataset: ", len(structures))

    return structures, targets
