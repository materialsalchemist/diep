from oganesson.genetic_algorithms import GA
from oganesson import OgStructure
import glob
import random
import ntpath
import shutil
import sys

print(sys.argv)
model = sys.argv[1]


def model_selector(model, filename):
    if model == "diep":
        folder_tag = model + "_2elements_" + filename
        model_src = model
    elif model == "mym3gnet":
        model_src = "/home/abshe/matgl/src/my_models/m3gnet_pes"
        folder_tag = "mym3gnet_2elements_" + filename
    elif model == "m3gnet":
        folder_tag = "m3gnet_2elements_" + filename
        model_src = model
    return model_src, folder_tag


log = open("test_ga/" + model + "_log", "w")
g = glob.glob("test_ga/sample_2elements/*.cif")
g_done = glob.glob("og_lab/ga_" + model + "_2elements_*")
g_done = [x.replace("og_lab/ga_" + model + "_2elements_", "") for x in g_done]
for gg in g:
    filename = ntpath.basename(gg).replace(".cif", "")
    model_src, folder_tag = model_selector(model, filename)

    if filename in g_done:
        log.write(">>>>>" + filename + " was done!\n")
        log.flush()
        continue
    og = OgStructure(file_name=gg)
    d = og().composition.as_dict()
    s = []
    for c in d.keys():
        s += [c] * int(d[c]) * 4
    print(s)
    log.write(">>>>>System:" + filename + "\n")
    log.write(">>>>>Composition:" + str(s) + "\n")
    log.flush()
    if len(s) > 12 or len(set(og().atomic_numbers).intersection(range(57, 104))) > 0:
        log.write(">>>>>skipping:" + str(s) + "\n")
        log.flush()
        continue
    try:
        ga = GA(species=s, population_size=20, experiment_tag=folder_tag, rmax=20, model=model_src)
        for i in range(5):
            ga.evolve(10)
    except Exception as e:
        log.write(str(e) + "\n\n")
        continue
    log.flush()
log.close()
