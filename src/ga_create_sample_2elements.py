from oganesson.genetic_algorithms import GA
from oganesson import OgStructure
import glob
import random
import ntpath
import shutil

g = glob.glob("test_ga/MP/*.cif")
g = random.sample(g, len(g))
l = []
#Limiting maximum Z to 89 to be able to use the SoftMutation operator in GA
for gg in g:
    if len(l) == 100:
        break
    filename = ntpath.basename(gg).replace(".cif", "")
    if filename.split("_")[1] not in l:
        if (
            len(set(OgStructure(file_name=gg).structure.atomic_numbers)) == 2
            # and OgStructure(file_name=gg).structure.atomic_numbers[0] <= 89
        ):
            l += [filename.split("_")[1]]
            shutil.copyfile(gg, "test_ga/sample_2elements/" + filename + ".cif")
