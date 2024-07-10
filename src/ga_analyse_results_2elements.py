from oganesson.genetic_algorithms import GA
from oganesson import OgStructure
import glob
import random
import ntpath
import shutil
from pymatgen.analysis.structure_matcher import StructureMatcher

model = "diep"
model = "mym3gnet"
sm = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)

sample_elemental = glob.glob("test_ga/sample_2elements/*.cif")
sample_elemental = [ntpath.basename(x).replace(".cif", "") for x in sample_elemental]
sample_elemental_structures = {}
for a in sample_elemental:
    sample_elemental_structures[a] = OgStructure(file_name="test_ga/sample_2elements/" + a + ".cif")

f = open("test_ga/results_2elements_" + model + ".csv",'w')
f.write("material,ga,rms\n")
f_summary = open("test_ga/results_2elements_summary_" + model + ".csv",'w')
f_summary.write("material,ga,rms\n")
for k in sample_elemental_structures.keys():
    ga_files = glob.glob("og_lab/ga_" + model + "_2elements_" + k + "/relaxed/*.cif")
    if ga_files:
        ga_file_min_rms =  ntpath.basename(ga_files[0]).replace(".cif", "")
        d_min_rms = sm.get_rms_dist(OgStructure(file_name=ga_files[0])(), sample_elemental_structures[k]())
        if d_min_rms is not None:
            d_min_rms = d_min_rms[0]
        for ga_file in ga_files:
            ga_structure = OgStructure(file_name=ga_file)
            ga_file = ntpath.basename(ga_file).replace(".cif", "")
            d = sm.get_rms_dist(ga_structure(), sample_elemental_structures[k]())
            if d is not None:
                d = d[0]
            if d is not None and d_min_rms is not None:
                if d_min_rms < d:
                    d_min_rms = d
                    ga_file_min_rms = ga_file
            elif d is not None and d_min_rms is None:
                d_min_rms = d
                ga_file_min_rms = ga_file
            elif d is None and d_min_rms is not None:
                pass

            print(k, ga_file, d)
            f.write(k + "," + ga_file + "," + str(d) + "\n")
        f_summary.write(k + "," + ga_file_min_rms + "," + str(d_min_rms) + "\n")

f.close()
f_summary.close()