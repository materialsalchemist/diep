from pymatgen.core import Structure
import json

MP = json.load(open('../../GetData_API/MP.json'))
print(MP[0])
compositions = []
for m in MP:
    if m['nsites'] <=4 and m['energy_above_hull'] == 0:
        s = Structure.from_dict(m['structure'])
        id = m['material_id']
        f = m['formula_pretty']
        s.to('test_ga/MP/'+id+'_'+f+'.cif')





