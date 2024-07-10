from  pymatgen.core import Structure,PeriodicSite
from ase.build import nanotube
from oganesson import OgStructure
import numpy as np
from ase.md.md import Trajectory

s = OgStructure(file_name='mp-341.cif')
print(list(s()[0].species.as_dict().keys())[0])
print(s()[0].species_string)
print(hasattr(s()[0],'species_string'))

t = Trajectory('src/og_lab/simulation_diep_VASP_mp-1078834_2x2x1/1000.traj')
for i in t:
    a = t[0].positions
    print(i)

# s.relax()
# s.simulate(loginterval=10,model="M3GNet-MP-2021.2.8-PES")


# atoms = nanotube(10, 10, length=20, bond=1.42, symbol='C', vacuum=20)
# atoms.set_pbc = True
# og = OgStructure(atoms)

# og.scale([1,1,1.1])
# positions = atoms.positions
# freeze = np.where(positions[:,2]<10)[0].tolist()
# og().to('text_initial.cif')
# og.relax(relax_cell=False,fix_atoms_indices=freeze,model="m3gnet")
# og().to('text_relaxed.cif')

# og().to('text_initial.cif')
# og.fracture(1.1,write_intermediate=True,method='opt_pulling')
# og().to('text_relaxed.cif')