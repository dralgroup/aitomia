#!/bin/env python

import mlatom as ml 
import os 

# 1. load molecule
molecule_path = '/home/dral/hexanol_IR_spectrum_AIQM2/hexanol_AIQM2_optimization/hexanol.json'

mol = ml.molecule.load(molecule_path, format='json')


# 2. define method
MODEL = ml.methods(method='AIQM2', program=None)

# 3. Geometry optimization
working_directory = '/home/dral/hexanol_IR_spectrum_AIQM2/hexanol_AIQM2_optimization'
MODEL.working_directory = working_directory
geomopt = ml.optimize_geometry(
    model=MODEL,
    initial_molecule=mol,
    program='geometric',
    working_directory=working_directory
)

# 4. Save results
optmol = geomopt.optimized_molecule
optmol.dump('/home/dral/hexanol_IR_spectrum_AIQM2/hexanol_AIQM2_optimization/optmol.json',format='json')
traj = geomopt.optimization_trajectory
traj.dump(os.path.join(working_directory,'opttraj.json'),format='json')
traj = traj.to_database()
traj.write_file_with_xyz_coordinates(os.path.join(working_directory,'opttraj.xyz'))