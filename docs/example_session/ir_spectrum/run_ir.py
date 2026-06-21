#!/bin/env python

import mlatom as ml 
import os

# 1. load molecule
molecule_path = '/home/dral/hexanol_IR_spectrum_AIQM2/hexanol_AIQM2_IR/optmol.json'

mol = ml.molecule.load(molecule_path, format='json')


# 2. define method
MODEL = ml.methods(method='AIQM2', program=None)

# 3. fill in settings of ir
working_directory = '/home/dral/hexanol_IR_spectrum_AIQM2/hexanol_AIQM2_IR'
MODEL.working_directory = working_directory
freq = ml.freq(
    molecule = mol,
    model = MODEL,
    program = 'pyscf',
    program_kwargs = {},
    #normal_mode_normalization = None,
    working_directory = working_directory, 
    anharmonic = None,
    ir = True,
    raman = False,
)

scaling_factor = None
if not scaling_factor is None:
    mol.frequencies = mol.frequencies * scaling_factor
spectrum = ml.spectra.ir.lorentzian(molecule=mol,fwhm=30)
spectrum.plot(os.path.join(working_directory,'ir.png'))
mol.dump(os.path.join(working_directory,'irmol.json'),format='json')
