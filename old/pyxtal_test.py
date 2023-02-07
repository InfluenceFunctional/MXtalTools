import numpy as np
from pyxtal import pyxtal


cry = pyxtal(molecular=True)
cry.from_random(dim=3,group=2,species=['aspirin'])
cry._get_coords_and_species()