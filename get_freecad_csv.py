import csv
from pathlib import Path
import aerosandbox.numpy as np
from utils import AeroLoss, get_airplane, convert_numpy, OptFuncSwither, fix_thickness
from addict import Addict
import aerosandbox as asb

# matplotlib.use('Qt5Agg')

import yaml


if __name__ == '__main__':
    outfile = 'plane.csv'

    with open("config.yaml", "r") as file:
        config = Addict(yaml.safe_load(file))

    plane = config.plane
    cannard_airfoil, plane['scale_y_cannard'] = fix_thickness(plane.cannard_airfoil, plane.cannard_thickness, plane.cannard_chord, soft=False, return_scale=True)

    if plane.body_len == 'adaptive':
        plane.body_len = (0.025 + plane.wing_base_start * 0.3 + plane.wing_chord)
    body_airfoil, plane['scale_y_body'] = fix_thickness(plane.body_airfoil, plane.body_height, plane.body_len, return_scale=True)
    wing_airfoil_base, plane['scale_y_wing_base'] = fix_thickness(plane.wing_airfoil_base, plane.wing_min_thickness, plane.wing_chord, soft=True, return_scale=True)
    wing_airfoil_tip, plane['scale_y_wing_tip'] = fix_thickness(plane.wing_airfoil_tip, plane.wing_min_thickness, plane.wing_chord, soft=True, return_scale=True)
    foot_airfoil, plane['scale_y_foot'] = fix_thickness(plane.foot_airfoil, plane.foot_thickness, plane.foot_chord, return_scale=True)


    with open(outfile, 'w', newline='') as file:
        writer = csv.DictWriter(file, plane.keys(), delimiter='\t')
        writer.writeheader()
        writer.writerow(plane)
    print(f'written plane configuration to {outfile}')
