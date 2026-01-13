import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib
from utils import AeroLoss, get_airplane
from addict import Addict
matplotlib.use('Qt5Agg')

import yaml

with open("config.yaml", "r") as file:
    config = Addict(yaml.safe_load(file))

airplane = get_airplane(
    wing_airfoil_base=config.plane.wing_airfoil_base,
    wing_airfoil_tip=config.plane.wing_airfoil_tip,
    winglet_airfoil=config.plane.winglet_airfoil,
    attack_angle=config.plane.attack_angle,
    body_span=config.plane.body_span,
    body_len=config.plane.body_len,
    wing_base_start=config.plane.wing_base_start,
    wing_chord=config.plane.wing_chord,
    wing_lift=config.plane.wing_lift,
    leading_edge_length=config.plane.leading_edge_length,
    sweep=config.plane.sweep,
    taper_ratio=config.plane.taper_ratio,
    washout=config.plane.washout,
    dihedral=config.plane.dihedral,
    winglet_sweep=config.plane.winglet_sweep,
    winglet_toe=config.plane.winglet_toe,
    winglet_angle=config.plane.winglet_angle,
    winglet_sections=config.plane.winglet_sections,
    winglet_radius=config.plane.winglet_radius,
    winglet_taper_ratio=config.plane.winglet_taper_ratio,
    winglet_leading_edge_len=config.plane.winglet_leading_edge_len,
)

if True:
    print('AeroBuildup')
    alphas = np.linspace(-5, 12, 10)
    loss_ab = AeroLoss(airplane, alphas=alphas, method='AB', sim_on_set=True, verbose=True)
    losses_ab = loss_ab.get_inverce_losses()
    results_ab = loss_ab.sim_results
    print('VortexLatticeMethod')
    alphas = [-5, 0, 12]
    loss_vlm = AeroLoss(airplane, alphas=alphas, method='VLM', sim_on_set=True, verbose=True)
    losses_vlm = loss_vlm.get_inverce_losses()
    results_vlm = loss_vlm.sim_results

airplane.draw_three_view()
