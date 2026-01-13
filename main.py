import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib
from utils import AeroLoss, get_airplane
matplotlib.use('Qt5Agg')

wing_airfoil_base = asb.Airfoil('mh60')
wing_airfoil_tip = asb.Airfoil('naca0012')
winglet_airfoil = asb.Airfoil('mh45')
attack_angle = 2  # deg
body_span = 0.1  # m
body_len = 0.3  # m
wing_base_start = 0.33  # % from body_len
wing_chord = 0.18  # m
wing_lift = 0  # m, from body midpoint
leading_edge_length = 0.4  # m
sweep = 20  # deg
taper_ratio = 0.5
washout = 1  # deg
dihedral = 0  # deg, keep at 0 for easier construction

winglet_sweep = 60  # deg
winglet_toe = 0  # deg
winglet_angle = 20  # deg
winglet_sections = 2  # ,affects calculation accuracy and speed
winglet_radius = 0.05  # m
winglet_taper_ratio = 0.5
winglet_leading_edge_len = 0.05  # m

airplane = get_airplane(
    wing_airfoil_base=wing_airfoil_base,
    wing_airfoil_tip=wing_airfoil_tip,
    winglet_airfoil=winglet_airfoil,
    attack_angle=attack_angle,
    body_span=body_span,
    body_len=body_len,
    wing_base_start=wing_base_start,
    wing_chord=wing_chord,
    wing_lift=wing_lift,
    leading_edge_length=leading_edge_length,
    sweep=sweep,
    taper_ratio=taper_ratio,
    washout=washout,
    dihedral=dihedral,
    winglet_sweep=winglet_sweep,
    winglet_toe=winglet_toe,
    winglet_angle=winglet_angle,
    winglet_sections=winglet_sections,
    winglet_radius=winglet_radius,
    winglet_taper_ratio=winglet_taper_ratio,
    winglet_leading_edge_len=winglet_leading_edge_len,
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
