import pyswarms as ps
from functools import partial
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
    CGx=config.plane.CGx,
    CGz=config.plane.CGz,
)

alphas = np.linspace(-5, 10, 10)
loss_ab = AeroLoss(airplane, alphas=alphas, method='AB', sim_on_set=False, verbose=True)
loss_vlm = AeroLoss(airplane, alphas=alphas, method='VLM', sim_on_set=False, verbose=True)
if False:
    print('AeroBuildup')
    losses_ab = loss_ab.get_inverce_losses()
    results_ab = loss_ab.sim_results
    print('VortexLatticeMethod')
    alphas = [-5, 0, 12]
    results_vlm = loss_vlm.sim_results
    losses_vlm = loss_vlm.get_inverce_losses()
    airplane.draw_three_view()

constraints = config.constraints
for_optimization = {key: value for key, value in config.constraints.items() if isinstance(value, list)}
not_for_optimization = {key: value for key, value in config.constraints.items() if not isinstance(value, list)}

bounds = np.array([value for value in for_optimization.values()]).T
# Initialize swarm
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
# Call instance of PSO with bounds argument
optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=len(for_optimization), options=options, bounds=bounds, ftol=1e-7, ftol_iter=3)

# Perform optimization
param_names = [_ for _ in for_optimization.keys()]
opt_func = partial(loss_ab.get_pso_loss, **not_for_optimization, param_names=[_ for _ in for_optimization.keys()])
cost, pos = optimizer.optimize(opt_func, iters=100)
final_airplane = get_airplane(**{key: value for key, value in zip(param_names, pos)}, **not_for_optimization)
final_airplane.draw_three_view()
breakpoint()
