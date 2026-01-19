import pyswarms as ps
from pathlib import Path
import copy
from functools import partial
import aerosandbox.numpy as np
import matplotlib
from utils import AeroLoss, get_airplane, convert_numpy, OptFuncSwither
from addict import Addict
from datetime import datetime

matplotlib.use('Qt5Agg')

import yaml


if __name__ == '__main__':

    with open("config.yaml", "r") as file:
        config = Addict(yaml.safe_load(file))
    Path(config.data.output_path).mkdir(parents=True, exist_ok=True)

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
        cannard=config.plane.cannard,
        cannard_attack_angle = config.plane.cannard_attack_angle,
        cannard_airfoil = config.plane.cannard_airfoil,
        cannard_start = config.plane.cannard_start,
        cannard_chord = config.plane.cannard_chord,
        cannard_len = config.plane.cannard_len,
        cannard_z_offset = config.plane.cannard_z_offset,
        cannard_thickness = config.plane.cannard_thickness,
    )
    # airplane.draw_three_view()
    # exit()

    alphas = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 7, 10]).astype(np.float32)
    # alphas = np.linspace(-5, 10, 15).astype(np.float32)
    loss_ab = AeroLoss(airplane, alphas=alphas, method='AB', sim_on_set=False, verbose=True, airfoils=config.airfoils, targets=config.targets, target_range=config.target_range, velocity=config.velocity)
    loss_vlm = AeroLoss(airplane, alphas=alphas, method='VLM', sim_on_set=False, verbose=True, airfoils=config.airfoils, targets=config.targets, target_range=config.target_range, velocity=config.velocity)
    loss_ab.get_inverce_losses()

    if True:

        constraints = config.constraints
        for_optimization = {key: value for key, value in config.constraints.items() if isinstance(value, list)}

        for_optimization['wing_airfoil_base'] = [0, len(config.airfoils)]
        for_optimization['wing_airfoil_tip'] = [0, len(config.airfoils)]
        for_optimization['winglet_airfoil'] = [0, len(config.airfoils)]
        for_optimization['cannard_airfoil'] = [0, len(config.airfoils)]

        not_for_optimization = {key: value for key, value in config.constraints.items() if not isinstance(value, list)}

        bounds = np.array([value for value in for_optimization.values()]).T
        # Initialize swarm
        options = {'c1': 1.494, 'c2': 0.5, 'w': 0.9, 'k': 3, 'p': 2}
        # options = {'c1': 1.5, 'c2': 0.5, 'w': 0.95}

        # Perform optimization
        method = config.optimization.method
        param_names = [_ for _ in for_optimization.keys()]

        _func = loss_vlm.get_pso_loss if method == 'vlm' else loss_ab.get_pso_loss
        opt_func = partial(_func, **not_for_optimization, param_names=[_ for _ in for_optimization.keys()])
        opt_func_ab = partial(loss_ab.get_pso_loss, **not_for_optimization, param_names=[_ for _ in for_optimization.keys()])
        opt_func_vlm = partial(loss_vlm.get_pso_loss, **not_for_optimization, param_names=[_ for _ in for_optimization.keys()])

        if method == 'both':
            optimizer_ab = ps.single.GlobalBestPSO(n_particles=config.optimization.particles_ab, dimensions=len(for_optimization), options=options, bounds=bounds, ftol=1e-7, ftol_iter=4)
            cost, pos = optimizer_ab.optimize(opt_func_ab, iters=100)

            init_pos = 0.2 * (bounds[1] - bounds[0]) * np.random.randn(config.optimization.particles_vlm - 1, len(pos)) + pos
            init_pos = np.vstack([pos, init_pos])
            init_pos = np.clip(init_pos, min=bounds[0], max=bounds[1])
            optimizer_vlm = ps.single.GlobalBestPSO(n_particles=config.optimization.particles_vlm, dimensions=len(for_optimization), options=options, bounds=bounds, ftol=1e-7, ftol_iter=3, init_pos=init_pos)
            cost, pos = optimizer_vlm.optimize(opt_func_vlm, iters=100)
        else:
            optimizer = ps.single.GlobalBestPSO(n_particles=config.optimization.particles_vlm if config.optimization.method == 'vlm' else config.optimization.particles_ab, dimensions=len(for_optimization), options=options, bounds=bounds, ftol=1e-7, ftol_iter=4)
            cost, pos = optimizer_vlm.optimize(opt_func, iters=100)

        final_airplane_params = {key: value for key, value in zip(param_names, pos)} | not_for_optimization
        final_airplane_params = {key: value if 'airfoil' not in key else config.airfoils[max(0, min(len(config.airfoils), int(value)))] for key, value in final_airplane_params.items()}

        final_airplane = get_airplane(**final_airplane_params)
        final_config = copy.deepcopy(config)
        for key, value in final_airplane_params.items():
            final_config.plane[key] = value
    else:
        final_config = copy.deepcopy(config)
        final_airplane = airplane
        cost = 0.
        method = 'none'


    if method == 'vlm' or method == 'both' or method == 'none':
        loss_vlm.set_airplane(final_airplane)
        final_results = loss_vlm()
    else:
        loss_ab.set_airplane(final_airplane)
        final_results = loss_ab()

    final_results['alphas'] = alphas
    final_config = final_results | final_config

    with open(Path(config.data.output_path) / f'{method}_{-cost:.2f}_{config.velocity}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.yaml', 'w') as file:
        yaml.safe_dump(convert_numpy(final_config), file, default_flow_style=False)

    # final_airplane.draw_three_view()
    # breakpoint()
