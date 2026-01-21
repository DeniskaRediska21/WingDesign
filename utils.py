import copy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import time
from typing import Literal, Callable
import aerosandbox.numpy as np
import aerosandbox as asb
from math import sin, cos, radians
import matplotlib.pyplot as plt

def prepare_files(airplane: asb.Airplane, dir: Path):
    output_path = Path(dir) / 'airplane.vspscript'
    airplane.export_OpenVSP_vspscript(str(output_path))
    airplane.export_cadquery_geometry(str(output_path.with_suffix('.step')))
    return output_path, output_path.with_suffix('.step')

def fix_thickness(airfoiol, thickness, chord, soft: bool = False):
    current_thickness = airfoiol.max_thickness() * chord

    if not soft or current_thickness < thickness:
        airfoiol = airfoiol.scale(scale_y=thickness/current_thickness)
    return airfoiol

def run_sim(simulator):
    return simulator.run_with_stability_derivatives()

def get_lowest_z(wings: list[asb.Wing]) -> float:
    lowest_z = np.inf
    for wing in wings:
        for xsec in wing.xsecs:
            lowest_z = lowest_z if xsec.xyz_le[-1] > lowest_z else xsec.xyz_le[-1]
    return lowest_z

def _get_inverce_losses(simfunc, keys_to_check: dict, alphas: list[float], targets: dict[str, float], target_range: list[float], results: dict | None = None, verbose: bool = False) -> dict:
    results = results if results is not None else simfunc()
    out = copy.deepcopy(keys_to_check)
    alphas = np.array(alphas)
    in_ranges = np.logical_and(alphas>target_range[0], alphas<target_range[1])
    results['CLCD'] = np.array(results['CL']) / np.array(results['CD'])
    for key, sign in keys_to_check.items():
        if key in results:
            result = []
            for in_range, res in zip(in_ranges, results[key]):
                if not in_range or key not in targets:
                    res = out[key] * np.array(res)
                    res = 2 * res if res < 0 else res * 0
                    result.append(res)
                else:
                    res = - np.abs(out[key]) * np.abs(res - targets[key])
                    result.append(res)

            out[key] = np.mean(result)
            # if verbose and not np.all(np.sign(results[key]) == np.sign(sign)):
                # print(f'{key}: {sum(np.sign(results[key]) == np.sign(sign))} / {len(results[key])} are right')
        else:
            if verbose:
                print(f'{key} is missing from results')
                out[key] = None
    return out


class AeroLoss():
    def __init__(self, airplane, alphas: list[float] | float = 0, velocity: float = 20., method: Literal['AB', 'VLM'] = 'AB', keys_to_check: dict[str, float] | None = None, verbose: bool = False, sim_on_set: bool = False, savefig: bool = True, airfoils: list[str] = ['mh60', 'naca0008'], targets: dict[str, float] = dict(), target_range: tuple[int, int] | None = None):
        self.keys_to_check = {
            'Cmq': -0.1,
            'Cma': -2, # important
            'Clp': -1,
            'Clb': -2, # important
            'Clr': -2,
            'Cnr': -2,
            'Cla': 0.5,
            'Cnb': 2, # important
            'Cnp': -1,
            'CYr': 1,
            'CYb': -1,
            'CYp': -1,
            'CLCD': 0.5,
        } if keys_to_check is None else keys_to_check
        self.targets = targets
        self.target_range = target_range
        self.verbose = verbose
        self.savefig = savefig
        self.airfoils = airfoils
        self.method = method
        alphas = [alphas] if isinstance(alphas, (float, int)) else alphas
        self.alphas = alphas
        match method:
            case 'AB':
                self.op_point=[asb.OperatingPoint(
                    velocity=velocity,  # m/s
                    alpha=alphas,  # degree
                )]
                self.sim_func = asb.AeroBuildup
            case 'VLM':
                self.op_point=[asb.OperatingPoint(
                    velocity=velocity,  # m/s
                    alpha=alpha,  # degree
                ) for alpha in alphas]
                self.sim_func = asb.VortexLatticeMethod
            case 'LL':
                self.op_point=[asb.OperatingPoint(
                    velocity=velocity,  # m/s
                    alpha=alpha,  # degree
                ) for alpha in alphas]
                self.sim_func = asb.LiftingLine
            case _:
                raise ValueError('No such simulation method')
        self.airplane = airplane
        self.sim_on_set = sim_on_set
        self.best_plane = None
        self.best_score = np.inf

        self.losses = None
        self.sim_results = None
        self.simulator = self.set_airplane(airplane)

    def set_airplane(self, airplane):
        self.simulator = [self.sim_func(
            airplane=airplane,
            op_point=op_point
        ) for op_point in self.op_point]
        self.losses = None
        self.sim_results = None
        if self.sim_on_set:
            self()
        return self.simulator

    def get_inverce_losses(self):
        simulator = partial(self.simulate, simulators=self.simulator, verbose=self.verbose)
        out = _get_inverce_losses(simulator, self.keys_to_check, self.alphas, self.targets, self.target_range, None, self.verbose)
        self.losses = out
        return out


    def get_pso_loss(self, params, param_names: list[str] | None = None, **kwargs):
        # TODO: parallelize the loop
        particle_losses = []
        simulators = []
        airplanes = []
        for particle in params:
            inputs = {key: value for key, value in zip(param_names, particle)}
            inputs = {key: value if 'airfoil' not in key else self.airfoils[max(0, min(len(self.airfoils) - 1, int(value)))] for key, value in inputs.items()}
            airplane = get_airplane(**inputs, **kwargs)
            airplanes.append(airplane)
            simulators.append(partial(self.simulate, simulators=self.set_airplane(airplane), verbose=self.verbose))

        with ProcessPoolExecutor() as executor:
            # Use list() to consume the generator and get final results
            particle_losses = list(executor.map(
                _get_inverce_losses, 
                simulators, 
                [self.keys_to_check]*len(simulators), 
                [self.alphas] * len(simulators),
                [self.targets] * len(simulators),
                [self.target_range] * len(simulators),
                [None]*len(simulators),
                [self.verbose]*len(simulators),
            ))
        current_best = np.inf
        current_losses = None
        for idx, losses in enumerate(particle_losses):
            loss = -sum([loss for loss in losses.values() if loss is not None])
            if loss < current_best:
                current_losses = copy.deepcopy(losses)
                current_best = loss
            particle_losses[idx] = loss
        print(current_losses)

        I = np.argmin(particle_losses)
        if particle_losses[I] < self.best_score:
            self.best_score = particle_losses[I]
            self.best_plane = airplanes[I]

        if self.savefig:
            axs = self.best_plane.draw_three_view(show=False)
            fig = axs[0, 0].get_figure()
            # Save the entire grid of subplots
            fig.savefig('best_plane.png', bbox_inches='tight', dpi=100)
            plt.close(fig)

        return particle_losses

    @staticmethod
    def simulate(simulators, verbose: bool = False, parallel: bool = False):
        results = {}
        start_time = time.perf_counter()

        if len(simulators) > 1:
            if parallel:
                with ProcessPoolExecutor() as executor:
                    # Use map to run the function for all simulators in parallel
                    all_results = list(executor.map(run_sim, simulators))
                for result in all_results:
                    for key, value in result.items():
                        if key in results:
                            results[key].append(value)
                        else:
                            results[key] = [value]
            else:
                for simulator in simulators:
                    result = simulator.run_with_stability_derivatives()
                    for key, value in result.items():
                        if key in results:
                            results[key].append(value)
                        else:
                            results[key] = [value]
        else:
            return simulators[0].run_with_stability_derivatives()

        end_time = time.perf_counter()
        # if verbose:
            # print(f'Iteration time: {end_time - start_time:.1f} s')
        return results

    def __call__(self, parallel: bool = False):
        results = self.simulate(simulators=self.simulator, verbose=self.verbose, parallel=parallel)
        self.sim_results = results
        return results
    
def get_airplane(
    attack_angle: float = 2,  # deg
    body_span: float = 0.1,  # m
    body_len: float = 0.3,  # m
    wing_base_start: float = 0.33,  # % from body_len
    wing_chord: float = 0.18,  # m
    wing_lift: float = 0,  # m, from body midpoint
    leading_edge_length: float = 0.4,  # m
    sweep: float = 20,  # deg
    taper_ratio: float = 0.5,
    washout: float = 1,  # deg
    dihedral: float = 0,  # deg, keep at for easier construction
    winglets: bool = True,
    winglet_sweep: float = 60,  # deg
    winglet_toe: float = 0,  # deg
    winglet_angle: float = 20,  # deg
    winglet_sections: float = 1,  # ,affects calculation accuracy and speed
    winglet_radius: float = 0.05,  # m
    winglet_taper_ratio: float = 0.5,
    winglet_leading_edge_len: float = 0.05,  # m
    wing_airfoil_base: asb.Airfoil | str = asb.Airfoil('mh60'),
    wing_airfoil_tip: asb.Airfoil | str = asb.Airfoil('naca0012'),
    winglet_airfoil: asb.Airfoil | str = asb.Airfoil('mh45'),
    CGx: float = 0.18,
    CGz: float = 0.,
    cannard: bool = False,
    cannard_attack_angle: float = 0.,  # % of body len
    cannard_start: float = 0.1,  # % of body len
    cannard_airfoil: asb.Airfoil | str | None = asb.Airfoil('naca0012'),
    cannard_chord: float | None = None,
    cannard_len: float | None = None,
    cannard_z_offset: float = 0.,
    cannard_thickness: float = 0.02,
    wing_min_thickness: float | None = None,
    body_height: float = 0.3 * 0.2,
    foot: bool = False,
    foot_base_start: float = 0.,
    foot_taper: float = 0.7,
    foot_airfoil: asb.Airfoil | str = asb.Airfoil('naca0012'),
    foot_chord: float = 0.05,
    foot_thickness: float = 0.01,
) -> asb.Airplane:

    if cannard and (cannard_airfoil is None or cannard_chord is None or cannard_len is None):
        raise ValueError('if cannard is set to True you should provide cannard_airfoil, cannard_chord and cannard_len')

    if body_len == 'adaptive':
        body_len = (0.025 + wing_base_start * 0.3 + wing_chord)

    if cannard:
        cannard_airfoil = cannard_airfoil if isinstance(cannard_airfoil, asb.Airfoil) else asb.Airfoil(cannard_airfoil)
        cannard_airfoil = fix_thickness(cannard_airfoil, cannard_thickness, cannard_chord, soft=False)
        cannard = asb.Wing(
            symmetric=True,
            xsecs=[
                    asb.WingXSec(
                        name='cannard_start',
                        xyz_le=[
                            cannard_start * body_len,
                            0,
                            cannard_z_offset,
                        ],
                        chord=cannard_chord,
                        twist=attack_angle + cannard_attack_angle,
                        airfoil=cannard_airfoil,
                    ),
                    asb.WingXSec(
                        name='wing_base',
                        xyz_le=[
                            cannard_start * body_len,
                            cannard_len,
                            cannard_z_offset,
                        ],
                        chord=cannard_chord,
                        twist=cannard_attack_angle,
                        airfoil=cannard_airfoil,
                    ),
            ],
        )
        

    wing_airfoil_base = wing_airfoil_base if isinstance(wing_airfoil_base, asb.Airfoil) else asb.Airfoil(wing_airfoil_base)
    wing_airfoil_tip = wing_airfoil_tip if isinstance(wing_airfoil_tip, asb.Airfoil) else asb.Airfoil(wing_airfoil_tip)
    winglet_airfoil = winglet_airfoil if isinstance(winglet_airfoil, asb.Airfoil) else asb.Airfoil(winglet_airfoil)
    body_airfoil = fix_thickness(asb.Airfoil("naca0020"), body_height, body_len)

    if wing_min_thickness is not None:
        wing_airfoil_base = fix_thickness(wing_airfoil_base, wing_min_thickness, wing_chord, soft=True)
        wing_airfoil_tip = fix_thickness(wing_airfoil_tip, wing_min_thickness, wing_chord, soft=True)

    wing_tip_chord = wing_chord * taper_ratio
    wing_end_coords = [
        wing_base_start * body_len + sin(radians(sweep)) * leading_edge_length,
        body_span + cos(radians(sweep)) * cos(radians(dihedral)) * leading_edge_length,
        sin(radians(dihedral)) * leading_edge_length,
    ]
    winglet_sweep = radians(winglet_sweep)

    xsecs = [
                    asb.WingXSec(
                        name='body',
                        xyz_le=[
                            0,
                            0,
                            0,
                        ],
                        chord=body_len,
                        twist=0,
                        airfoil=body_airfoil,
                    ),
                    asb.WingXSec(
                        name='wing_base',
                        xyz_le=[
                            wing_base_start * body_len,
                            body_span,
                            wing_lift,
                        ],
                            chord=wing_chord,
                        twist=attack_angle,
                        airfoil=wing_airfoil_base,
                    ),
                    asb.WingXSec(
                        name='wing_tip',
                        xyz_le=wing_end_coords,
                        chord=wing_tip_chord,
                        twist=attack_angle + washout,
                        airfoil=wing_airfoil_tip,
                    ),
                ]

    theta = np.linspace(0, winglet_angle / 180 * np.pi, int(winglet_sections + 1))
    winglet_length = cos(winglet_sweep) * winglet_leading_edge_len
    winglet_curve_length = np.abs(winglet_angle) / 180 * np.pi * winglet_radius

    winglet_chords = wing_tip_chord * np.linspace(
          start=1,
          stop=1 - ((1 - winglet_taper_ratio) * winglet_curve_length / (winglet_curve_length + winglet_length)),
          num=len(theta)
      )

    if winglets:
        for idx, t in enumerate(theta[1:]):
            winglet_xyz_le = [
                    wing_end_coords[0] + np.sign(t) * winglet_radius * np.sin(t) * np.tan(winglet_sweep),
                    wing_end_coords[1] + np.sign(t) * winglet_radius * np.sin(t),
                    wing_end_coords[2] + np.sign(t) * winglet_radius * (1 - np.cos(t))
                ]
            xsecs.append(asb.WingXSec(
                name=f'winglet_transition_{idx}',
                xyz_le=winglet_xyz_le,
                chord=winglet_chords[idx + 1],
                twist=attack_angle + washout,  # TODO: mb needs to be calculated from winglet_toe for gradual angle change
                airfoil=wing_airfoil_tip if idx+1 != len(theta) - 1 else winglet_airfoil,  # use wingtip airfoil till the last section, then transition to winglet airfoil
            ))

        xsecs.append(
            asb.WingXSec(
                name='winglet',
                xyz_le=[
                    winglet_xyz_le[0] + sin(winglet_sweep) * winglet_leading_edge_len,
                    winglet_xyz_le[1] + cos(radians(winglet_angle)) * cos(winglet_sweep) * winglet_leading_edge_len,
                    winglet_xyz_le[2] + sin(radians(winglet_angle)) * cos(winglet_sweep) * winglet_leading_edge_len,
                ],
                chord=wing_tip_chord * winglet_taper_ratio,
                twist=winglet_toe,  # mb needs implementation in the curved section too, now is a form of washout to winglet tips
                airfoil=winglet_airfoil,
            )
        )

    wings=[
        asb.Wing(
            symmetric=True,
            xsecs=xsecs,
        ),
    ]

    if cannard:
        wings.append(cannard)

    if foot:
        lowest_z = get_lowest_z(wings)
        foot_airfoil = fix_thickness(foot_airfoil, foot_thickness, foot_chord)
        foot = asb.Wing(
            symmetric=True,
            xsecs=[
                    asb.WingXSec(
                        name='cannard_start',
                        xyz_le=[
                            foot_base_start,
                            0,
                            0,
                        ],
                        chord=foot_chord,
                        twist=0,
                        airfoil=foot_airfoil,
                    ),
                    asb.WingXSec(
                        name='wing_base',
                        xyz_le=[
                            foot_base_start + (foot_chord - foot_chord * foot_taper) if foot_taper > 0 else 0.,
                            0,
                            lowest_z,
                        ],
                        chord=foot_chord * np.abs(foot_taper),
                        twist=0,
                        airfoil=foot_airfoil,
                    ),
            ],
        )
        wings.append(foot)

    CG = (CGx * max(body_len, (wing_end_coords[0] + wing_tip_chord)), 0, CGz)
    airplane = asb.Airplane(
        name="The Wing",
        xyz_ref=CG,  # TODO: CG location
        wings=wings,
    )
    return airplane


def convert_numpy(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy(v) for v in obj]
    if isinstance(obj, (float, int, str)):
        return obj


class OptFuncSwither:
    def __init__(self, opt_funcs: list[Callable], switch_after: int | list[int] = 1, fake_improvement_coef = 0.9):
        self.switch_afters = switch_after if isinstance(switch_after, list) else [switch_after]
        self.opt_funcs = opt_funcs if isinstance(opt_funcs, list) else [opt_funcs]

        self.fake_improvement_coef = fake_improvement_coef
        self.idx = 0
        self.results = None
        self.set_opt_func()
        self.min = None
        self.max = None

    def set_opt_func(self):
        self.min = np.min(self.results) if self.results is not None else None
        self.max = None
        self.called = 0
        self.opt_func = self.opt_funcs[min(len(self.opt_funcs), self.idx)]
        if self.idx < len(self.switch_afters):
            self.switch_after = self.switch_afters[self.idx]
        else:
            self.switch_after = None

    def __call__(self, *args, **kwargs):
        self.results = self.opt_func(*args, **kwargs)
        if self.min is not None:
            if self.max is None:
                self.max = np.min(self.results)
            self.results = self.results / self.max * self.min * self.fake_improvement_coef

        self.called += 1
        if self.switch_after is not None and self.called == self.switch_after: 
            self.idx += 1
            self.set_opt_func()
        return self.results
        
