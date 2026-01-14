import copy
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import time
from typing import Literal
import aerosandbox.numpy as np
import aerosandbox as asb
from math import sin, cos, radians
import matplotlib.pyplot as plt


def _get_inverce_losses(simfunc, keys_to_check: dict, results: dict | None = None, verbose: bool = False) -> dict:
    results = results if results is not None else simfunc()
    out = copy.deepcopy(keys_to_check)
    for key, sign in keys_to_check.items():
        if key in results:
            out[key] = np.mean(out[key] * np.array(results[key]))
            out[key] = out[key] if out[key] < 0 else out[key] * 0.001
            if verbose and not np.all(np.sign(results[key]) == np.sign(sign)):
                print(f'{key}: {sum(np.sign(results[key]) == np.sign(sign))} / {len(results[key])} are right')
        else:
            if verbose and key != 'CLCD':
                print(f'{key} is missing from results')
            out[key] = None
    out['CLCD'] = np.mean(np.array(results['CL']) / np.array(results['CD']))
    return out


class AeroLoss():
    def __init__(self, airplane, alphas: list[float] | float = 0, velocity: float = 20., method: Literal['AB', 'VLM'] = 'AB', keys_to_check: dict[str, float] | None = None, verbose: bool = False, sim_on_set: bool = False, savefig: bool = True):
        self.keys_to_check = {
            'Cmq': -0.1,
            'Cma': -0.1, # important
            'Clp': -1,
            'Clb': -1, # important
            'Clr': -1,
            'Cnr': -1,
            'Cnb': 2, # important
            'Cnp': -1,
            'CYr': 1,
            'CYb': 1,
            'CYp': -1,
            'CLCD': 0.1,
        } if keys_to_check is None else keys_to_check
        self.verbose = verbose
        self.savefig = savefig
        self.method = method
        alphas = [alphas] if isinstance(alphas, (float, int)) else alphas
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
            case _:
                raise ValueError('No such simulation method')
        self.airplane = airplane
        self.sim_on_set = sim_on_set
        self.best_plane = None
        self.best_score = np.inf

        self.losses = None
        self.sim_results = None
        self.set_airplane(airplane)

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
        out = _get_inverce_losses(self.simulator, self.keys_to_chesk, self.verbose)
        self.losses = out
        return out


    def get_pso_loss(self, params, param_names: list[str] | None = None, **kwargs):
        # TODO: parallelize the loop
        particle_losses = []
        simulators = []
        airplanes = []
        for particle in params:
            inputs = {key: value for key, value in zip(param_names, particle)}
            airplane = get_airplane(**inputs, **kwargs)
            airplanes.append(airplane)
            simulators.append(partial(self.simulate, simulators=self.set_airplane(airplane), verbose=self.verbose))

        with ProcessPoolExecutor() as executor:
            # Use list() to consume the generator and get final results
            particle_losses = list(executor.map(
                _get_inverce_losses, 
                simulators, 
                [self.keys_to_check]*len(simulators), 
                [None]*len(simulators),
                [self.verbose]*len(simulators),
            ))
        for idx, losses in enumerate(particle_losses):
            particle_losses[idx] = -sum([loss for loss in losses.values() if loss is not None])

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

        # for simulator in simulators:
        #     losses = self._get_inverce_losses(simulator, self.keys_to_check, None, self.verbose)
        #     particle_losses.append(-sum([loss for loss in losses.values() if loss is not None]))
        return particle_losses

    @staticmethod
    def simulate(simulators, verbose: bool = False):
        results = {}
        start_time = time.perf_counter()
        for simulator in simulators:
            result = simulator.run_with_stability_derivatives()
            if len(simulators) > 1:
                for key, value in result.items():
                    if key in results:
                        results[key].append(value)
                    else:
                        results[key] = [value]
            else:
                results = result
        end_time = time.perf_counter()
        if verbose:
            print(f'Iteration time: {end_time - start_time:.1f} s')
        return results

    def __call__(self):
        results = self.simulate(simulators=self.simulator, verbose=self.verbose)
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
    winglet_sweep: float = 60,  # deg
    winglet_toe: float = 0,  # deg
    winglet_angle: float = 20,  # deg
    winglet_sections: float = 2,  # ,affects calculation accuracy and speed
    winglet_radius: float = 0.05,  # m
    winglet_taper_ratio: float = 0.5,
    winglet_leading_edge_len: float = 0.05,  # m
    wing_airfoil_base: asb.Airfoil | str = asb.Airfoil('mh60'),
    wing_airfoil_tip: asb.Airfoil | str = asb.Airfoil('naca0012'),
    winglet_airfoil: asb.Airfoil | str = asb.Airfoil('mh45'),
    CGx: float = 0.18,
    CGz: float = 0.,
) -> asb.Airplane:
    # TODO: documentation for params
    CG = (CGx, 0, CGz)

    wing_airfoil_base = wing_airfoil_base if isinstance(wing_airfoil_base, asb.Airfoil) else asb.Airfoil(wing_airfoil_base)
    wing_airfoil_tip = wing_airfoil_tip if isinstance(wing_airfoil_tip, asb.Airfoil) else asb.Airfoil(wing_airfoil_tip)
    winglet_airfoil = winglet_airfoil if isinstance(winglet_airfoil, asb.Airfoil) else asb.Airfoil(winglet_airfoil)

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
                        airfoil=asb.Airfoil("naca0020"),
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

    theta = np.linspace(0, winglet_angle / 180 * np.pi, winglet_sections + 1)
    winglet_length = cos(winglet_sweep) * winglet_leading_edge_len
    winglet_curve_length = winglet_angle / 180 * np.pi * winglet_radius

    winglet_chords = wing_tip_chord * np.linspace(
          start=1,
          stop=1 - ((1 - winglet_taper_ratio) * winglet_curve_length / (winglet_curve_length + winglet_length)),
          num=len(theta)
      )

    for idx, t in enumerate(theta[1:]):
        winglet_xyz_le = [
                wing_end_coords[0] + winglet_radius * np.sin(t) * np.tan(winglet_sweep),
                wing_end_coords[1] + winglet_radius * np.sin(t),
                wing_end_coords[2] + winglet_radius * (1 - np.cos(t))
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

    airplane = asb.Airplane(
        name="The Wing",
        xyz_ref=CG,  # TODO: CG location
        wings=[
            asb.Wing(
                symmetric=True,
                xsecs=xsecs,
            ),
        ],
    )
    return airplane
