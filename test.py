from math import sin, cos, radians, degrees
import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
matplotlib.use('Qt5Agg')
wing_airfoil_base = asb.Airfoil('mh60')
wing_airfoil_tip = asb.Airfoil('naca0012')
winglet_airfoil = asb.Airfoil('mh45')

atack_angle = 2  # deg
body_span = 0.1  # m
body_len = 0.3  # m
wing_base_start = 0.33  # % from body_len
wing_chord = 0.18  # m
wing_lift = 0  # m, from body midpoint
leading_edge_length = 0.4  # m
sweep = 20  # deg
taper_ratio = 0.5
washout = 1  # deg
dihedral = 0  # deg, keep at for easier construction

winglet_sweep = 60  # deg
winglet_toe = 0  # deg
winglet_angle = 20  # deg
winglet_sections = 4  # ,affects calculation accuracy and speed
winglet_radius = 0.05  # m
winglet_taper_ratio = 0.5
winglet_leading_edge_len = 0.05

wing_tip_chord = wing_chord * taper_ratio
wing_end_coords = [
    wing_base_start * body_len + sin(radians(sweep)) * leading_edge_length,
    body_span + cos(radians(sweep)) * leading_edge_length,
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
                    airfoil=asb.Airfoil("naca0030"),
                ),
                asb.WingXSec(
                    name='wing_base',
                    xyz_le=[
                        wing_base_start * body_len,
                        body_span,
                        wing_lift,
                    ],
                        chord=wing_chord,
                    twist=atack_angle,
                    airfoil=wing_airfoil_base,
                ),
                asb.WingXSec(
                    name='wing_tip',
                    xyz_le=wing_end_coords,
                    chord=wing_tip_chord,
                    twist=washout,
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
            winglet_radius * (1 - np.cos(t))
        ]
    xsecs.append(asb.WingXSec(
        name=f'winglet_transition_{idx}',
        xyz_le=winglet_xyz_le,
        chord=winglet_chords[idx + 1],
        twist=0,  # TODO: mb needs to be calculated from winglet_toe for gradual angle change
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
    xyz_ref=[0, 0, 0],  # TODO: CG location
    wings=[
        asb.Wing(
            symmetric=True,
            xsecs=xsecs,
        ),
    ],
)

if False:
    alpha = np.linspace(-5, 15, 20)
    vlm = asb.AeroBuildup(
        airplane=airplane,
        op_point=asb.OperatingPoint(
            velocity=25,  # m/s
            alpha=alpha,  # degree
        ),
    )

    start_time = time.perf_counter()
    results = vlm.run_with_stability_derivatives()
    end_time = time.perf_counter()
    print(end_time-start_time)

    alpha = 5
    vlm = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(
            velocity=25,  # m/s
            alpha=alpha,  # degree
        ),
    )
    start_time = time.perf_counter()
    results = vlm.run_with_stability_derivatives()
    end_time = time.perf_counter()
    print(end_time-start_time)

airplane.draw_three_view()
