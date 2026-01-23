import pyswarms as ps
from pathlib import Path
import copy
from functools import partial
import aerosandbox.numpy as np
import matplotlib
from utils import AeroLoss, get_airplane, convert_numpy, OptFuncSwither, prepare_files, prepare_config
from addict import Addict
from datetime import datetime
import streamlit as st
import matplotlib.pyplot as plt
import yaml
from tempfile import TemporaryDirectory

st.set_page_config(layout="wide")

with open("config.yaml", "r") as file:
    config = Addict(yaml.safe_load(file))
Path(config.data.output_path).mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':

    if 'current_params' not in st.session_state:
        st.session_state['current_params'] = {}
    if 'airplane' not in st.session_state:
        st.session_state['airplane'] = None
    if 'page' not in st.session_state:
        st.session_state.page = 'Editor'
    if 'tempdir' not in st.session_state:
        st.session_state.tempdir = TemporaryDirectory()
    if 'stepfile' not in st.session_state:
        st.session_state.stepfile = None
    if 'config_file' not in st.session_state:
        st.session_state.config_file = None


    def go_to_results(): st.session_state.page = "Results"
    def go_to_editor(): st.session_state.page = "Editor"


    if st.session_state.page == "Editor":
        with st.sidebar:
            for key, bounds in config.constraints.items():
                if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                    if "cannard" in key and "cannard" in st.session_state.current_params and not st.session_state.current_params['cannard']:
                        continue
                    if "foot" in key and "foot" in st.session_state.current_params and not st.session_state.current_params['foot']:
                        continue
                    if ("winglets" in key or "winglet" in key) and "winglets" in st.session_state.current_params and not st.session_state.current_params['winglets']:
                        continue
                    # Create a slider if there are two values (min, max)
                    min_val, max_val = bounds
                    if key not in st.session_state.current_params:
                        default_val = config.plane.get(key, min_val) 
                        default_val = float(default_val) if not isinstance(default_val, str) else (0.025 + config.plane.wing_base_start * 0.3 + config.plane.wing_chord)
                    else:
                        default_val = st.session_state.current_params[key]
            
                    val = st.sidebar.slider(
                        label=f"{key}",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=default_val,
                        step=0.001,
                    )
                    st.session_state.current_params[key] = val
                elif isinstance(bounds, bool):
                    if key not in st.session_state.current_params:
                        default_val = config.plane.get(key, bounds)
                    else:
                        default_val = st.session_state.current_params[key]
                    val = st.checkbox(
                        label=f"{key}",
                        value=float(default_val)
                    )
                    st.session_state.current_params[key] = val
                else:
                    # Show as text if there is only 1 value (or not a 2-tuple)
                    st.sidebar.text(f"{key}: {bounds}")
                    st.session_state.current_params[key] = config.plane.get(key, bounds)

            airfoil_keys = [
                'wing_airfoil_base', 
                'wing_airfoil_tip', 
                'winglet_airfoil', 
                'cannard_airfoil'
            ]

            st.divider()
            for key in airfoil_keys:
                # Determine default index based on config.plane
                default_val = config.plane.get(key)
                try:
                    default_index = config.airfoils.index(default_val)
                except ValueError:
                    default_index = 0  # Fallback to first option if not found
        
                st.session_state.current_params[key] = st.sidebar.selectbox(
                    label=f"{key.replace('_', ' ')}",
                    options=config.airfoils,
                    index=default_index
                )

            st.session_state.airplane = get_airplane(**st.session_state.current_params)
            if 'ab' not in st.session_state:
                st.session_state['ab'] = AeroLoss(st.session_state.airplane, alphas=np.array(config.alphas, dtype=float), method='AB', sim_on_set=False, verbose=True, airfoils=config.airfoils, targets=config.targets, target_range=config.target_range, velocity=config.velocity)
            if 'vlm' not in st.session_state:
                st.session_state['vlm'] = AeroLoss(st.session_state.airplane, alphas=np.array(config.alphas, dtype=float), method='VLM', sim_on_set=False, verbose=True, airfoils=config.airfoils, targets=config.targets, target_range=config.target_range, velocity=config.velocity)
            if 'll' not in st.session_state:
                st.session_state['ll'] = AeroLoss(st.session_state.airplane, alphas=np.array(config.alphas, dtype=float), method='LL', sim_on_set=False, verbose=True, airfoils=config.airfoils, targets=config.targets, target_range=config.target_range, velocity=config.velocity)

            st.divider()
            disabled = 'foot' in st.session_state.current_params and st.session_state.current_params['foot']
            if st.button("Run VLM Analysis", use_container_width=True, disabled = disabled, help = "Doesn't work with foot==True" if disabled else ''):
                st.session_state.vlm.set_airplane(st.session_state.airplane)
                st.session_state.results = st.session_state.vlm(parallel=True)
                st.session_state.results['CLCD'] = np.array(st.session_state.results['CL']) / np.array(st.session_state.results['CD'])
                go_to_results()
                st.rerun()
    
            if st.button("Run AB Analysis", use_container_width=True):
                st.session_state.ab.set_airplane(st.session_state.airplane)
                st.session_state.results = st.session_state.ab(parallel=True)
                st.session_state.results['CLCD'] = np.array(st.session_state.results['CL']) / np.array(st.session_state.results['CD'])
                go_to_results()
                st.rerun()

            if st.button("Run LL Analysis", use_container_width=True):
                st.session_state.ll.set_airplane(st.session_state.airplane)
                st.session_state.results = st.session_state.ll(parallel=True)
                st.session_state.results['CLCD'] = np.array(st.session_state.results['CL']) / np.array(st.session_state.results['CD'])
                go_to_results()
                st.rerun()

            st.divider()

            if st.button('Prepare airplane for export', use_container_width=True):
                st.session_state.vspscript, st.session_state.stepfile = prepare_files(st.session_state.airplane, st.session_state.tempdir.name)
                st.session_state.config_file = prepare_config(config, st.session_state.current_params, st.session_state.tempdir.name)

            if st.session_state.stepfile is not None:
                st.divider()
                with open(str(st.session_state.stepfile), "rb") as file:
                    btn = st.download_button(
                        label="Download Airplane STEP File",
                        data=file, # Pass the file object directly
                        file_name="airplane.step",
                        mime="application/octet-stream" # General MIME type for binary data
                    )
                with open(str(st.session_state.vspscript), "rb") as file:
                    btn1 = st.download_button(
                        label="Download Airplane vspscript File",
                        data=file, # Pass the file object directly
                        file_name="airplane.vspscript",
                        mime="application/octet-stream" # General MIME type for binary data
                    )

            if st.session_state.config_file is not None:
                with open(str(st.session_state.config_file), "rb") as file:
                    btn = st.download_button(
                        label="Download Airplane config.yaml File",
                        data=file, # Pass the file object directly
                        file_name="config.yaml",
                        mime="application/octet-stream" # General MIME type for binary data
                    )


        st.session_state.airplane.draw_three_view(show=False)
        fig = plt.gcf()
        st.pyplot(fig, clear_figure=True, width="content")

    elif st.session_state.page == "Results":
        st.title("Analysis Results")
        if st.button("← Back to Editor"):
            go_to_editor()
            st.rerun()

        if st.session_state.results:
            alphas = np.array(config.alphas, dtype=float)
        
            # Dynamically create plots for each target in config
            for target, value in config.targets.items():
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(alphas, st.session_state.results[target], marker='o', label=target)
                ax.axhline(y=value, color='red', linestyle='--', 
                           linewidth=2, label=f"Target {target}")
                ax.axvline(x=0, color='blue', linestyle='--', linewidth=2)
                ax.set_xlabel("Angle of Attack, deg")
                ax.set_ylabel(target)
                ax.set_title(f"{target} vs Angle of Attack")
                ax.grid(True)
            
                st.pyplot(fig, clear_figure=True, width="content")

