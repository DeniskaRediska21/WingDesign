import pyswarms as ps
from pathlib import Path
import copy
from functools import partial
import aerosandbox.numpy as np
import matplotlib
from utils import AeroLoss, get_airplane, convert_numpy, OptFuncSwither
from addict import Addict
from datetime import datetime
import streamlit as st
import matplotlib.pyplot as plt
import yaml

st.set_page_config(layout="wide")
matplotlib.use('Qt5Agg')

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

    def go_to_results(): st.session_state.page = "Results"
    def go_to_editor(): st.session_state.page = "Editor"


    if st.session_state.page == "Editor":
        with st.sidebar:
            for key, bounds in config.constraints.items():
                if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                    # Create a slider if there are two values (min, max)
                    min_val, max_val = bounds
                    default_val = config.plane.get(key, min_val)
            
                    val = st.sidebar.slider(
                        label=f"{key}",
                        min_value=float(min_val),
                        max_value=float(max_val),
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

            col1, col2 = st.columns(2)
            if col1.button("Run VLM Analysis"):
                st.session_state.vlm.set_airplane(st.session_state.airplane)
                st.session_state.results = st.session_state.vlm()
                st.session_state.results['CLCD'] = np.array(st.session_state.results['CL']) / np.array(st.session_state.results['CD'])
                go_to_results()
                st.rerun()
    
            if col2.button("Run AB Analysis"):
                st.session_state.ab.set_airplane(st.session_state.airplane)
                st.session_state.results = st.session_state.ab()
                st.session_state.results['CLCD'] = st.session_state.results['CL'] / st.session_state.results['CD']
                go_to_results()
                st.rerun()


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
                ax.set_xlabel("Alpha (Angle of Attack)")
                ax.set_ylabel(target)
                ax.set_title(f"{target} vs Alpha")
                ax.grid(True)
            
                st.pyplot(fig, clear_figure=True, width="content")
