import streamlit as st
from opensees_model import (create_vehicle,
                            run_analysis,
                            plot_envelope_streamlit,
                            plot_two_vehicles_env_streamlit,
                            compute_ratios_between_vehicles,
                            plot_vehicle_ratio_streamlit)
from vehicle_library import PREDEFINED_VEHICLES


st.set_page_config(page_title="Beam model (OpenSees) — Streamlit", layout="wide")
# st.header("Bridge Properties Input")

# --- Sidebar: Model definition ---
st.sidebar.header("Model geometry and properties")
span_count = st.sidebar.number_input("Number of spans", min_value=1, max_value=10, value=3)

span_lengths = []
st.sidebar.subheader("Span lengths (m)")
for i in range(span_count):
    length = st.sidebar.number_input(f"Length of span {i+1} m",
                                     min_value=0.1,
                                     value=5.*0.8 if i!=1 else 5.0,
                                     step=0.1,
                                     key=f"span_len_{i}")
    span_lengths.append(length)


nodes_per_span = st.sidebar.number_input("Nodes per span", min_value=2, value=11)

st.sidebar.markdown("---")
# --- Support configuration ---
st.sidebar.header("Support Conditions")

support_options = ["second-class", "pinned", "fixed"]

supports = []


st.sidebar.write("Select support type for each node (span boundaries):")

for i in range(span_count + 1):
    support = st.sidebar.selectbox(
        f"Support No {i+1}",
        support_options,
        index=0,
        key=f"support_{i}"
    )
    supports.append(support)

st.sidebar.write("Selected supports:", supports)

st.sidebar.markdown("---")
st.sidebar.header("Estructural Properties")
E = st.sidebar.number_input("Elastic modulus E (kPa)", min_value=1e3, value=25_000_000.0, step=100000.0, format="%.1f")
A = st.sidebar.number_input("Area A (m2)", min_value=0.001, value=0.6, step=0.01)
I = st.sidebar.number_input("Inertia I (m4)", min_value=0.0001, value=0.05, step=0.001)

properties = {
    'span_lengths': span_lengths,
    'nodes_per_span': int(nodes_per_span),
    'support_types': supports,
    'E': float(E),
    'A': float(A),
    'I': float(I),
}

st.sidebar.markdown("---")
# --- Vehicle configuration ---
st.sidebar.header("Vehicle – Custom Axle Loads and Positions")

# Number of axles

vehicle_name = st.sidebar.text_input("Vehicle name", value="Custom Vehicle")
n_axles = st.sidebar.number_input("Number of axles", min_value=1, max_value=20, value=3)


axle_loads = []
axle_positions = []

st.sidebar.subheader("Axle-by-axle configuration")

for i in range(n_axles):
    load = st.sidebar.number_input(
        f"Axle {i+1} Load (kN)",
        min_value=0.0,
        value=98.0 if i == 0 else 441.0,
        step=1.0,
        key=f"axle_load_{i}"
    )
    axle_loads.append(load)

for i in range(n_axles-1):
    pos = st.sidebar.number_input(
        f"Axle {i+1} Position from front (m)",
        min_value=0.0,
        value=10.5 if i == 0 else 15.0,
        step=0.1,
        key=f"axle_pos_{i}"
    )
    axle_positions.append(pos)

trib_select = st.sidebar.number_input("Tributary width custom vehicle", min_value=1.0, value=1.0, step=0.5)

# Build final vehicle dict
cust_vehicle = {
    "name": vehicle_name,
    "n_axles": n_axles,
    "loads": [load/trib_select for load in axle_loads],
    "positions": axle_positions,
}

st.sidebar.write("Custom Vehicle:", cust_vehicle)

st.sidebar.markdown("---")

st.sidebar.header("Reference Vehicle Selection")

vehicle_names = list(PREDEFINED_VEHICLES.keys())

selected_vehicle = st.sidebar.selectbox(
    "Choose a predefined vehicle",
    vehicle_names
)

trib_ref = st.sidebar.number_input("Tributary width Reference Vehicle", min_value=1.0, value=1.0, step=0.5)

if selected_vehicle != "Custom vehicle":

    veh_data = PREDEFINED_VEHICLES[selected_vehicle]

    n_axles_ref = veh_data["n_axles"]
    axle_loads_ref = [load/trib_ref for load in veh_data["axle_loads"]]
    axle_positions_ref = veh_data["axle_positions"]
    vehicle_name_ref = selected_vehicle

    st.sidebar.success(f"Loaded: {selected_vehicle}")

ref_vehicle = {
    "name": selected_vehicle,
    "n_axles": veh_data["n_axles"],
    "axle_loads": [load/trib_ref for load in veh_data["axle_loads"]],
    "axle_positions": veh_data["axle_positions"],
}

st.sidebar.write("Reference Vehicle:", ref_vehicle)

st.sidebar.markdown("---")





# --------- BUTTON TO USE THE DICTIONARY ----------
if st.sidebar.button("Run Analysis"):
    st.sidebar.success("Analysis Complete!")
    create_vehicle(properties, n_axles, [load/trib_select for load in axle_loads], axle_positions, vehicle_name)
    create_vehicle(properties, n_axles_ref, [load/trib_ref for load in veh_data["axle_loads"]], axle_positions_ref, vehicle_name_ref)

    run_analysis(properties)

    ratios = compute_ratios_between_vehicles(properties,
        ref_vehicle=selected_vehicle,
        new_vehicle=vehicle_name
    )

    # st.header("Envelope diagrams")

    figV_ref, figM_ref = plot_envelope_streamlit(properties, selected_vehicle)
    figV_cust, figM_cust = plot_envelope_streamlit(properties, vehicle_name)

    figV, figM = plot_two_vehicles_env_streamlit(properties, selected_vehicle, vehicle_name)

    figVratio, figMratio = plot_vehicle_ratio_streamlit(properties, selected_vehicle, vehicle_name)

    colA, colB = st.columns(2)

    with colA:

        st.subheader(f"Shear Ratio - {vehicle_name} / {selected_vehicle}")
        st.pyplot(figVratio)
        st.subheader(f"Shear Envelope - {selected_vehicle}")
        st.pyplot(figV_ref)
        st.subheader(f"Shear Envelope - {vehicle_name}")
        st.pyplot(figV_cust)
        st.subheader("Shear Envelope")
        st.pyplot(figV)

    with colB:

        st.subheader(f"Moment Ratio - {vehicle_name} / {selected_vehicle}")
        st.pyplot(figMratio)
        st.subheader(f"Moment Envelope - {selected_vehicle}")
        st.pyplot(figM_ref)
        st.subheader(f"Moment Envelope - {vehicle_name}")
        st.pyplot(figM_cust)
        st.subheader("Moment Envelope")
        st.pyplot(figM)

# st.json(vehicle_properties)
# # Show the dictionary to the user
# st.subheader("Generated Properties Dictionary")
# st.json(properties)



