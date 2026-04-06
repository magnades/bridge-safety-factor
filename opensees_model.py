import openseespy.opensees as ops
import numpy as np
from trame_vtk.modules.vtk.serializers.helpers import linspace


# --- Build node coordinates ---
# Create node coordinates along the beam (1D line in X, y=0)

def build_beam_geometry(prop_dict):

    span_lengths = prop_dict['span_lengths']
    nodes_per_span = prop_dict['nodes_per_span']

    coords = [0.0]
    supports_id = [1]  # first node is always a support
    for L in span_lengths:
        # create nodes for this span excluding first point (already present)
        xs = np.linspace(coords[-1], coords[-1] + L, nodes_per_span)
        supports_id.append(supports_id[-1]+len(xs)-1)
        # skip the first because it's duplicated
        for x in xs[1:]:
            coords.append(float(x))

    n_nodes = len(coords)

    prop_dict["n_nodes"] = n_nodes
    prop_dict["node_ids"] = list(range(1, n_nodes + 1))
    prop_dict["coords"] = coords
    prop_dict["supports_id"] = supports_id


# --- Helper: map support preset to fix flags ---
def support_flags(type):
    """
    Return fixity flags (ux, uy, rz) using Python 3.10+ match-case.

    Presets:
        - "second-class"  → left pinned, right roller
        - "pinned"
        - "fixed"
    """

    match type:
        case "second-class":
            return (1, 1, 0)

        case "pinned":
            return (0, 1, 0)

        case "fixed":
            return (1, 1, 1)

        case _:
            # Default → second-class
            return (1, 1, 0)


# --- OpenSees model build and analysis functions ---

def build_opensees_model(prop_dict):
    ops.wipe()
    print("ops.wipe()")
    ops.model('basic', '-ndm', 2, '-ndf', 3)
    print("ops.model('basic', '-ndm', 2, '-ndf', 3)")

    # create geometric transformation
    ops.geomTransf('Linear', 1)
    print("ops.geomTransf('Linear', 1)")

    # Variables
    node_ids = prop_dict['node_ids']
    coords = prop_dict['coords']

    # create nodes

    for nid, x in zip(node_ids, coords):
        ops.node(nid, x, 0.0)
        print(f"ops.node({nid}, {x}, 0.0)")


    support_types = prop_dict['support_types']
    support_ids = prop_dict['supports_id']

    # apply boundary conditions at span ends only (first and last node)
    for id, type in zip(support_ids, support_types):
        flags = support_flags(type)
        ops.fix(id, *flags)
        print(f"ops.fix({id}, {flags})")

    # create elements (one element between each pair of nodes)
    E = prop_dict['E']  # Young's modulus in kPa
    A = prop_dict['A']  # Cross-sectional area in m^2
    I = prop_dict['I']  # Moment of inertia in m^4

    element_ids = []

    for i in range(len(node_ids)-1):
        eid = i+1
        element_ids.append(eid)
        ops.element('elasticBeamColumn', eid, node_ids[i], node_ids[i+1], A, E, I, 1)
        print(f"ops.element('elasticBeamColumn', {eid}, {node_ids[i]}, {node_ids[i+1]}, {A}, {E}, {I}, 1)")

    prop_dict["element_ids"] = element_ids

    # define load pattern (constant time series)
    ops.timeSeries('Linear', 1)
    print("ops.timeSeries('Linear', 1)")
    ops.pattern('Plain', 1, 1)
    print("ops.pattern('Plain', 1, 1)")


def run_static_analysis():
    # prepare analysis

    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('NormDispIncr', 1.0e-6, 6, 2)
    ops.algorithm('Linear')
    ops.integrator('LoadControl', 1)
    ops.analysis('Static')
    ok = ops.analyze(1)
    return ok

def plot_internal_forces():
    import opsvis as opsv
    import matplotlib.pyplot as plt

    ops.printModel()

    opsv.plot_model()

    plt.title('plot_model after defining elements')

    opsv.plot_load()

    opsv.plot_reactions()

    # sfac = 80.

    opsv.plot_defo()

    # opsv.plot_defo(sfac)

    # fmt_interp = {'color': 'blue', 'linestyle': 'solid', 'linewidth': 1.2, 'marker': '.', 'markersize': 6}

    # opsv.plot_defo(sfac, fmt_interp=fmt_interp)

    # 4. plot N, V, M forces diagrams

    sfacN, sfacV, sfacM = 5.e-5, 5.e-5, 5.e-5

    opsv.section_force_diagram_2d('N', sfacN)

    max, min, ax = opsv.section_force_diagram_2d('M', sfacM)

    plt.title('Axial force distribution')

    opsv.section_force_diagram_2d('T', sfacV)

    plt.title('Shear force distribution')

    opsv.section_force_diagram_2d('M', sfacM)


    plt.title('Bending moment distribution')

    plt.show()


def nearest_node(x_target):
    """
    Devuelve el ID del nodo más cercano a la coordenada x_target.

    node_list : lista de nodos existentes
    x_target  : coordenada x donde quieres localizar el nodo más cercano
    """
    # obtener coordenadas x de cada nodo
    node_list = ops.getNodeTags()
    coords = {n: ops.nodeCoord(n)[0] for n in node_list}

    # identificar nodo más cercano
    nearest = min(node_list, key=lambda n: abs(coords[n] - x_target))

    return nearest


def create_vehicle(prop_dict, num_axles, loads, distances, name):
    """
    Creates a vehicle definition using:
    - number of axles
    - axle loads
    - distances between axles

    Parameters
    ----------
    num_axles : int
        Total number of axles of the vehicle.

    loads : list of float
        List of axle loads [P1, P2, ...].
        Must have length = num_axles.

    distances : list of float
        Distances between consecutive axles [d12, d23, ...].
        Must have length = num_axles - 1.

    Returns
    -------
    vehicle : dict
        {
            "num_axles": num_axles,
            "loads": [...],
            "distances": [...],
            "positions": [...],   # x position of each axle (0 = front axle)
        }
    """

    if len(loads) != num_axles:
        raise ValueError("The loads list must have 'num_axles' elements.")

    if len(distances) != num_axles - 1:
        raise ValueError("The distances list must have 'num_axles - 1' elements.")

    # Compute cumulative axle positions
    positions = [0.0]  # first axle at x = 0
    for d in distances:
        positions.append(positions[-1] + d)

    if 'vehicles' not in prop_dict:
        prop_dict['vehicles'] = {}

    prop_dict['vehicles'][name] = {
        "num_axles": num_axles,
        "loads": loads,
        "distances": distances,
        "positions": positions
    }


def vehicle_loads_at_position(prop_dict, x_ref, vehicle_name, direction='right'):
    """
    Compute the loads applied by a vehicle when its first axle is located at x_ref.

    Parameters
    ----------
    x_ref : float
        Coordinate of the first axle (reference axle).

    vehicle : dict
        Vehicle dictionary created by `create_vehicle()`, containing:
            - "num_axles"
            - "loads"
            - "positions"  (relative axle positions)

    direction : str
        Direction of travel:
            "right"  = vehicle moving in positive x direction
            "left"  = vehicle moving in negative x direction

    Returns
    -------
    list of tuples (node_id, load)
        Example: [(12, 150.0), (15, 150.0), ...]
    """

    if direction not in ["right", "left"]:
        raise ValueError("direction must be 'right' or 'left'")

    vehicle = prop_dict['vehicles'][vehicle_name]

    node_loads = []

    for i in range(vehicle["num_axles"]):
        axle_offset = vehicle["positions"][i]

        # Compute global position
        if direction == "right":
            x_axle = x_ref - axle_offset
        else:  # direction == "-"
            x_axle = x_ref + axle_offset

        # Get nearest node
        if x_axle < 0 or x_axle > prop_dict['coords'][-1]:
            continue  # axle before start of beam
        node_id = nearest_node(x_axle)

        # Append (node_id, load)
        node_loads.append((node_id, vehicle["loads"][i]))
        print(f'Axle {i+1}: x={x_axle:.2f} m -> Node {node_id} with load {vehicle["loads"][i]} kN')

    return node_loads


def run_vehicle_load_analysis(prop_dict, x_ref, vehicle_name, direction='right'):

    ops.wipe()
    build_beam_geometry(prop_dict)
    build_opensees_model(prop_dict)

    # Clear existing loads by re-creating load pattern
    ops.remove("loadPattern", 2)
    ops.pattern('Plain', 2, 1)

    # Get vehicle loads at position
    node_loads = vehicle_loads_at_position(prop_dict, x_ref, vehicle_name, direction)

    # Apply loads
    for node_id, load in node_loads:
        ops.load(node_id, 0.0, -load, 0.0)
        print(f"ops.load({node_id}, 0.0, {-load}, 0.0)")

    # Run analysis
    ok = run_static_analysis()
    return ok


def update_internal_forces(prop_dict, vehicle_name):
    """
    Reads internal forces from each element after an analysis step and updates
    max/min history for each vehicle separately.
    """

    # Ensure vehicles container exists
    if 'vehicles' not in prop_dict:
        prop_dict['vehicles'] = {}

    # Ensure the specific vehicle exists
    if vehicle_name not in prop_dict['vehicles']:
        raise ValueError(f"Vehicle '{vehicle_name}' not defined in properties.")

    # Ensure top-level container for forces exists for this vehicle
    if 'forces' not in prop_dict['vehicles'][vehicle_name]:
        prop_dict['vehicles'][vehicle_name]['forces'] = {}

    if 'reactions' not in prop_dict['vehicles'][vehicle_name]:
        prop_dict['vehicles'][vehicle_name]['reactions'] = {}

    forces_dict = prop_dict['vehicles'][vehicle_name]['forces']
    reactions_dict = prop_dict['vehicles'][vehicle_name]['reactions']

    # Get element list
    element_ids = ops.getEleTags()

    # Get internal forces
    forces = getFrameForces(prop_dict)
    node_list = ops.getNodeTags()

    int_forces_dict = {n: {'P': forces['axialForce'][n-1], 'V': forces['shearForceZ'][n-1], 'M': forces['torsion'][n-1]} for n in node_list}


    for ele_tag in element_ids:

        # Init element container if needed
        if ele_tag not in forces_dict:
            forces_dict[ele_tag] = {}

        # Get internal forces

        ndi, ndj = ops.eleNodes(ele_tag)


        current_forces = {
            'P_i': int_forces_dict[ndi]['P'], 'V_i': int_forces_dict[ndi]['V'], 'M_i': int_forces_dict[ndi]['M'],
            'P_j': int_forces_dict[ndj]['P'], 'V_j': int_forces_dict[ndj]['V'], 'M_j': int_forces_dict[ndj]['M'],
        }

        # Update min/max
        for fkey, value in current_forces.items():

            if fkey not in forces_dict[ele_tag]:
                forces_dict[ele_tag][fkey] = {'max': value, 'min': value}

            else:
                if value > forces_dict[ele_tag][fkey]['max']:
                    forces_dict[ele_tag][fkey]['max'] = value

                if value < forces_dict[ele_tag][fkey]['min']:
                    forces_dict[ele_tag][fkey]['min'] = value


    ## Grabacion de Reacciones

    ops.reactions()
    for supp_id in prop_dict['supports_id']:


        rx, ry, mz = ops.nodeReaction(supp_id)

        current_reactions = {
            'Rx': rx,
            'Ry': ry,
            'Mz': mz,
        }

        # Update min/max
        for rkey, value in current_reactions.items():

            if supp_id not in reactions_dict:
                reactions_dict[supp_id] = {}

            if rkey not in reactions_dict[supp_id]:
                reactions_dict[supp_id][rkey] = {'max': value, 'min': value}

            else:
                if value > reactions_dict[supp_id][rkey]['max']:
                    reactions_dict[supp_id][rkey]['max'] = value

                if value < reactions_dict[supp_id][rkey]['min']:
                    reactions_dict[supp_id][rkey]['min'] = value



def build_envelope_from_history(prop_dict, vehicle_name, coord_index=0):
    """
    Converts stored element-end max/min forces into nodal envelopes for a beam.
    Returns:
        {
            'x': [...],
            'V_max': [...], 'V_min': [...],
            'M_max': [...], 'M_min': [...],
            'node_order': [...]
        }
    """

    # Get element force history
    forces_dict = prop_dict['vehicles'][vehicle_name]['forces']

    # Storage for per-node contributions
    node_data = {}   # node : dict with 'x', 'V_list', 'M_list'

    for ele_tag, result in forces_dict.items():

        # get nodes of element
        ni, nj = ops.eleNodes(ele_tag)

        # get coords
        xi = ops.nodeCoord(ni)[coord_index]
        xj = ops.nodeCoord(nj)[coord_index]

        # ensure containers
        if ni not in node_data:
            node_data[ni] = {'x': xi, 'V': [], 'M': []}
        if nj not in node_data:
            node_data[nj] = {'x': xj, 'V': [], 'M': []}

        # --- LEFT end (i end)
        Vi_max = result['V_i']['max']
        Vi_min = result['V_i']['min']
        Mi_max = result['M_i']['max']
        Mi_min = result['M_i']['min']

        node_data[ni]['V'] += [Vi_max, Vi_min]
        node_data[ni]['M'] += [Mi_max, Mi_min]


        # --- RIGHT end (j end)
        Vj_max = result['V_j']['max']
        Vj_min = result['V_j']['min']
        Mj_max = result['M_j']['max']
        Mj_min = result['M_j']['min']

        node_data[nj]['V'] += [Vj_max, Vj_min]
        node_data[nj]['M'] += [Mj_max, Mj_min]


    # --- Build node-based envelope
    nodes_sorted = sorted(node_data.items(), key=lambda kv: kv[1]['x'])

    x_list = []
    V_max_list = []
    V_min_list = []
    M_max_list = []
    M_min_list = []

    for node, info in nodes_sorted:
        x_list.append(info['x'])

        V_max_list.append(max(info['V']))
        V_min_list.append(min(info['V']))

        M_max_list.append(max(info['M']))
        M_min_list.append(min(info['M']))

    prop_dict['vehicles'][vehicle_name]['plot_info'] = {
        'x': x_list,
        'V_max': V_max_list,
        'V_min': V_min_list,
        'M_max': M_max_list,
        'M_min': M_min_list,
        'node_order': [n for n,_ in nodes_sorted]
    }


def plot_envelope_with_labels(prop_dict, vehicle_name, title_prefix="Envelope", decimals=2, text_offset=0.015):
    """
    Plots V and M envelope diagrams with annotated points.

    env = output of build_envelope_from_history()
    decimals = number of decimals for printed values
    text_offset = vertical offset (fraction of diagram range)
    """

    import matplotlib.pyplot as plt

    env = prop_dict['vehicles'][vehicle_name]['plot_info']

    x = env['x']

    # ----------------- SHEAR -----------------
    Vmax = env['V_max']
    Vmin = env['V_min']

    fig = plt.figure()
    plt.plot(x, Vmax, marker='o', label="V max")
    plt.plot(x, Vmin, marker='o', label="V min")
    plt.fill_between(x, Vmin, Vmax, alpha=0.20)

    # Range for text offset
    if len(Vmax) > 0:
        V_range = max(Vmax + Vmin) - min(Vmax + Vmin)
    else:
        V_range = 1.0

    for xi, v in zip(x, Vmax):
        plt.text(
            xi, v + text_offset * V_range,
            f"{v:.{decimals}f}",
            ha='center', va='bottom', fontsize=8
        )
    for xi, v in zip(x, Vmin):
        plt.text(
            xi, v - text_offset * V_range,
            f"{v:.{decimals}f}",
            ha='center', va='top', fontsize=8
        )

    plt.title(f"{title_prefix} – Shear")
    plt.xlabel("Position (m)")
    plt.ylabel("Shear Force (kN)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ----------------- MOMENT -----------------
    Mmax = env['M_max']
    Mmin = env['M_min']

    fig = plt.figure()
    plt.plot(x, Mmax, marker='o', label="M max")
    plt.plot(x, Mmin, marker='o', label="M min")
    plt.fill_between(x, Mmin, Mmax, alpha=0.20)

    # Range for text offset
    if len(Mmax) > 0:
        M_range = max(Mmax + Mmin) - min(Mmax + Mmin)
    else:
        M_range = 1.0

    for xi, m in zip(x, Mmax):
        plt.text(
            xi, m + text_offset * M_range,
            f"{m:.{decimals}f}",
            ha='center', va='bottom', fontsize=8
        )
    for xi, m in zip(x, Mmin):
        plt.text(
            xi, m - text_offset * M_range,
            f"{m:.{decimals}f}",
            ha='center', va='top', fontsize=8
        )

    plt.title(f"{title_prefix} – Moment")
    plt.xlabel("Position (m)")
    plt.ylabel("Bending Moment (kN)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()

def plot_envelope_streamlit(prop_dict, vehicle_name, title_prefix="Envelope", decimals=2, text_offset=0.015):
    """
    Returns two matplotlib figures:
    - fig_V : shear envelope figure
    - fig_M : moment envelope figure
    """

    import matplotlib.pyplot as plt

    env = prop_dict['vehicles'][vehicle_name]['plot_info']

    x = env['x']

    # ----------------- SHEAR -----------------
    Vmax = env['V_max']
    Vmin = env['V_min']

    fig_V, axV = plt.subplots()

    axV.plot(x, Vmax, marker='o', label="V max")
    axV.plot(x, Vmin, marker='o', label="V min")
    axV.fill_between(x, Vmin, Vmax, alpha=0.20)

    # Range
    V_range = max(Vmax + Vmin) - min(Vmax + Vmin) if len(Vmax) else 1.0

    for xi, v in zip(x, Vmax):
        axV.text(xi, v + text_offset * V_range, f"{v:.{decimals}f}",
                 ha='center', va='bottom', fontsize=8)
    for xi, v in zip(x, Vmin):
        axV.text(xi, v - text_offset * V_range, f"{v:.{decimals}f}",
                 ha='center', va='top', fontsize=8)

    axV.set_title(f"{title_prefix} – Shear")
    axV.set_xlabel("Position (m)")
    axV.set_ylabel("Shear Force (kN)")
    axV.grid(True)
    axV.legend()
    fig_V.tight_layout()

    # ----------------- MOMENT -----------------
    Mmax = env['M_max']
    Mmin = env['M_min']

    fig_M, axM = plt.subplots()

    axM.plot(x, Mmax, marker='o', label="M max")
    axM.plot(x, Mmin, marker='o', label="M min")
    axM.fill_between(x, Mmin, Mmax, alpha=0.20)

    # Range
    M_range = max(Mmax + Mmin) - min(Mmax + Mmin) if len(Mmax) else 1.0

    for xi, m in zip(x, Mmax):
        axM.text(xi, m + text_offset * M_range, f"{m:.{decimals}f}",
                 ha='center', va='bottom', fontsize=8)
    for xi, m in zip(x, Mmin):
        axM.text(xi, m - text_offset * M_range, f"{m:.{decimals}f}",
                 ha='center', va='top', fontsize=8)

    axM.set_title(f"{title_prefix} – Moment")
    axM.set_xlabel("Position (m)")
    axM.set_ylabel("Bending Moment (kN)")
    axM.grid(True)
    axM.legend()
    fig_M.tight_layout()

    return fig_V, fig_M

def plot_two_vehicles_envelopes(prop_dict, veh_gray, veh_blue, coord_index=0):
    """
    Plots envelopes of two vehicles:
    - veh_gray in grey (both max/min)
    - veh_blue in blue (both max/min)
    Only one legend label per vehicle.
    """

    import matplotlib.pyplot as plt

    # Load envelopes
    envG = prop_dict['vehicles'][veh_gray]['plot_info']
    envB = prop_dict['vehicles'][veh_blue]['plot_info']

    xG = envG['x']
    xB = envB['x']

    # ---------------------------------------------------------
    #                       SHEAR
    # ---------------------------------------------------------
    plt.figure()

    # Grey vehicle
    plt.plot(xG, envG['V_max'], color='grey', linestyle='-', marker='o', label=veh_gray)
    plt.plot(xG, envG['V_min'], color='grey', linestyle='-', marker='o')
    plt.fill_between(xG, envG['V_min'], envG['V_max'], color='grey', alpha=0.15)

    # Blue vehicle
    plt.plot(xB, envB['V_max'], color='blue', linestyle='-', marker='s', label=veh_blue)
    plt.plot(xB, envB['V_min'], color='blue', linestyle='-', marker='s')
    plt.fill_between(xB, envB['V_min'], envB['V_max'], color='blue', alpha=0.15)

    plt.title(f"Shear Envelope – {veh_gray} (grey) vs {veh_blue} (blue)")
    plt.xlabel("Position")
    plt.ylabel("Shear")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ---------------------------------------------------------
    #                       MOMENT
    # ---------------------------------------------------------
    plt.figure()

    # Grey vehicle
    plt.plot(xG, envG['M_max'], color='grey', linestyle='-', marker='o', label=veh_gray)
    plt.plot(xG, envG['M_min'], color='grey', linestyle='-', marker='o')
    plt.fill_between(xG, envG['M_min'], envG['M_max'], color='grey', alpha=0.15)

    # Blue vehicle
    plt.plot(xB, envB['M_max'], color='blue', linestyle='-', marker='s', label=veh_blue)
    plt.plot(xB, envB['M_min'], color='blue', linestyle='-', marker='s')
    plt.fill_between(xB, envB['M_min'], envB['M_max'], color='blue', alpha=0.15)

    plt.title(f"Moment Envelope – {veh_gray} (grey) vs {veh_blue} (blue)")
    plt.xlabel("Position")
    plt.ylabel("Moment")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()

def plot_two_vehicles_env_streamlit(prop_dict, veh_gray, veh_blue, coord_index=0):
    """
    Returns two matplotlib figures for Streamlit:
    - Shear envelope comparison
    - Moment envelope comparison
    """

    import matplotlib.pyplot as plt

    # Load envelopes
    envG = prop_dict['vehicles'][veh_gray]['plot_info']
    envB = prop_dict['vehicles'][veh_blue]['plot_info']

    xG = envG['x']
    xB = envB['x']

    # ---------------------------------------------------------
    #                  FIGURE 1 — SHEAR
    # ---------------------------------------------------------
    figV, axV = plt.subplots(figsize=(8, 4))

    # Grey vehicle
    axV.plot(xG, envG['V_max'], color='grey', linestyle='-', marker='o', label=f"{veh_gray} max/min")
    axV.plot(xG, envG['V_min'], color='grey', linestyle='-', marker='o')
    axV.fill_between(xG, envG['V_min'], envG['V_max'], color='grey', alpha=0.15)

    # Blue vehicle
    axV.plot(xB, envB['V_max'], color='blue', linestyle='-', marker='s', label=f"{veh_blue} max/min")
    axV.plot(xB, envB['V_min'], color='blue', linestyle='-', marker='s')
    axV.fill_between(xB, envB['V_min'], envB['V_max'], color='blue', alpha=0.15)

    axV.set_title(f"Shear Envelope – {veh_gray} vs {veh_blue}")
    axV.set_xlabel("Position (m)")
    axV.set_ylabel("Shear (kN)")
    axV.grid(True)
    axV.legend()
    figV.tight_layout()

    # ---------------------------------------------------------
    #                  FIGURE 2 — MOMENT
    # ---------------------------------------------------------
    figM, axM = plt.subplots(figsize=(8, 4))

    # Grey vehicle
    axM.plot(xG, envG['M_max'], color='grey', linestyle='-', marker='o', label=f"{veh_gray} max/min")
    axM.plot(xG, envG['M_min'], color='grey', linestyle='-', marker='o')
    axM.fill_between(xG, envG['M_min'], envG['M_max'], color='grey', alpha=0.15)

    # Blue vehicle
    axM.plot(xB, envB['M_max'], color='blue', linestyle='-', marker='s', label=f"{veh_blue} max/min")
    axM.plot(xB, envB['M_min'], color='blue', linestyle='-', marker='s')
    axM.fill_between(xB, envB['M_min'], envB['M_max'], color='blue', alpha=0.15)

    axM.set_title(f"Moment Envelope – {veh_gray} vs {veh_blue}")
    axM.set_xlabel("Position (m)")
    axM.set_ylabel("Moment (kN·m)")
    axM.grid(True)
    axM.legend()
    figM.tight_layout()

    return figV, figM


def compute_ratios_between_vehicles(prop_dict, ref_vehicle, new_vehicle, coord_index=0):
    """
    Computes node-by-node ratios between a reference vehicle and a new vehicle
    using envelope results. Ratios are stored in prop_dict for the new vehicle.

    Ratio = new_vehicle_value / reference_vehicle_value
    """

    # 1. Build envelopes
    env_ref = prop_dict['vehicles'][ref_vehicle]['plot_info']
    env_new = prop_dict['vehicles'][new_vehicle]['plot_info']

    # Extract data
    x_ref = env_ref['x']
    x_new = env_new['x']

    # --- Ensure output container exists ---
    if 'ratios' not in prop_dict['vehicles'][new_vehicle]:
        prop_dict['vehicles'][new_vehicle]['ratios'] = {}

    ratio_dict = prop_dict['vehicles'][new_vehicle]['ratios']

    # --- Node matching strategy ---
    # We match by *x-coordinate* because the vehicle envelopes
    # may not have the same node IDs but positions should match.
    for i in range(len(x_new)):

        x_target = x_new[i]

        # Find closest point in reference envelope
        # (handles tolerance in floating coordinates)
        distances = [abs(xr - x_target) for xr in x_ref]
        j = distances.index(min(distances))  # index of closest reference node

        # Load values
        Vref_max = env_ref['V_max'][j]
        Vref_min = env_ref['V_min'][j]
        Mref_max = env_ref['M_max'][j]
        Mref_min = env_ref['M_min'][j]

        Vnew_max = env_new['V_max'][i]
        Vnew_min = env_new['V_min'][i]
        Mnew_max = env_new['M_max'][i]
        Mnew_min = env_new['M_min'][i]

        # Avoid division by zero
        def safe_ratio(new, ref):
            if abs(ref) < 1e-12:
                return 0.   # or 0, or np.nan
            return new / ref

        # Compute ratios
        ratios = {
            'V_max_ratio': safe_ratio(Vnew_max, Vref_max),
            'V_min_ratio': safe_ratio(Vnew_min, Vref_min),
            'M_max_ratio': safe_ratio(Mnew_max, Mref_max),
            'M_min_ratio': safe_ratio(Mnew_min, Mref_min),
            'x_position': x_target,
        }

        # Save using x-position as unique key
        ratio_dict[x_target] = ratios

    return ratio_dict

def compute_global_fs(prop_dict, cust_vehicle):
    """
    Computes global factors of safety (minimum ratios) between a reference vehicle
    and a new vehicle using envelope results.

    Returns:
        {
            'V_max_fs': ...,
            'V_min_fs': ...,
            'M_max_fs': ...,
            'M_min_fs': ...,
        }
    """

    ratios = prop_dict['vehicles'][cust_vehicle]['ratios']

    V_max_ratios = [ratios[xi]['V_max_ratio'] for xi in ratios]
    V_min_ratios = [ratios[xi]['V_min_ratio'] for xi in ratios]
    M_max_ratios = [ratios[xi]['M_max_ratio'] for xi in ratios]
    M_min_ratios = [ratios[xi]['M_min_ratio'] for xi in ratios]

    global_fs = {
        'FSV': min(V_max_ratios + V_min_ratios),
        'FSM': min(M_max_ratios + M_min_ratios),
    }

    return global_fs


def plot_vehicle_ratio_lines(prop_dict, ref_vehicle, new_vehicle):
    """
    Plots the ratios (new/ref) along the beam for:
      - V_max_ratio
      - V_min_ratio
      - M_max_ratio
      - M_min_ratio

    Uses the ratio dictionary produced by compute_ratios_between_vehicles().
    """

    import matplotlib.pyplot as plt

    # Retrieve ratio data
    ratios = prop_dict['vehicles'][new_vehicle]['ratios']

    # Extract in order of increasing x
    x = sorted(ratios.keys())

    Vmax_r = [ratios[xi]['V_max_ratio'] for xi in x]
    Vmin_r = [ratios[xi]['V_min_ratio'] for xi in x]
    Mmax_r = [ratios[xi]['M_max_ratio'] for xi in x]
    Mmin_r = [ratios[xi]['M_min_ratio'] for xi in x]

    # ---------------------------------------------------------
    #               Shear Ratio Plot
    # ---------------------------------------------------------
    plt.figure()
    plt.plot(x, Vmax_r, marker='o', linestyle='-', label='V_max_ratio', color='red')
    plt.plot(x, Vmin_r, marker='o', linestyle='--', label='V_min_ratio', color='blue')

    plt.axhline(1.0, color='black', linestyle=':', label='ratio = 1')
    plt.grid(True)
    plt.xlabel("Position")
    plt.ylabel("Shear Ratio (new / reference)")
    plt.title(f"Shear Ratios – {new_vehicle} / {ref_vehicle}")
    plt.legend()
    plt.tight_layout()

    # ---------------------------------------------------------
    #               Moment Ratio Plot
    # ---------------------------------------------------------
    plt.figure()
    plt.plot(x, Mmax_r, marker='s', linestyle='-', label='M_max_ratio', color='red')
    plt.plot(x, Mmin_r, marker='s', linestyle='--', label='M_min_ratio', color='blue')

    plt.axhline(1.0, color='black', linestyle=':', label='ratio = 1')
    plt.grid(True)
    plt.xlabel("Position")
    plt.ylabel("Moment Ratio (new / reference)")
    plt.title(f"Moment Ratios – {new_vehicle} / {ref_vehicle}")
    plt.legend()
    plt.tight_layout()

    plt.show()

def plot_vehicle_ratio_streamlit(prop_dict, ref_vehicle, new_vehicle):
    """
    Returns two matplotlib figures for Streamlit:
      - V_max_ratio & V_min_ratio vs position
      - M_max_ratio & M_min_ratio vs position
    """

    import matplotlib.pyplot as plt

    ratios = prop_dict['vehicles'][new_vehicle]['ratios']

    # Sorted position keys
    x = sorted(ratios.keys())

    Vmax_r = [ratios[xi]['V_max_ratio'] for xi in x]
    Vmin_r = [ratios[xi]['V_min_ratio'] for xi in x]
    Mmax_r = [ratios[xi]['M_max_ratio'] for xi in x]
    Mmin_r = [ratios[xi]['M_min_ratio'] for xi in x]

    # ---------------------------------------------------------
    #                 FIGURE 1 — SHEAR RATIO
    # ---------------------------------------------------------
    figVratio, axV = plt.subplots(figsize=(8, 4))

    axV.plot(x, Vmax_r, marker='o', linestyle='-', color='red',  label='V_max_ratio')
    axV.plot(x, Vmin_r, marker='o', linestyle='--', color='blue', label='V_min_ratio')

    axV.axhline(1.0, color='black', linestyle=':', label='ratio = 1')

    axV.set_title(f"Shear Ratios – {new_vehicle} / {ref_vehicle}")
    axV.set_xlabel("Position (m)")
    axV.set_ylabel("Shear Ratio")
    axV.grid(True)
    axV.legend()
    figVratio.tight_layout()

    # ---------------------------------------------------------
    #                 FIGURE 2 — MOMENT RATIO
    # ---------------------------------------------------------
    figMratio, axM = plt.subplots(figsize=(8, 4))

    axM.plot(x, Mmax_r, marker='s', linestyle='-',  color='red',  label='M_max_ratio')
    axM.plot(x, Mmin_r, marker='s', linestyle='--', color='blue', label='M_min_ratio')

    axM.axhline(1.0, color='black', linestyle=':', label='ratio = 1')

    axM.set_title(f"Moment Ratios – {new_vehicle} / {ref_vehicle}")
    axM.set_xlabel("Position (m)")
    axM.set_ylabel("Moment Ratio")
    axM.grid(True)
    axM.legend()
    figMratio.tight_layout()

    return figVratio, figMratio



def run_analysis(prop_dict):
    for vehicle in prop_dict['vehicles']:
        for direction in ['left', 'right']:
            for x_ref in linspace(0, sum(prop_dict['span_lengths']), 21 ):

                print(f'\n--- Analyzing position x_ref={x_ref:.2f} m, direction={direction} ---')
                ok = run_vehicle_load_analysis(prop_dict, x_ref, vehicle, direction=direction)
                update_internal_forces(prop_dict, vehicle)

        build_envelope_from_history(prop_dict, vehicle_name=vehicle)
        plot_envelope_with_labels(prop_dict, vehicle_name=vehicle, title_prefix=f"{vehicle} Envelope")



def getFrameForces(prop_dict, axis: str = 'x'):

    elementList = ops.getEleTags()

    bendMomentZ = []
    shearForceZ = []
    bendMomentY = []
    shearForceY = []
    axialForce = []
    torsion = []

    numEle = len(elementList)
    count = 1

    # Recorrer los elementos y obtener las fuerzas globales
    for ele in elementList:

        fuerza_local = ops.eleResponse(ele, 'localForce')
        ndI, ndJ = ops.eleNodes(int(ele))
        ix, iy = ops.nodeCoord(ndI)
        jx, jy = ops.nodeCoord(ndJ)

        match axis:
            case 'x':
                i = ix
                j = jx

        if count != numEle:

            if j > i:  ## Esta tomando las fuerzas del nodo j
                # mx = 5
                # vx = 1
                # my = 4
                vy = 1
                p = 3  ## TODO: Se cambia el indice de la carga axial para que coincida con el signo de las fuerzas  en el diagrama
                t = 2




            bendMomentZ.append(-10)  ## TODO: Este -1 es una improvisacion ya que no estoy seguro por que funciona para fuerzas aplicadas en X o Y en elementos Viga
            shearForceY.append(-10)

            bendMomentY.append(-10)

            shearForceZ.append(fuerza_local[vy])

            axialForce.append(fuerza_local[p])
            torsion.append(-1 * fuerza_local[
                t])  ## TODO: Este -1 es una improvisacion ya que lo coloco para que la torsion corresponda en pilotes

            count += 1

        else:
            if j > i:
                # mx = 5
                # vx = 1
                # my = 4
                vy = 1
                p = 3  ## TODO: Se cambia el indice de la carga axial para que coincida con el signo de las fuerzas  en el diagrama
                t = 2

                bendMomentZ.append(-10)  ## TODO: Este -1 es una improvisacion ya que no estoy seguro por que funciona para fuerzas aplicadas en X o Y en elementos Viga
                shearForceY.append(-10)

                bendMomentY.append(-10)
                shearForceZ.append(fuerza_local[vy])

                axialForce.append(fuerza_local[p])
                torsion.append(-1 * fuerza_local[
                    t])  ## TODO: Este -1 es una improvisacion ya que lo coloco para que la torsion corresponda en pilotes

                # mx = 5
                # vx = 1
                # my = 4
                vy = 4
                p = 0  ## TODO: Se cambia el indice de la carga axial para que coincida con el signo de las fuerzas  en el diagrama
                t = 5

                bendMomentZ.append(-10)  ## TODO: Este -1 es una improvisacion ya que no estoy seguro por que funciona para fuerzas aplicadas en X o Y en elementos Viga
                shearForceY.append(-10)

                bendMomentY.append(-10)
                shearForceZ.append(fuerza_local[vy])

                axialForce.append(fuerza_local[p])
                torsion.append(-1 * fuerza_local[
                    t])  ## TODO: Este -1 es una improvisacion ya que lo coloco para que la torsion corresponda en pilotes



    bendMomentY[-1] = -1 * bendMomentY[-1]
    shearForceZ[-1] = -1 * shearForceZ[-1]
    bendMomentZ[-1] = -1 * bendMomentZ[-1]
    shearForceY[-1] = -1 * shearForceY[-1]
    axialForce[-1] = -1 * axialForce[-1]
    torsion[-1] = -1 * torsion[
        -1]  ##TODO: Estas lineas de codigo son un remiendo a la logica, revisar correctamente esto debido a que parece arreglar los diagramas de torsion pero danar los de cortante y flexion

    return {
        'bendMomentZ': bendMomentZ,
        'shearForceZ': shearForceZ,
        'bendMomentY': bendMomentY,
        'shearForceY': shearForceY,
        'axialForce': axialForce,
        'torsion': torsion
    }

def plot_internal_forces(forces_dict: dict):
    import matplotlib.pyplot as plt

     # obtener coordenadas x de cada nodo
    node_list = ops.getNodeTags()
    Nodedisplacements = [0]*len(node_list)

    NodeCoordinates = [ops.nodeCoord(n)[0] for n in node_list]
    shearForceZ = forces_dict['shearForceZ']
    bendMomentY = forces_dict['bendMomentY']
    shearForceY = forces_dict['shearForceY']
    bendMomentZ = forces_dict['bendMomentZ']
    axialForce = forces_dict['axialForce']
    torsion = forces_dict['torsion']

    fig, ax = plt.subplots(2)

    ## Shear Force Z plot
    plot_beamSubplot(ax[0], shearForceZ, NodeCoordinates,
                          'Vz')  ##TODO: Corregir el hecho que el grafico de desplazamientos cuando el elemento es una viga secundaria el desplazamiento dice en y pero se plotea el desplazamiento en X
    # ## Bending moment Y plot
    # plot_beamSubplot(ax[0, 1], bendMomentY, NodeCoordinates, 'My')
    # ## Shear Force Y plot
    # plot_beamSubplot(ax[1, 0], shearForceY, NodeCoordinates, 'Vy')
    # ## Bending moment Z plot
    # plot_beamSubplot(ax[1, 1], bendMomentZ, NodeCoordinates, 'Mz')

    # # Ajustar el espaciado entre subplots
    # plt.tight_layout()
    #
    # ## Grafica de Desplazamientos, Fuerza Axial y Torsion
    #
    # fig, ax = plt.subplots(2, 2)

    # ## Displacement plot Z direction
    # plot_beamSubplot(ax[0, 0], Nodedisplacements, NodeCoordinates, 'dz')
    # ## Displacement plot Y direction
    # plot_beamSubplot(ax[0, 1], Nodedisplacements, NodeCoordinates, 'dy')
    # ## Axial Force X direction
    # plot_beamSubplot(ax[1, 0], axialForce, NodeCoordinates, 'P')
    ## Torsion Moment plot X direction
    plot_beamSubplot(ax[1], torsion, NodeCoordinates, 'T')

    # Ajustar el espaciado entre subplots
    plt.tight_layout()

    # Mostrar la gráfica
    plt.show()

def plot_beamSubplot(ax: 'Axes',
                     internalForce: list[float],
                     NodeCoordinates: list[float],
                     typeForce: str):

    maxPoint = max(internalForce)
    minPoint = min(internalForce)

    id_max = internalForce.index(maxPoint)
    id_min = internalForce.index(minPoint)

    limite = 3e-11

    internalForce = [0 if abs(num) < limite else num for num in internalForce]

    NodeCoordinatesX = [0] * (
        len(internalForce))  # Considerando que NodeCoordinatesX es de la misma longitud que internalForce

    ax.plot(NodeCoordinates, NodeCoordinatesX, color="r")
    ax.plot(NodeCoordinates, internalForce, color="b")

    match typeForce:
        case 'P':
            ytitle = 'Axial Force X kN'
            forceTitle = "{:5.2f} kN"
        case 'T':
            ytitle = 'Torsion Force X kN-m'
            forceTitle = "{:5.2f} kN-m"
        case 'Vz':
            ytitle = 'Shear Force Z kN'
            forceTitle = "{:5.2f} kN"
        case 'Vy':
            ytitle = 'Shear Force Y kN'
            forceTitle = "{:5.2f} kN"
        case 'Mz':
            ytitle = 'Bending Moment Z kN-m'
            forceTitle = "{:5.2f} kN-m"
        case 'My':
            ytitle = 'Bending Moment Y kN-m'
            forceTitle = "{:5.2f} kN-m"
        case 'dz':
            ytitle = 'Displacement Z mm'
            forceTitle = '{:5.2f} mm'
        case 'dy':
            ytitle = 'Displacement Y mm'
            forceTitle = '{:5.2f} mm'
        case 'basicPx':
            ytitle = 'Basic Axial Force X kN'
            forceTitle = '{:5.2f} kN'
        case 'basicTx':
            ytitle = 'Basic Torsion Moment X kN-m'
            forceTitle = '{:5.2f} kN-m'
        case 'basicMz':
            ytitle = 'Basic Bending Moment Z kN-m'
            forceTitle = '{:5.2f} kN-m'
        case 'basicMy':
            ytitle = 'Basic Bending Moment Y kN-m'
            forceTitle = '{:5.2f} kN-m'
        case 'basicex':
            ytitle = 'Basic Axial Strain X m/m'
            forceTitle = '{:8.3g} m/m'
        case 'basickx':
            ytitle = 'Basic Curvature X 1/m'
            forceTitle = '{:8.3g} 1/m'
        case 'basickz':
            ytitle = 'Basic Curvature Z 1/m'
            forceTitle = '{:8.3g} 1/m'
        case 'basicky':
            ytitle = 'Basic Curvature Y 1/m'
            forceTitle = '{:8.3g} 1/m'

    xtitle = 'Beam Length m'

    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    ax.text(NodeCoordinates[id_min], minPoint, forceTitle.format(minPoint), color='red')
    ax.text(NodeCoordinates[id_max], maxPoint, forceTitle.format(maxPoint), color='red')

    ax.minorticks_on()
    ax.grid(True)
    ax.grid(True, which='minor', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

# # --- Example usage ---
if __name__ == "__main__":
    # Define properties

    properties = {
        'span_lengths': [40*0.8, 40.0, 40*0.8],  # lengths of each span in meters
        'nodes_per_span': 64,        # number of nodes per span
        'support_types': ['second-class', 'pinned', 'pinned', 'pinned'],
        'E': 25000000,
        'A': 0.6,
        'I': 0.05,
    }



    # ops.load(6, 0.0, -10000.0, 0.0)  # Apply a point load of 10 kN at node 6
    #
    # ok = run_static_analysis()
    #
    # # plot_internal_forces()
    #
    # nodeId = nearest_node(2.6)
    #
    create_vehicle(properties, 3, [200, 200, 200], [2.5 , 2.5], "Portugal")

    create_vehicle(properties, 3, [400, 600, 600], [9.75, 11.50], "Analisis")

    create_vehicle(properties, 3, [100, 100, 100], [4.0, 8.0], "Tandem")

    # ok = run_vehicle_load_analysis(properties, 2.5, "Tandem", direction='left')
    # update_internal_forces(properties, 'Tandem')

    # plot_internal_forces()

    run_analysis(properties)

    plot_two_vehicles_envelopes(properties, 'Portugal', 'Analisis')

    ratios = compute_ratios_between_vehicles(
        properties,
        ref_vehicle="Portugal",
        new_vehicle="Analisis"
    )

    plot_vehicle_ratio_lines(properties, 'Portugal', 'Analisis')

    global_fs = compute_global_fs(properties, 'Analisis')

    print(f'Global Factors of Safety for Analisis vs Portugal: FSV={global_fs["FSV"]:.3f}, FSM={global_fs["FSM"]:.3f}')

    # plot_internal_forces()

    ok = run_vehicle_load_analysis(properties, 16.0, "Tandem", direction='left')

    forces = getFrameForces(properties)

    plot_internal_forces(forces)



    print(f'Fin: Analysis completed with status: {ok}')

