import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Try to import OpenSeesPy; if not available, provide a mock for interface so UI still runs
try:
    import openseespy.opensees as ops
    OSP_AVAILABLE = True
except Exception as e:
    OSP_AVAILABLE = False

    class _MockOPS:
        def __getattr__(self, name):
            def _no_op(*args, **kwargs):
                return None
            return _no_op

    ops = _MockOPS()

st.set_page_config(page_title="Beam model (OpenSees) — Streamlit", layout="wide")
st.title("Interactive beam model (OpenSeesPy)")
st.markdown(
    """
    Build a multi-span beam model, apply a train of point loads (leading load + following loads) and run a quasi-static analysis.

    Notes:
    * If OpenSeesPy is not installed this demo will still let you configure loads and see an estimated deflected shape using simple Euler-Bernoulli beam theory as a fallback.
    * "Second-class supports" is implemented here as a common simply-supported span: **left end pinned (Ux=Uy=0, Rz free) and right end roller (Uy=0 only)**. If you intended a different meaning, tell me and I will change it.
    """
)

# --- Sidebar: Model definition ---
st.sidebar.header("Model geometry and properties")
num_spans = st.sidebar.number_input("Number of spans (beams)", min_value=1, max_value=10, value=2)
span_lengths = []
with st.sidebar.form(key='spans_form'):
    st.write("Enter length (m) for each span:")
    cols = st.columns(2)
    for i in range(num_spans):
        length = cols[i % 2].number_input(f"Span {i+1} length (m)", min_value=0.1, value=5.0)
        span_lengths.append(length)
    E = st.number_input("Young's modulus E (Pa)", value=2.5e7, format="%.0f")
    A = st.number_input("Axial area A (m^2)", value=0.60)
    I = st.number_input("Moment of inertia I (m^4)", value=0.05)
    st.form_submit_button("Save geometry")

# total length and discretization
total_length = sum(span_lengths)
n_nodes_per_span = st.sidebar.slider("Nodes per span (mesh density)", min_value=2, max_value=50, value=11)
nodes_per_span = n_nodes_per_span

# --- Loads ---
st.sidebar.header("Moving point loads (vehicle)")
leading_load = st.sidebar.number_input("Leading load magnitude (kN)", value=100.0)
n_following = st.sidebar.number_input("Number of following loads (behind leading)", min_value=0, max_value=10, value=2)
follow_mag = st.sidebar.number_input("Following loads magnitude (kN)", value=50.0)
spacing = st.sidebar.number_input("Distance between successive vehicle axles (m)", value=2.0)

st.sidebar.markdown("---")
support_type = st.sidebar.selectbox("Support preset", options=["second-class (left pinned, right roller)", "both pinned", "both fixed", "left fixed, right roller"], index=0)

# Simulation control
st.sidebar.header("Simulation")
lead_pos = st.sidebar.slider("Leading load position along beam (m)", min_value=0.0, max_value=float(total_length), value=0.0, step=0.1)
run_analysis = st.sidebar.button("Run analysis / update plot")

# --- Build node coordinates ---
# Create node coordinates along the beam (1D line in X, y=0)
n_span = num_spans
coords = [0.0]
for L in span_lengths:
    # create nodes for this span excluding first point (already present)
    xs = np.linspace(coords[-1], coords[-1] + L, nodes_per_span)
    # skip the first because it's duplicated
    for x in xs[1:]:
        coords.append(float(x))

n_nodes = len(coords)
node_ids = list(range(1, n_nodes + 1))

# --- Helper: map support preset to fix flags ---
def support_flags(preset, is_left):
    # DOF order in 2D: ux, uy, rz
    if preset == "second-class (left pinned, right roller)":
        if is_left:
            return (1, 1, 0)  # pinned
        else:
            return (0, 1, 0)  # roller (allows horizontal movement)
    if preset == "both pinned":
        return (1, 1, 0)
    if preset == "both fixed":
        return (1, 1, 1)
    if preset == "left fixed, right roller":
        return (1, 1, 1) if is_left else (0, 1, 0)
    return (1, 1, 0)

# --- OpenSees model build and analysis functions ---

def build_opensees_model():
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    # create geometric transformation
    ops.geomTransf('Linear', 1)

    # create nodes
    for nid, x in zip(node_ids, coords):
        ops.node(nid, x, 0.0)

    # apply boundary conditions at span ends only (first and last node)
    left_flags = support_flags(support_type, True)
    right_flags = support_flags(support_type, False)
    ops.fix(node_ids[0], *left_flags)
    ops.fix(node_ids[-1], *right_flags)

    # create elements (one element between each pair of nodes)
    E_si = E
    A_si = A
    I_si = I
    for i in range(len(node_ids)-1):
        eid = i+1
        ops.element('elasticBeamColumn', eid, node_ids[i], node_ids[i+1], A_si, E_si, I_si, 1)

    # define load pattern (constant time series)
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)


def apply_point_loads(lead_pos_m):
    # clear existing loads by re-creating load pattern
    # In OpenSees, loads are added to the current pattern; easiest is to remove and recreate pattern id 1
    # But OpenSeesPy doesn't support removing a pattern easily; as a pragmatic approach, wipe loads by creating a new pattern id (2) and use it.
    ops.pattern('Plain', 2, 1)

    # positions of loads along the global beam axis
    positions = []
    # leading
    positions.append(lead_pos_m)
    # following loads behind leading (positive distance behind -> smaller x)
    for i in range(1, n_following+1):
        positions.append(lead_pos_m - i*spacing)

    # convert magnitudes to N
    mags = [leading_load*1e3] + [follow_mag*1e3]*n_following

    # for each load position, find the nearest node and apply vertical load (negative Y)
    for pos, mag in zip(positions, mags):
        if pos < 0 or pos > total_length:
            continue
        # find nearest node index
        idx = min(range(len(coords)), key=lambda j: abs(coords[j]-pos))
        nid = node_ids[idx]
        ops.load(nid, 0.0, -mag, 0.0)


def run_static_analysis():
    # prepare analysis
    ops.system('BandGeneral')
    ops.numberer('Plain')
    ops.constraints('Plain')
    ops.test('NormUnbalance', 1e-8, 10)
    ops.algorithm('Newton')
    ops.integrator('LoadControl', 1.0)
    ops.analysis('Static')
    # do one step (pattern 2 contains our loads)
    ok = ops.analyze(1)
    return ok

# --- Fallback beam theory analysis (if OpenSeesPy not available) ---
def euler_bernoulli_single_span_displacement(x_coords, loads):
    # Very simple superposition using influence functions for simply-supported beam spans.
    # This is only a crude estimate: assumes single-span simply-supported with total length = total_length
    L = total_length
    def disp_at_x(x):
        w = 0.0
        for px, P in loads:
            a = px
            b = L - a
            if x < a:
                # use formula for point load on simply supported beam
                w += P * x * b * (L - b - x) / (6*E*I*L)
            else:
                w += P * a * (L - x) * (L - a - (L - x)) / (6*E*I*L)
        return w
    return np.array([disp_at_x(x) for x in x_coords])

# --- UI: Show geometry and nodes ---
col1, col2 = st.columns([1,2])
with col1:
    st.subheader("Beam geometry")
    st.write(f"Spans: {num_spans}, total length = {total_length:.2f} m")
    st.write("Node count:", n_nodes)
    st.write("Support preset:", support_type)
    st.write("Leading load at x =", f"{lead_pos:.2f} m")
    st.write(f"Leading load = {leading_load} kN; {n_following} following loads, spacing = {spacing} m")

# small table of nodes
with col1:
    st.subheader("Nodes (x coord)")
    nodetable = {"node": node_ids, "x (m)": [round(c, 4) for c in coords]}
    st.dataframe(nodetable)

# --- Analysis / Plot ---
fig, ax = plt.subplots(figsize=(9,3))
ax.plot(coords, np.zeros_like(coords), '-k', label='undeformed')

if run_analysis and OSP_AVAILABLE:
    build_opensees_model()
    apply_point_loads(lead_pos)
    ok = run_static_analysis()
    if ok is None:
        st.warning("OpenSeesPy returned no status; results may be available anyway.")
    # read vertical displacements
    uys = [ops.nodeDisp(nid, 2) or 0.0 for nid in node_ids]
    ax.plot(coords, uys, label='OpenSeesPy deflection')
    ax.set_ylabel('Vertical displacement (m)')
    ax.set_xlabel('x (m)')
    ax.legend()
    st.pyplot(fig)

elif run_analysis and not OSP_AVAILABLE:
    # estimate using very simple approach: treat as single simply-supported span
    # compute point load positions
    positions = [lead_pos] + [lead_pos - i*spacing for i in range(1, n_following+1)]
    loads = [(p, -leading_load*1e3) if i==0 else (p, -follow_mag*1e3) for i,p in enumerate(positions)]
    # clamp positions
    loads = [(max(0,min(total_length,p)), P) for p,P in loads if 0<=p<=total_length]
    # compute simple displacements (crude)
    uys = euler_bernoulli_single_span_displacement(coords, loads)
    ax.plot(coords, uys, label='Estimated deflection (simple beam)')
    ax.set_ylabel('Vertical displacement (m)')
    ax.set_xlabel('x (m)')
    ax.legend()
    st.warning("OpenSeesPy not available — showing crude Euler-Bernoulli estimate instead.")
    st.pyplot(fig)

else:
    # show undeformed and mark loads
    # compute load positions for visualization
    positions = [lead_pos] + [lead_pos - i*spacing for i in range(1, n_following+1)]
    for i,p in enumerate(positions):
        if 0<=p<=total_length:
            ax.plot([p,p], [0, -0.2*max(span_lengths)], linewidth=2)
            ax.text(p, -0.25*max(span_lengths), f"L{i+1}", ha='center')
    ax.set_ylim(-0.3*max(span_lengths), 0.3*max(span_lengths))
    ax.set_xlabel('x (m)')
    ax.set_yticks([])
    st.pyplot(fig)

# --- Footer / help ---
st.markdown("---")
st.markdown(
    "**How to use**: adjust spans, mesh density, vehicle loads and move the 'Leading load position' slider. Click 'Run analysis' to run OpenSeesPy (if installed) or to see a simple estimate otherwise.\n\n"
    "**If you want:** I can add dynamic moving-load stepping (loop through positions and create an animation), different element types, or allow mixed support types per span."
)

if not OSP_AVAILABLE:
    st.info("OpenSeesPy not installed in this environment. To run full analyses install `openseespy` (pip install openseespy) and re-run this app.")
