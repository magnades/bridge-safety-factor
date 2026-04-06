import sectionproperties.pre.library.primitive_sections as sections
from sectionproperties.analysis.section import Section


import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from sectionproperties.pre.geometry import Geometry
from sectionproperties.analysis.section import Section


def build_beam_slab_section(
    slab_width,
    slab_thickness,
    beam_width,
    beam_height,
    num_beams,
    overhang
):
    """
    Creates the geometry of a slab supported by multiple rectangular beams.

    Parameters
    ----------
    slab_width : float
        Total slab width
    slab_thickness : float
        Slab thickness
    beam_width : float
        Beam width
    beam_height : float
        Beam height
    num_beams : int
        Number of beams
    overhang : float
        Distance from the first/last beam center to the slab edge

    Returns
    -------
    Geometry
        Combined slab + beams geometry
    """


    beam_spacing = (slab_width - 2 * overhang) / (num_beams - 1)

    # Create slab geometry
    slab = sections.rectangular_section(b=slab_width, d=slab_thickness)

    # Move slab so that the global section is centered at X = 0
    # and the slab sits on top of the beams
    slab = slab.shift_section(
        x_offset=-slab_width / 2,
        y_offset=beam_height
    )

    # Initialize the total geometry with the slab
    total_geometry = slab

    # Compute position of the first beam center so the system is centered
    first_beam_center = -(num_beams - 1) * beam_spacing / 2

    # Create each beam
    for i in range(num_beams):

        # Compute beam center position
        beam_center_x = first_beam_center + i * beam_spacing

        # Create rectangular beam
        beam = sections.rectangular_section(b=beam_width, d=beam_height)

        # Move beam to the correct position
        beam = beam.shift_section(
            x_offset=beam_center_x - beam_width / 2,
            y_offset=0
        )

        # Add beam to the total geometry
        total_geometry += beam

    return total_geometry

def create_parametric_bridge_section(
        total_height=1.30,
        top_width=5.6,
        bottom_width=1.8,
        top_inner_width=2.8,
        flange_thickness=0.25,
        hole_top_clearance=0.15,
        hole_bottom_clearance=0.25,
        mesh_size=0.005
):
    """
    Generates a parametric bridge cross-section with a circular void.
    """

    # Calculate derived parameters for the circular void
    hole_diameter = total_height - hole_top_clearance - hole_bottom_clearance
    hole_radius = hole_diameter / 2.0
    hole_center_x = 0.0
    hole_center_y = hole_bottom_clearance + hole_radius

    # Define outer boundary points (x, y) based on input parameters.
    # The origin (0, 0) is placed at the center of the bottom edge.
    outer_points = [
        (-bottom_width / 2.0, 0.0),  # Bottom-left corner
        (bottom_width / 2.0, 0.0),  # Bottom-right corner
        (top_inner_width / 2.0, total_height - flange_thickness),  # Inner corner right
        (top_width / 2.0, total_height - flange_thickness),  # Overhang bottom-right
        (top_width / 2.0, total_height),  # Top-right corner
        (-top_width / 2.0, total_height),  # Top-left corner
        (-top_width / 2.0, total_height - flange_thickness),  # Overhang bottom-left
        (-top_inner_width / 2.0, total_height - flange_thickness)  # Inner corner left
    ]

    # Create the outer polygon using Shapely
    outer_polygon = Polygon(outer_points)

    # Create the hole polygon (quad_segs=64 for a smooth circle)
    hole_polygon = Point(hole_center_x, hole_center_y).buffer(hole_radius, quad_segs=64)

    # Subtract the hole from the outer shape
    cross_section_polygon = outer_polygon.difference(hole_polygon)

    # Convert to sectionproperties Geometry
    geometry = Geometry(cross_section_polygon)

    # Generate the finite element mesh
    geometry.create_mesh(mesh_sizes=[mesh_size])

    # Return the ready-to-analyze Section object
    return Section(geometry)


def plot_triangles_progressively(
    section_to_plot: Section,
    pause_time: float = 0.05,
    title: str = "Mesh Triangles (Sequential Plot)",
    **kwargs,
):
    """
    Plot mesh triangles sequentially.

    Parameters
    ----------
    pause_time : float
        Time delay between drawing triangles.
    title : str
        Plot title.
    """

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(**kwargs)

    ax.set_title(title)
    ax.set_aspect("equal")

    # Extract mesh information
    nodes = section_to_plot.mesh_nodes
    triangles = section_to_plot.mesh_elements[:, 0:3]

    # Loop over each triangle
    for i, tri_nodes in enumerate(triangles):

        n1, n2, n3 = tri_nodes

        # Get node coordinates
        x = [
            nodes[n1, 0],
            nodes[n2, 0],
            nodes[n3, 0],
            nodes[n1, 0],  # close polygon
        ]

        y = [
            nodes[n1, 1],
            nodes[n2, 1],
            nodes[n3, 1],
            nodes[n1, 1],
        ]

        # Draw triangle
        ax.plot(x, y, color="black")
        ax.fill(x, y, alpha=0.3)

        # Optional: label element number
        cx = (x[0] + x[1] + x[2]) / 3
        cy = (y[0] + y[1] + y[2]) / 3
        ax.text(cx, cy, str(i), fontsize=7)

        plt.pause(pause_time)

    plt.show()

    return ax

def animate_mesh_triangles(self, interval: int = 50):
    """
    Animate mesh triangles sequentially using matplotlib.animation.

    Parameters
    ----------
    interval : int
        Time between frames in milliseconds.
    """
    import matplotlib
    # Fuerza a Matplotlib a abrir una ventana interactiva fuera de PyCharm.
    # Si tienes PyQt5 instalado, puedes probar con 'Qt5Agg' en lugar de 'TkAgg'.
    matplotlib.use('TkAgg')

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    nodes = self.mesh_nodes
    triangles = self.mesh_elements[:, 0:3]

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_title("Mesh Triangles Animation")

    # Calcular y establecer los límites de los ejes para que la gráfica no "salte"
    ax.set_xlim(nodes[:, 0].min(), nodes[:, 0].max())
    ax.set_ylim(nodes[:, 1].min(), nodes[:, 1].max())

    def update(frame):
        # Obtener los índices de los nodos para el triángulo actual
        n1, n2, n3 = triangles[frame]

        # No es necesario repetir el primer nodo, ax.fill cierra la figura
        x = [nodes[n1, 0], nodes[n2, 0], nodes[n3, 0]]
        y = [nodes[n1, 1], nodes[n2, 1], nodes[n3, 1]]

        # ax.fill devuelve una lista de polígonos (patches)
        poly = ax.fill(x, y, facecolor="blue", edgecolor="black", alpha=0.4)

        return poly

    # Es VITAL asignar FuncAnimation a una variable (anim)
    # para que Python no elimine la animación de la memoria.
    anim = FuncAnimation(
        fig,
        update,
        frames=len(triangles),
        interval=interval,
        repeat=False,
        blit=False  # Ponlo en False; blit=True a veces causa fallos visuales en PyCharm
    )

    # El block=True pausa la ejecución del script hasta que cierres la ventana
    plt.show(block=True)

    return anim


# ==========================================
# DESIGN PARAMETERS (example in meters)
# ==========================================

beam_width = 0.50
beam_height = 1.00

slab_thickness = 0.25

num_beams = 2
# beam_spacing = 1.20
overhang = 1.2

# Compute slab width automatically
slab_width = 3.0*4


# Create geometry
geometry = build_beam_slab_section(
    slab_width,
    slab_thickness,
    beam_width,
    beam_height,
    num_beams,
    # beam_spacing,
    overhang
)

# Generate finite element mesh
geometry.create_mesh(mesh_sizes=[0.01])

# Create section object
section = Section(geometry)

# Compute geometric properties
section.calculate_geometric_properties()

# Plot mesh
section.plot_mesh()

# Display section properties
section.display_results()


# plot_triangles_progressively(section)


animate_mesh_triangles(section)

### Segunda Seccion Transversal




# --- Example Usage ---

if __name__ == "__main__":
    # 1. Analyze the default section (the exact dimensions from your image)
    print("Analyzing the original section...")
    original_section = create_parametric_bridge_section()
    original_section.calculate_geometric_properties()
    original_section.plot_mesh()
    original_section.display_results()

    # 2. Create and analyze a NEW section with modified parameters
    # Let's make it taller, wider, and with a thicker top flange
    print("Analyzing a modified, larger section...")
    modified_section = create_parametric_bridge_section(
        total_height=1.50,  # Increased from 1.05
        top_width=7.0,  # Increased from 5.6
        bottom_width=2.2,  # Increased from 1.8
        top_inner_width=3.2,  # Increased from 2.8
        flange_thickness=0.35,  # Thicker flange
        hole_top_clearance=0.20,  # More concrete above the hole
        hole_bottom_clearance=0.30,  # More concrete below the hole
        mesh_size=0.01  # Slightly coarser mesh for speed
    )
    modified_section.calculate_geometric_properties()
    modified_section.plot_mesh()
    modified_section.display_results()

print("End of Analysis")



