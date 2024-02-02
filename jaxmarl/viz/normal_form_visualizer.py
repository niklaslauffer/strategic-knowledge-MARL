import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from matplotlib.markers import MarkerStyle

tetrahedron_vertices = jnp.array([
    [0, 0, 0],
    [1, 0, 0],
    [0.5, jnp.sqrt(3)/2, 0],
    [0.5, 1/3*jnp.sqrt(3), jnp.sqrt(2/3)]
])

triangle_vertices = jnp.array([
        [0, 0],
        [1, 0],
        [0.5, jnp.sqrt(3)/2]
])

def barycentric_to_cartesian(barycentric_coords, vertices):
    """
    Convert barycentric coordinates to Cartesian coordinates.
    """
    return jnp.dot(barycentric_coords, vertices)

def plot_tetrahedron(vertices):
    """
    Plot the tetrahedron defined by the vertices.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Define the tetrahedron faces
    faces = [
        [vertices[j] for j in [0, 1, 2]],
        [vertices[j] for j in [0, 1, 3]],
        [vertices[j] for j in [0, 2, 3]],
        [vertices[j] for j in [1, 2, 3]]
    ]
    
    # Plot the tetrahedron
    ax.add_collection3d(Poly3DCollection(faces, alpha=0.25, linewidths=1, edgecolors='r'))
    
    return ax

def plot_triangle(vertices):
    """
    Plot the triangle defined by the vertices.
    """
    fig, ax = plt.subplots()
    
    # Plot the triangle
    triangle = plt.Polygon(vertices, fill=None, edgecolor='black', linewidth=1)
    ax.add_patch(triangle)

    # Add labels to the vertices
    labels = ['0', '1', '2']
    for i, vertex in enumerate(vertices):
        pos = ['right', 'left', 'center']
        ax.text(vertex[0], vertex[1], labels[i], fontsize=20, ha=pos[i])

    # Turn off the axis
    ax.axis('off')
    
    # Remove the box/border
    ax.set_frame_on(False)
    
    return fig, ax

def point_generator(vertices, point_seq):
    for points in point_seq:
        barycentric_coords = points
        cartesian_coords = barycentric_to_cartesian(barycentric_coords, vertices)
        yield cartesian_coords

def update(cartesian_coords, sc, point_colors, point_markers):
    sc.set_facecolor(point_colors)
    marker_dict = {marker: MarkerStyle(marker).get_path().transformed(MarkerStyle(marker).get_transform()) for marker in point_markers}
    paths = [marker_dict[marker] for marker in point_markers]
    sc.set_paths(paths)
    sc.set_offsets(cartesian_coords)
    return sc,

def animate_triangle(point_seq, colors, markers, save_gif):
    fig, ax = plot_triangle(triangle_vertices)

    # Initial scatter plot
    sc = ax.scatter([], [])
    
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, jnp.sqrt(3)/2)
    ax.set_aspect('equal', 'box')
    
    gen = point_generator(triangle_vertices, point_seq)
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=gen, fargs=(sc,colors, markers), interval=100, blit=True)

    if save_gif:
        ani.save(save_gif, writer=PillowWriter(fps=20))
    
    plt.close()

