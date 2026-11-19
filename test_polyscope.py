"""Minimal Polyscope test to check if GUI is working."""
import polyscope as ps
import numpy as np

# Initialize
ps.init()

# Create a simple cube
vertices = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
])

faces = np.array([
    [0, 1, 2], [0, 2, 3],  # bottom
    [4, 5, 6], [4, 6, 7],  # top
    [0, 1, 5], [0, 5, 4],  # front
    [2, 3, 7], [2, 7, 6],  # back
    [0, 3, 7], [0, 7, 4],  # left
    [1, 2, 6], [1, 6, 5]   # right
])

# Register mesh
ps.register_surface_mesh("cube", vertices, faces)

print("Window should be visible now. Press Ctrl+C to exit.")

# Show
ps.show()
