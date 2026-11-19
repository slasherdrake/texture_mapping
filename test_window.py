"""Test if Polyscope can create a window at all."""
import polyscope as ps

print("Initializing Polyscope...")
ps.init()

print("Creating simple mesh...")
import numpy as np
vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
faces = np.array([[0, 1, 2]])
ps.register_surface_mesh("triangle", vertices, faces)

print("\nAbout to call ps.show()...")
print("IMPORTANT: Look for a window that appears!")
print("Window should appear in 2 seconds and auto-close after 3 seconds...")

import time
time.sleep(2)

# Show for 3 seconds then auto-close
ps.show(3)

print("\nWindow should have closed. Did you see it? (yes/no)")
