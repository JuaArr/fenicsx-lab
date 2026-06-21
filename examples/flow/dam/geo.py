import numpy as np
import gmsh
import sys
from pathlib import Path

try:
	outputflag_idx = sys.argv.index('-o')
	geom_file = sys.argv[outputflag_idx+1]
except:
	geom_file = 'geom.msh'

# --- Initialization ---
gmsh.initialize()
gmsh.model.add('dam')

# --- Parameter definition ---
lc1 = 1.0
lc2 = 0.25
L = 45
H = 20
ld = 25
m = 10
t1 = 3.34
t2 = 1.67
t3 = 1.67
t4 = 3.34
b1 = 1.67
b2 = 1.67
b3 = 1.67
b4 = 1.67

# --- Geometry definition
# Points
p1 = gmsh.model.geo.addPoint(0, 0, 0, lc1)
p2 = gmsh.model.geo.addPoint(L, 0, 0, lc1)
p3 = gmsh.model.geo.addPoint(L, H, 0, lc1)
p4 = gmsh.model.geo.addPoint(0, H, 0, lc1)
p5 = gmsh.model.geo.addPoint(m, H, 0, lc2)
p6 = gmsh.model.geo.addPoint(m, H-t1, 0, lc2)
p7 = gmsh.model.geo.addPoint(m+b1, H-t1, 0, lc2)
p8 = gmsh.model.geo.addPoint(m+b1+b2, H-t2, 0, lc2)
p9 = gmsh.model.geo.addPoint(m+ld-b3-b4, H-t3, 0, lc2)
p10 = gmsh.model.geo.addPoint(m+ld-b4, H-t4, 0, lc2)
p11 = gmsh.model.geo.addPoint(m+ld, H-t4, 0, lc2)
p12 = gmsh.model.geo.addPoint(m+ld, H, 0, lc2)

# Lines
l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p12)
l4 = gmsh.model.geo.addLine(p12, p11)
l5 = gmsh.model.geo.addLine(p11, p10)
l6 = gmsh.model.geo.addLine(p10, p9)
l7 = gmsh.model.geo.addLine(p9, p8)
l8 = gmsh.model.geo.addLine(p8, p7)
l9 = gmsh.model.geo.addLine(p7, p6)
l10 = gmsh.model.geo.addLine(p6, p5)
l11 = gmsh.model.geo.addLine(p5, p4)
l12 = gmsh.model.geo.addLine(p4, p1)

# Curve loop and surface
cl1 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12])
ps1 = gmsh.model.geo.addPlaneSurface([cl1])

# Model synchronization
gmsh.model.geo.synchronize()

# --- Physical groups ---
# Adding physical groups with the following tag dim-XX
gmsh.model.addPhysicalGroup(dim=1, tags=[l1], tag=101, name='impermeable_layer')
gmsh.model.addPhysicalGroup(dim=1, tags=[l2], tag=102, name='aquifer_outlet')
gmsh.model.addPhysicalGroup(dim=1, tags=[l12], tag=103, name='aquifer_inlet')
gmsh.model.addPhysicalGroup(dim=1, tags=[l3], tag=104, name='surface_right')
gmsh.model.addPhysicalGroup(dim=1, tags=[l11], tag=105, name='surface_left')
gmsh.model.addPhysicalGroup(dim=1, tags=[l4], tag=106, name='embeded_right')
gmsh.model.addPhysicalGroup(dim=1, tags=[l10], tag=107, name='embeded_left')
gmsh.model.addPhysicalGroup(dim=1, tags=[l10, l9, l8, l7, l6, l5, l4], tag=108, name='dam_base')
gmsh.model.addPhysicalGroup(dim=2, tags=[ps1], tag=201, name='soil')

# --- Mesh generation ---
gmsh.model.mesh.generate(2)

# --- Save mesh file ---
gmsh.write(geom_file)

# --- Additional manipulation for post-processing ---
needed_points = [p5, p6, p7, p8, p9, p10, p11, p12]
len_needed_points = len(needed_points)

# Get the coordinates of the points and collect it
needed_p_coords = np.empty(shape=(len_needed_points, 3))
for idx in range(len_needed_points):
	needed_p_coords[idx, :] = gmsh.model.getValue(0, needed_points[idx], [])

offset_coords = np.array([
	[-0.001, -0.001, 0.0],
	[-0.001, -0.001, 0.0],
	[0.001, -0.001, 0.0],
	[0.001, -0.001, 0.0],
	[-0.001, -0.001, 0.0],
	[-0.001, -0.001, 0.0],
	[0.001, -0.001, 0.0],
	[0.001, -0.001, 0.0]
])

needed_p_coords += offset_coords

# Retrieve required data
wd = Path(geom_file).parent
np.save(wd/'point_coords.npy', needed_p_coords)

# --- Visualization with GMSH ---
if '-nopopup' not in sys.argv:
	gmsh.fltk.run()

# --- Finalization ---
gmsh.finalize()