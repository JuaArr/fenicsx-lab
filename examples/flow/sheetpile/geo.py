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
gmsh.model.add('sheetpile')

# --- Parameter definition ---
lc1 = 0.5
lc2 = lc1/4
B = 20
H = 10
t = 1e-4
L = 3

# --- Geometry definition
# Points
p1 = gmsh.model.geo.addPoint(-B, -H, 0, lc1)
p2 = gmsh.model.geo.addPoint(B, -H, 0, lc1)
p3 = gmsh.model.geo.addPoint(B, 0, 0, lc1)
p4 = gmsh.model.geo.addPoint(t/2, 0, 0, lc2)
p5 = gmsh.model.geo.addPoint(t/2, -L, 0, lc2)
p6 = gmsh.model.geo.addPoint(-t/2, -L, 0, lc2)
p7 = gmsh.model.geo.addPoint(-t/2, 0, 0, lc2)
p8 = gmsh.model.geo.addPoint(-B, 0, 0, lc1)

# Lines
l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p5)
l5 = gmsh.model.geo.addLine(p5, p6)
l6 = gmsh.model.geo.addLine(p6, p7)
l7 = gmsh.model.geo.addLine(p7, p8)
l8 = gmsh.model.geo.addLine(p8, p1)

# Curve loop and surface
cl1 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4, l5, l6, l7, l8])
ps1 = gmsh.model.geo.addPlaneSurface([cl1])

# Model synchronization
gmsh.model.geo.synchronize()

# --- Physical groups ---
# Adding physical groups with the following tag dim-XX
gmsh.model.addPhysicalGroup(dim=1, tags=[l1], tag=101, name='impermeable_layer')
gmsh.model.addPhysicalGroup(dim=1, tags=[l3], tag=102, name='surface_right')
gmsh.model.addPhysicalGroup(dim=1, tags=[l7], tag=103, name='surface_left')
gmsh.model.addPhysicalGroup(dim=1, tags=[l4], tag=104, name='sheetpile_right')
gmsh.model.addPhysicalGroup(dim=1, tags=[l5], tag=105, name='sheetpile_bottom')
gmsh.model.addPhysicalGroup(dim=1, tags=[l6], tag=106, name='sheetpile_left')
gmsh.model.addPhysicalGroup(dim=2, tags=[ps1], tag=201, name='soil')

# --- Mesh generation ---
gmsh.model.mesh.generate(2)

# --- Save mesh file ---
gmsh.write(geom_file)

# --- Visualization with GMSH ---
if '-nopopup' not in sys.argv:
	gmsh.fltk.run()

# --- Finalization ---
gmsh.finalize()