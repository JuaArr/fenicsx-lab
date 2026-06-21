import gmsh
import sys


try:
	outputflag_idx = sys.argv.index('-o')
	geom_file = sys.argv[outputflag_idx+1]
except:
	geom_file = 'geom.msh'

# --- Initialization ---
gmsh.initialize()
gmsh.model.add('soil')

# --- Parameter definition ---
lc = 0.1
B = 2.0
H = 3.5

# --- Geometry definition ---
# Points
p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
p2 = gmsh.model.geo.addPoint(B, 0, 0, lc)
p3 = gmsh.model.geo.addPoint(B, H, 0, lc)
p4 = gmsh.model.geo.addPoint(0, H, 0, lc)

# Lines
l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p1)

# Curve loop and surface
cl1 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
ps1 = gmsh.model.geo.addPlaneSurface([cl1])

# --- Transfinite properties ---
nx = 2
ny = 6

# Adding transfinite curves and surface
gmsh.model.geo.mesh.setTransfiniteCurve(l1, nx)
gmsh.model.geo.mesh.setTransfiniteCurve(l2, ny)
gmsh.model.geo.mesh.setTransfiniteCurve(l3, nx)
gmsh.model.geo.mesh.setTransfiniteCurve(l4, ny)
gmsh.model.geo.mesh.setTransfiniteSurface(ps1)

# --- Mesh element type ---
# Recombinar para cuadriláteros
gmsh.model.geo.mesh.setRecombine(2, ps1)

# --- Physical groups ---
# Adding physical groups with the following tag dim-XX
gmsh.model.addPhysicalGroup(dim=1, tags=[l1], tag=101, name='bottom')
gmsh.model.addPhysicalGroup(dim=1, tags=[l2], tag=102, name='right-side')
gmsh.model.addPhysicalGroup(dim=1, tags=[l4], tag=103, name='left-side')
gmsh.model.addPhysicalGroup(dim=1, tags=[l3], tag=104, name='top')
gmsh.model.addPhysicalGroup(dim=2, tags=[ps1], tag=201, name='soil')

# Model synchronization
gmsh.model.geo.synchronize()

# --- Mesh generation ---
gmsh.model.mesh.generate(2)

# --- Save mesh file ---
gmsh.write(geom_file)

# --- Visualization with GMSH ---
if '-nopopup' not in sys.argv:
	gmsh.fltk.run()

# --- Finalization ---
gmsh.finalize()