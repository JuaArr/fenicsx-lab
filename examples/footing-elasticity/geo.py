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
lc = 0.25
L = 10.0
H = 10.0
b = 1.0

# --- Geometry definition
# Points
p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
p2 = gmsh.model.geo.addPoint(b, 0, 0, lc)
p3 = gmsh.model.geo.addPoint(L, 0, 0, lc)
p4 = gmsh.model.geo.addPoint(L, H, 0, lc)
p5 = gmsh.model.geo.addPoint(b, H, 0, lc)
p6 = gmsh.model.geo.addPoint(0, H, 0, lc)

# Lines
l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p5)
l3 = gmsh.model.geo.addLine(p5, p6)
l4 = gmsh.model.geo.addLine(p6, p1)

l5 = gmsh.model.geo.addLine(p2, p3)
l6 = gmsh.model.geo.addLine(p3, p4)
l7 = gmsh.model.geo.addLine(p4, p5)

# Curve loop and surface
cl1 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
cl2 = gmsh.model.geo.addCurveLoop([l5, l6, l7, -l2])
ps1 = gmsh.model.geo.addPlaneSurface([cl1])
ps2 = gmsh.model.geo.addPlaneSurface([cl2])

# --- Transfinite properties ---
n1x = 10
n2x = 50
ny = 50
# Left rectangle
gmsh.model.geo.mesh.setTransfiniteCurve(l1, n1x)
gmsh.model.geo.mesh.setTransfiniteCurve(l2, ny)
gmsh.model.geo.mesh.setTransfiniteCurve(l3, n1x)
gmsh.model.geo.mesh.setTransfiniteCurve(l4, ny)
gmsh.model.geo.mesh.setTransfiniteSurface(ps1)

# Right rectangle
gmsh.model.geo.mesh.setTransfiniteCurve(l5, n2x)
gmsh.model.geo.mesh.setTransfiniteCurve(l6, ny)
gmsh.model.geo.mesh.setTransfiniteCurve(l7, n2x)
gmsh.model.geo.mesh.setTransfiniteSurface(ps2)

# --- Mesh element type ---
# Recombinar para cuadriláteros
gmsh.model.geo.mesh.setRecombine(2, ps1)
gmsh.model.geo.mesh.setRecombine(2, ps2)

# --- Physical groups ---
# Adding physical groups with the following tag dim-XX
gmsh.model.addPhysicalGroup(dim=1, tags=[l1, l5], tag=101, name='bottom')
gmsh.model.addPhysicalGroup(dim=1, tags=[l6], tag=102, name='right-side')
gmsh.model.addPhysicalGroup(dim=1, tags=[l4], tag=103, name='left-side')
gmsh.model.addPhysicalGroup(dim=1, tags=[l3], tag=104, name='footing')
gmsh.model.addPhysicalGroup(dim=2, tags=[ps1, ps2], tag=201, name='soil')

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