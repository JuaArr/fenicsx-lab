import sys
import subprocess
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import pyvista as pv

from mpi4py import MPI
from petsc4py import PETSc
from basix import ufl as bufl
import dolfinx as dlx
from dolfinx.fem.petsc import LinearProblem
from ufl import (FacetNormal, Measure, TestFunction, TrialFunction, ds, dx, inner, lhs, grad, rhs, dot, SpatialCoordinate)

from ...utils import required_font_size, required_img_scale, export_boundaries

# --- Creation of required directories ---
# current working directory
wd = Path(__file__).parent
# export
ed = wd/'export'
ed.mkdir(exist_ok=True, parents=True)
# logs
logd = wd/'logs'
logd.mkdir(exist_ok=True, parents=True)

# --- MPI communicator ---
comm = MPI.COMM_SELF
rank = comm.Get_rank()

# --- Call to the GMSH-API integration and catching of the stdout and stderr ---
geom_file = wd/'sheetpile.msh'
with open(logd/'geom.log', 'w') as log_file:
	log = subprocess.run(
		['python', wd/'geo.py', '-nopopup',  '-o', geom_file],
		stdout=log_file,
		stderr=log_file,
		text=True
	)

# --- Importing mesh and boundary markers ---
dim = 2
mesh, _, ft = dlx.io.gmshio.read_from_msh(filename=geom_file, comm=comm, rank=rank, gdim=dim)
ds = Measure(integral_type="ds", domain=mesh, subdomain_data=ft)

# --- Definition of parameter values ---
K = dlx.fem.Constant(mesh, PETSc.ScalarType(2e-5)) # K = 2x10^-5 m/s
p_ls = dlx.fem.Constant(mesh, PETSc.ScalarType(12.0)) # p = 10 m
p_rs = dlx.fem.Constant(mesh, PETSc.ScalarType(15.0)) # p = 0 m
g = dlx.fem.Constant(mesh, PETSc.ScalarType(0.0)) # u = 0 m/s
gammaw = dlx.fem.Constant(mesh, PETSc.ScalarType(9.81)) # gammaw = 9.81 kN/m3
depth = dlx.fem.Constant(mesh, PETSc.ScalarType(10.0)) # H = 10 m
n = FacetNormal(mesh) # normal vector of the boundaries

# --- Funtion space creation (for pressure) ---
s_cg1 = bufl.element(family="P", cell=mesh.topology.cell_name(), degree=1) # s_cg1 = scalar continuos galerkin, degree 1
Q = dlx.fem.functionspace(mesh=mesh, element=s_cg1)
p, q = TrialFunction(Q), TestFunction(Q)
F = K*inner(grad(p), grad(q))*dx 

# --- Define boundary conditions (Neumann == rates) ---
F += inner(g, q)*ds(101) # Impermeable layer
F += inner(g, q)*ds(104) # Sheetpile right
F += inner(g, q)*ds(105) # Sheetpile bottom
F += inner(g, q)*ds(106) # Sheetpile left

# --- Define boundary conditions (Dirichlet == specific value) ---
fdim = dim-1
# Surface left (behind the structure)
bcp_ls = dlx.fem.dirichletbc(p_ls, dlx.fem.locate_dofs_topological(Q, fdim, ft.find(103)), Q)
# Surface right (in front of the structure)
bcp_rs = dlx.fem.dirichletbc(p_rs, dlx.fem.locate_dofs_topological(Q, fdim, ft.find(102)), Q)
# Summarize
bcp = [bcp_ls, bcp_rs]

# --- Solve linear variational problem ---
a = lhs(F)
L = rhs(F)
problem = LinearProblem(a, L, bcs=bcp, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
H = problem.solve()
H.name = 'total head'

# --- Find the pore pressure ---
# u_w = h_p * gamma_w = (H - h_z) * gamma_w

X = SpatialCoordinate(mesh)
y = X[1] 
uw_expression = (H - (depth + y)) * gammaw # y-values are negative (see geometry)
expr = dlx.fem.Expression(uw_expression, Q.element.interpolation_points())

uw = dlx.fem.Function(Q)
uw.interpolate(expr)
uw.name = 'pore pressure'

# --- Extracting the flow velocity ---
v_cg2 = bufl.element("P", mesh.topology.cell_name(), 2, shape=(dim, )) # v_cg2 = vector continuos galerkin, degree 2
V = dlx.fem.functionspace(mesh, v_cg2)

velocity_expr = -K*grad(H)
v, w = TestFunction(V), TrialFunction(V)
a = inner(w, v)*dx
L = inner(velocity_expr, v)*dx

problem = LinearProblem(a, L, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u = problem.solve()
u.name = 'flow velocity'

# --- Saving data to visualize using ParaView ---
with dlx.io.VTXWriter(comm, ed/'results_energy.bp', [H], engine='BP4') as vtx:
	vtx.write(0.0)
	
with dlx.io.VTXWriter(comm, ed/'results_pore_pressure.bp', [uw], engine='BP4') as vtx:
	vtx.write(0.0)

with dlx.io.VTXWriter(comm, ed/'results_velocity.bp', [u], engine='BP4') as vtx:
	vtx.write(0.0)
	
export_boundaries(ed, mesh, ft)

# --- Flow --- ???

# flux_form = uh[0]*n[0]*ds(105) + uh[1]*n[1]*ds(105)

# Q_boundary = dlx.fem.assemble_scalar(dlx.fem.form(flux_form))

# flux_form2 = uh[0]*n[0]*ds(104) + uh[1]*n[1]*ds(104)

# Q_boundary2 = dlx.fem.assemble_scalar(dlx.fem.form(flux_form2))

# print(Q_boundary, Q_boundary2)

# --- Setting pyvista configuration ---
# if '-offscreen' in sys.argv:
# 	pv.OFF_SCREEN = True

# # --- Plotting with pyvista ---
# # Visualization and scaling
# target_width_cm = 18.0
# aspect_ratio = [4, 3]
# base_width_px = 512
# min_dpi = 600
# img_scale, new_dpi = required_img_scale(min_dpi, target_width_cm, base_width_px, aspect_ratio)
# window_size = [base_width_px*aspect_ratio[0], base_width_px*aspect_ratio[1]]
# label_size, title_size = required_font_size(10, target_width_cm, aspect_ratio, window_size)

# # Create the pyvista-grid for the mesh
# mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
# grid = pv.UnstructuredGrid(*dlx.plot.vtk_mesh(mesh, mesh.topology.dim))

# # Create the pressure-scalar-field
# topology, cell_types, geometry = dlx.plot.vtk_mesh(Q)
# pressure_grid = pv.UnstructuredGrid(topology, cell_types, geometry)
# pressure_grid.point_data['ph'] = ph.x.array
# sargs = dict(
# 	title='pore pressure\n',
# 	vertical=False,
# 	height=0.1,
# 	width=0.4,
# 	position_x=0.3,
# 	position_y=0.05,
# 	unconstrained_font_size=True,
# 	title_font_size=title_size,
# 	label_font_size=label_size,
# 	shadow=True,
# 	n_labels=5,
# 	fmt="%.1f",
# 	font_family="times"
# )

# # Create contours
# mi, ma = round(np.min(pressure_grid['ph']), ndigits=1), round(np.max(pressure_grid['ph']), ndigits=1)
# st = 1.0
# cntrs = np.arange(mi, ma+st, st)
# contours = pressure_grid.contour(isosurfaces=cntrs, scalars='ph')
# contours.points[:, -1] += 0.001 # Correction for visualization

# # Create the velocity-vector-field
# topology, cell_types, geometry = dlx.plot.vtk_mesh(V)
# v_values = np.empty((geometry.shape[0], 3), dtype=np.float64)
# v_values[:, :len(uh)] = uh.x.array.real.reshape((geometry.shape[0], len(uh)))

# velocity_vf = pv.UnstructuredGrid(topology, cell_types, geometry)
# velocity_vf['u'] = v_values
# glyphs = velocity_vf.glyph(orient='u', scale=False, factor=1.0, tolerance=0.015)

# # Create plotter
# plotter = pv.Plotter(window_size=window_size, image_scale=img_scale)
# plotter.add_mesh(grid, style='wireframe', color='gray', opacity=0.3)
# plotter.add_mesh(pressure_grid, show_scalar_bar=True, scalars='ph', clim=[mi, ma], scalar_bar_args=sargs, opacity=0.7)
# plotter.add_mesh(contours, color='black', line_width=4, opacity=1.0)
# plotter.add_mesh(glyphs, color='black', show_scalar_bar=False, opacity=1.0)
# plotter.view_xy()
# plotter.zoom_camera(1.4)

# if not pv.OFF_SCREEN:
# 	plotter.show()
# else:
# 	pv.start_xvfb()
# 	plotter.save_graphic(ed/'flow_net.svg')
# 	print('The image has been saved!')