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
from ufl import (FacetNormal, Measure, TestFunction, TrialFunction, ds, dx, inner, grad, sym, nabla_div, dot, Identity)


save_data = False

# --- Creation of required directories ---
# current working directory
wd = Path(__file__).parent
# export
ed = wd/'export'
ed.mkdir(exist_ok=True, parents=True)

# --- MPI communicator ---
comm = MPI.COMM_SELF
rank = comm.Get_rank()

geom_file = wd/'geom.msh'
# --- Importing mesh and boundary markers ---
dim = 2
mesh, _, ft = dlx.io.gmshio.read_from_msh(filename=geom_file, comm=comm, rank=rank, gdim=dim)
ds = Measure(integral_type="ds", domain=mesh, subdomain_data=ft)

# --- Definition of parameter values ---
E = dlx.fem.Constant(domain=mesh, c=PETSc.ScalarType(10e6)) # E = 200 MPa
nu = dlx.fem.Constant(domain=mesh, c=PETSc.ScalarType(0.3)) # nu = 0.3
mu = E/(2.0*(1.0+nu))
lambda_ = (E*nu)/((1.0+nu)*(1.0-2.0*nu))
p = dlx.fem.Constant(domain=mesh, c=PETSc.ScalarType(-500e3)) # p = 500 kPa
u_bottom = dlx.fem.Constant(domain=mesh, c=PETSc.ScalarType((0.0, 0.0)))
u_side = dlx.fem.Constant(domain=mesh, c=PETSc.ScalarType(0.0))
n = FacetNormal(mesh) # normal vector of the boundaries

# --- Funtion space creation (for pressure) ---
v_cg1 = bufl.element(family="Lagrange", cell=mesh.topology.cell_name(), degree=1, shape=(dim,))
V = dlx.fem.functionspace(mesh, v_cg1)
u, v = TrialFunction(V), TestFunction(V)

# --- Define boundary conditions (Dirichlet == specific value) ---
fdim = dim-1
# Bottom
bcu_b = dlx.fem.dirichletbc(u_bottom, dlx.fem.locate_dofs_topological(V, fdim, ft.find(101)), V)
# Right- and left-side
bcu_rs = dlx.fem.dirichletbc(u_side, dlx.fem.locate_dofs_topological(V.sub(0), fdim, ft.find(102)), V.sub(0))
bcu_ls = dlx.fem.dirichletbc(u_side, dlx.fem.locate_dofs_topological(V.sub(0), fdim, ft.find(103)), V.sub(0))
# Summarize
bcu = [bcu_b, bcu_rs, bcu_ls]

# --- Variational formulation ---
def epsilon(u):
    return sym(grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

def sigma(u):
    return lambda_*nabla_div(u)*Identity(dim) + 2.0*mu*epsilon(u)

rho = 17.5
g = 9.81
f = dlx.fem.Constant(mesh, dlx.default_scalar_type((0, -rho * g)))
a = inner(sigma(u), epsilon(v)) * dx
L = dot(f, v) * dx + dot(p*n, v) * ds(104)

problem = LinearProblem(a, L, bcs=bcu, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
uh.name = "displacement"

# --- Funtion space creation (for pressure) ---
t_cg2 = bufl.element(family="Lagrange", cell=mesh.topology.cell_name(), degree=2, shape=(dim, dim))
V_ = dlx.fem.functionspace(mesh, t_cg2)
u_, v_ = TrialFunction(V_), TestFunction(V_)

a = inner(u_, v_) * dx
L = inner(sigma(uh), v_) * dx

problem = LinearProblem(a, L, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
sh = problem.solve()
sh.name = "stress"

# --- Saving data to visualize using ParaView ---
if save_data:
    with dlx.io.VTXWriter(comm, ed/'results_displacement.bp', [uh], engine='BP4') as vtx:
        vtx.write(0.0)

    with dlx.io.VTXWriter(comm, ed/'results_stress.bp', [sh], engine='BP4') as vtx:
        vtx.write(0.0)