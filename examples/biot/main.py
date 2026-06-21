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
from ufl import (FacetNormal, Measure, TestFunction, TrialFunction, lhs, rhs, ds, dx, inner, grad, sym, tr, div, dot, Identity)

from utils import export_boundaries


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
n = FacetNormal(mesh)  # normal vector of the boundaries
# export_boundaries(ed, mesh, ft)

# --- Definition of parameter values ---
# Time integration
t_max = 100.0
n_steps = 100
dt = t_max / n_steps

# Elastic parameters
E = dlx.fem.Constant(domain=mesh, c=PETSc.ScalarType(18.15)) # E = 18.15 kg/cm2
nu = dlx.fem.Constant(domain=mesh, c=PETSc.ScalarType(0.33)) # nu = 0.33
mu = E/(2.0*(1.0+nu))
lmbda = (E*nu)/((1.0+nu)*(1.0-2.0*nu))

# Fluid parameters
k = dlx.fem.Constant(mesh, PETSc.ScalarType(6e-6)) # K = 6x10^-6 cm/min


# Coupling parameters
e = 1.0
phi = e/(e+1.0)
alpha = 1.0
K_f = 22433.8 # K_f = 2.2 GPa = 22433.8 kg/cm2
M = K_f/phi

# BC
load = dlx.fem.Constant(domain=mesh, c=PETSc.ScalarType(-4)) # p = 4 kg/cm2
u_bottom = dlx.fem.Constant(domain=mesh, c=PETSc.ScalarType((0.0, 0.0)))
u_side = dlx.fem.Constant(domain=mesh, c=PETSc.ScalarType(0.0))
p_top = dlx.fem.Constant(domain=mesh, c=PETSc.ScalarType(0.0))

# --- ELEMENT SPACES ---
v_cg1 = bufl.element(family="Lagrange", cell=mesh.topology.cell_name(), degree=1, shape=(dim,))
s_cg1 = bufl.element(family="Lagrange", cell=mesh.topology.cell_name(), degree=1)
TH = bufl.mixed_element([v_cg1, s_cg1]) # Taylor-Hood element

W = dlx.fem.functionspace(mesh, TH)
V, _ = W.sub(0).collapse()  # Subespacio desplazamiento
Q, _ = W.sub(1).collapse()  # Subespacio presión

# --- BOUNDARY CONDITIONS (DIRICHLET) ---
fdim = dim-1

# Displacement subspace 
bcu_b = dlx.fem.dirichletbc(u_bottom, dlx.fem.locate_dofs_topological(V, fdim, ft.find(101)), V)
bcu_rs = dlx.fem.dirichletbc(u_side, dlx.fem.locate_dofs_topological(V.sub(0), fdim, ft.find(102)), V.sub(0))
bcu_ls = dlx.fem.dirichletbc(u_side, dlx.fem.locate_dofs_topological(V.sub(0), fdim, ft.find(103)), V.sub(0))

bcu = [bcu_b, bcu_rs, bcu_ls]

# Pressure subspace
bcp_t = dlx.fem.dirichletbc(p_top, dlx.fem.locate_dofs_topological(Q, fdim, ft.find(104)), Q)

bcp = [bcp_t]

# --- Functions ---
# Previous step
u_n = dlx.fem.Function(V, name="displacement_n")
p_n = dlx.fem.Function(Q, name="pressure_n")

# Actual step
u = TrialFunction(V)
p = TrialFunction(Q)

# Test functions
v = TestFunction(V)
q = TestFunction(Q)

# --- Initial prssure field ---
p_n.x.array[:] = 0.0
u_n.assign()

# --- Crank-Nicholson ---
theta = 0.5
u_mid = theta * u + (1-theta) * u_n
p_mid_for_u = theta * p_n + (1-theta) * p_n

# --- Variational formulation ---
def epsilon(u):
    return sym(grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

def sigma(u):
    eps = epsilon(u)
    return lmbda*tr(eps)*Identity(dim) + 2.0*mu*eps

# Momentum
F_u = inner(sigma(u_mid), epsilon(v)) * dx
F_u -= alpha * p_mid_for_u * div(v) * dx
F_u -= dot(load * n, v) * ds(104)

a_u = lhs(F_u)
L_u = rhs(F_u)

# Flow
F_p =  alpha * (div(u) - div(u_n)) / dt * q * dx
F_p += (1/M) * (p - p_n) / dt * q * dx
F_p += k * inner(grad(p_mid), grad(q)) * dx

a_p = lhs(F_p)
L_p = rhs(F_p)

# --- Solvers ---
problem_u = LinearProblem(
    a_u, L_u, bcs=bcu,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)

# problem_p = LinearProblem(
#     a_p, L_p, bcs=bcp,
#     petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
# )

# --- Solution ---
# Crear escritores VTX separados para cada campo
vtx_u = dlx.io.VTXWriter(mesh.comm, ed/"biot_displacement.bp", [u_n], engine="BP4")
vtx_p = dlx.io.VTXWriter(mesh.comm, ed/"biot_pressure.bp", [p_n], engine="BP4")

# Loop temporal con iteración de punto fijo
t = 0.0
max_iter = 20
tol = 1e-8

print(f"{'Paso':<6} {'Tiempo':<10} {'Iter':<6} {'||Δu||':<12} {'||Δp||':<12} {'p_max':<12}")
print("-" * 68)

u_sol = dlx.fem.Function(V)
p_sol = dlx.fem.Function(Q)

for step in range(n_steps):
    t += dt

    # Iteración de punto fijo para acoplamiento
    u_sol.x.array[:] = u_n.x.array[:]
    p_sol.x.array[:] = p_n.x.array[:]

    for iteration in range(max_iter):
        u_old = u_sol.x.array.copy()
        p_old = p_sol.x.array.copy()
        
        # Resolver momentum (con p del paso anterior)
        u_sol = problem_u.solve()
        
        # Resolver flujo (con u recién calculado)
        p_sol = problem_p.solve()
        
        # Convergencia
        error_u = np.linalg.norm(u_sol.x.array - u_old)
        error_p = np.linalg.norm(p_sol.x.array - p_old)
        
        if error_u < tol and error_p < tol:
            p_max = np.max(np.abs(p_sol.x.array))
            print(f"{step+1:<6} {t:<10.3f} {iteration+1:<6} {error_u:<12.2e} {error_p:<12.2e} {p_max:<12.2e}")
            break
    
    # Actualizar para siguiente paso
    u_n.x.array[:] = u_sol.x.array[:]
    p_n.x.array[:] = p_sol.x.array[:]
    
    # Guardar resultados cada 10 pasos
    if (step + 1) % 10 == 0:
        vtx_u.write(t)
        vtx_p.write(t)

# Cerrar archivos
vtx_u.close()
vtx_p.close()


# # --- Funtion space creation (for pressure) ---
# t_cg2 = bufl.element(family="Lagrange", cell=mesh.topology.cell_name(), degree=2, shape=(dim, dim))
# V_ = dlx.fem.functionspace(mesh, t_cg2)
# u_, v_ = TrialFunction(V_), TestFunction(V_)

# a = inner(u_, v_) * dx
# L = inner(sigma(uh), v_) * dx

# problem = LinearProblem(a, L, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
# sh = problem.solve()
# sh.name = "stress"

# # --- Saving data to visualize using ParaView ---
# if save_data:
#     with dlx.io.VTXWriter(comm, ed/'results_displacement.bp', [uh], engine='BP4') as vtx:
#         vtx.write(0.0)

#     with dlx.io.VTXWriter(comm, ed/'results_stress.bp', [sh], engine='BP4') as vtx:
#         vtx.write(0.0)