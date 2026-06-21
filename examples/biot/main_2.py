import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import ufl
from basix import ufl as bufl

# Crear malla
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0, 0]), np.array([1, 1])],
    [30, 30],
    cell_type=mesh.CellType.triangle
)

# Parámetros del modelo de Biot
E = 1e7          # Módulo de Young [Pa]
nu = 0.3         # Coeficiente de Poisson
k = 1e-10        # Permeabilidad [m²]
mu_f = 1e-3      # Viscosidad del fluido [Pa·s]
alpha = 0.8      # Coeficiente de Biot
M = 1e9          # Módulo de Biot [Pa]

# Derivar parámetros elásticos
lmbda = E * nu / ((1 + nu) * (1 - 2*nu))
G = E / (2 * (1 + nu))
K = lmbda + 2*G/3

# Permeabilidad específica
kappa = k / mu_f

# Paso de tiempo
dt = fem.Constant(domain, 0.005)
t_end = 4.0

# ============================================
# DEFINICIÓN DE ESPACIOS DE ELEMENTOS FINITOS
# ============================================

# Elementos para desplazamiento (vectorial, grado 2) y presión (escalar, grado 1)
P2_vec = bufl.element("Lagrange", domain.topology.cell_name(), 2, shape=(domain.geometry.dim,))
P1 = bufl.element("Lagrange", domain.topology.cell_name(), 1)

# Espacio mixto
mixed_element = bufl.mixed_element([P2_vec, P1])
W = fem.functionspace(domain, mixed_element)

# Espacios individuales para sub-funciones
V, V_to_W = W.sub(0).collapse()  # Desplazamiento
Q, Q_to_W = W.sub(1).collapse()  # Presión

print(f"=== Configuración del problema de Biot (FEniCSx) ===")
print(f"Grados de libertad totales: {W.dofmap.index_map.size_global}")
print(f"DOFs desplazamiento: {V.dofmap.index_map.size_global}")
print(f"DOFs presión: {Q.dofmap.index_map.size_global}")

# ============================================
# DEFINICIÓN DE FUNCIONES Y VARIABLES
# ============================================

# Funciones de solución
w = fem.Function(W)           # Solución en tiempo n+1
w_old = fem.Function(W)       # Solución en tiempo n

# Separar componentes (desplazamiento y presión)
u, p = ufl.split(w)
u_old, p_old = ufl.split(w_old)

# Funciones de prueba
v, q = ufl.TestFunctions(W)

# ============================================
# CONDICIONES DE FRONTERA
# ============================================

# Localizar facetas de frontera
fdim = domain.topology.dim - 1

# Frontera inferior (y = 0): desplazamiento fijo
def boundary_bottom(x):
    return np.isclose(x[1], 0)

facets_bottom = mesh.locate_entities_boundary(domain, fdim, boundary_bottom)
dofs_bottom = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_bottom)

u_bc = fem.Function(V)
u_bc.x.array[:] = 0.0
bc_u = fem.dirichletbc(u_bc, dofs_bottom, W.sub(0))

# Fronteras laterales (x = 0 y x = 1): presión nula (drenaje)
def boundary_sides(x):
    return np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1))

facets_sides = mesh.locate_entities_boundary(domain, fdim, boundary_sides)
dofs_sides = fem.locate_dofs_topological((W.sub(1), Q), fdim, facets_sides)

p_bc = fem.Function(Q)
p_bc.x.array[:] = 0.0
bc_p = fem.dirichletbc(p_bc, dofs_sides, W.sub(1))

# Lista de condiciones de frontera
bcs = [bc_u, bc_p]

# ============================================
# TENSOR DE DEFORMACIÓN Y ESFUERZOS
# ============================================

def epsilon(u):
    """Tensor de deformación"""
    return ufl.sym(ufl.grad(u))

def sigma_eff(u):
    """Tensor de esfuerzos efectivos (elástico lineal)"""
    return 2*G*epsilon(u) + lmbda*ufl.tr(epsilon(u))*ufl.Identity(len(u))

def sigma(u, p):
    """Tensor de esfuerzos totales de Biot"""
    return sigma_eff(u) - alpha*p*ufl.Identity(len(u))

# ============================================
# FORMULACIÓN DÉBIL
# ============================================

# Ecuación de equilibrio mecánico
F_momentum = ufl.inner(sigma(u, p), epsilon(v)) * ufl.dx

# Ecuación de conservación de masa del fluido
# Término de almacenamiento: (alpha²/M)(p - p_old)/dt
storage_term = (alpha**2/M) * (p - p_old) / dt * q * ufl.dx

# Término de acoplamiento volumétrico: alpha * (div(u) - div(u_old))/dt
coupling_term = alpha * (ufl.div(u) - ufl.div(u_old)) / dt * q * ufl.dx

# Término de difusión: kappa * grad(p)·grad(q)
diffusion_term = kappa * ufl.dot(ufl.grad(p), ufl.grad(q)) * ufl.dx

F_mass = storage_term + coupling_term + diffusion_term

# Carga externa (ejemplo: carga vertical en tope)
# Definir medida para frontera superior
def boundary_top(x):
    return np.isclose(x[1], 1)

facets_top = mesh.locate_entities_boundary(domain, fdim, boundary_top)
mt = mesh.meshtags(domain, fdim, facets_top, np.full_like(facets_top, 1))
ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)

f_load = fem.Constant(domain, (0.0, -1e4))
F_momentum -= ufl.dot(f_load, v) * ds(1)

# Forma variacional completa
F = F_momentum + F_mass

# ============================================
# CONFIGURACIÓN DEL SOLVER NO LINEAL
# ============================================

# Crear problema no lineal
problem = NonlinearProblem(F, w, bcs)

# Crear solver de Newton
solver = NewtonSolver(MPI.COMM_WORLD, problem)

# Configurar parámetros del solver
solver.atol = 1e-8
solver.rtol = 1e-7
solver.max_it = 25
solver.convergence_criterion = "incremental"

# Configurar solver lineal (MUMPS para sistemas acoplados)
ksp = solver.krylov_solver
opts = PETSc.Options()
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"
opts["pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

# ============================================
# ARCHIVOS DE SALIDA
# ============================================

# Crear funciones para guardar resultados
u_out = fem.Function(V)
p_out = fem.Function(Q)

# Archivos VTX (formato moderno de FEniCSx)
vtx_u = io.VTXWriter(domain.comm, "results/displacement.bp", [u_out], engine="BP4")
vtx_p = io.VTXWriter(domain.comm, "results/pressure.bp", [p_out], engine="BP4")

print(f"\nAlpha (coeficiente de Biot): {alpha}")
print(f"M (módulo de Biot): {M:.2e} Pa")
print(f"Paso de tiempo: {dt.value} s")
print(f"Parámetros elásticos: λ={lmbda:.2e}, G={G:.2e}, K={K:.2e}")
print("\n=== Iniciando solución temporal ===\n")

# ============================================
# LOOP TEMPORAL
# ============================================

t = 0.0
step = 0

while t < t_end:
    t += dt.value
    step += 1
    
    if domain.comm.rank == 0:
        print(f"Paso {step}: t = {t:.2f} s")
    
    # Resolver el sistema acoplado
    n_iter, converged = solver.solve(w)
    
    if converged:
        # Extraer componentes de la solución
        u_out.x.array[:] = w.x.array[V_to_W]
        p_out.x.array[:] = w.x.array[Q_to_W]
        
        # Guardar resultados
        vtx_u.write(t)
        vtx_p.write(t)
        
        if domain.comm.rank == 0:
            print(f"  Convergió en {n_iter} iteraciones")
            print(f"  Max desplazamiento: {u_out.x.array.max():.6e} m")
            print(f"  Max presión: {p_out.x.array.max():.6e} Pa\n")
        
        # Actualizar solución anterior
        w_old.x.array[:] = w.x.array[:]
    else:
        if domain.comm.rank == 0:
            print("  ¡No convergió!")
        break

# Cerrar archivos
vtx_u.close()
vtx_p.close()

if domain.comm.rank == 0:
    print("=== Simulación completada ===")