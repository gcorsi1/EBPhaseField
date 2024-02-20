from argparse import ArgumentParser

import gmsh
import meshio
import numpy as np
import pygmsh
import sympy
import ufl_legacy as ufl
from dolfin import *

# Suppress excessive output in parallel:
parameters["std_out_all_processes"] = False

# Get the communicator world
mpi_comm = MPI.comm_world
mpiRank = MPI.rank(mpi_comm)


# create the mesh, can specify 'triangle' or 'line' as cell types in order to
# get the mesh or the domains, respectively
def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(
        points=points, cells={cell_type: cells}, cell_data={"region": [cell_data]}
    )
    return out_mesh


def eigvalues(T):
    TI, TII = (tr(T) + sqrt(tr(T) ** 2 - 4 * det(T))) / 2, (
        tr(T) - sqrt(tr(T) ** 2 - 4 * det(T))
    ) / 2
    return [TI, TII]


def eigvectors(T):
    tol = 1.0e-12

    TI, TII = eigvalues(T)[0], eigvalues(T)[1]
    Txx, Txy, Tyy = T[0, 0], T[0, 1], T[1, 1]

    theta_I = ufl.atan_2(TI - Txx, Txy)
    theta_II = theta_I + pi / 2

    vI = conditional(
        gt(abs(Txy), tol),
        as_vector([cos(theta_I), sin(theta_I)]),
        as_vector([0.0, 1.0]),
    )
    vII = conditional(
        gt(abs(Txy), tol),
        as_vector([cos(theta_II), sin(theta_II)]),
        as_vector([1.0, 0.0]),
    )

    return [vI, vII]


pp = lambda f: (f + abs(f)) / 2
pn = lambda f: (f - abs(f)) / 2


# Spectral decomposition
def sd_p(tensor):
    l0, l1 = eigvalues(tensor)[0], eigvalues(tensor)[1]
    p0, p1 = eigvectors(tensor)[0], eigvectors(tensor)[1]

    positive_spectral_decomposition = pp(l0) * outer(p0, p0) + pp(l1) * outer(p1, p1)

    return positive_spectral_decomposition


def sd_n(tensor):
    l0, l1 = eigvalues(tensor)[0], eigvalues(tensor)[1]
    p0, p1 = eigvectors(tensor)[0], eigvectors(tensor)[1]

    negative_spectral_decomposition = pn(l0) * outer(p0, p0) + pn(l1) * outer(p1, p1)

    return negative_spectral_decomposition


################################################################################################################

# Geometric and constitutive parameters. Units: kN, mm
Gcf = 1.0e-4
residual_stiffness = 2.0e-4
# Damage parameters
Gc, ell = Constant(Gcf), 0.02
Gc2 = Constant(10.0 * Gcf)

# Geometry and Material
Lx, Ly, h = 4.0, 1.0, 1.0
mesh_size_ref = ell / 2.0

# Elastic properties
nu = Constant(0.3)
Yf = 20.8
Y = Constant(Yf)
mu0 = Y / 2.0 / (1.0 + nu)
lb0 = 2.0 * mu0 * nu / (1.0 - 2.0 * nu)


# function for mesh refinement
def mesh_size(entity_dim, entity_tag, x, y, z, lc):
    return min(h / 10.0, mesh_size_ref + np.sqrt((x - Lx / 2.0) ** 2) / 100 * Lx)


# Generate the mesh
filename = "mesh.xdmf"
gmsh.initialize()
fact = 0.3
l_low = -h / 2.0 + fact * h
if mpiRank == 0:
    geometry = pygmsh.geo.Geometry()
    # Fetch model we would like to add data to
    model = geometry.__enter__()

    p1 = model.add_point([-Lx / 2.0, l_low, 0.0])
    p5 = model.add_point([Lx / 2.0, l_low, 0.0])
    p6 = model.add_point([Lx / 2.0, h / 2.0, 0.0])
    p11 = model.add_point([-Lx / 2.0, h / 2.0, 0.0])
    p2 = model.add_point([-Lx / 2.0, -h / 2.0, 0.0])
    p3 = model.add_point([Lx / 2.0, -h / 2.0, 0.0])

    l1 = model.add_line(p1, p5)
    l2 = model.add_line(p5, p6)
    l3 = model.add_line(p6, p11)
    l4 = model.add_line(p11, p1)
    l5 = model.add_line(p1, p2)
    l6 = model.add_line(p2, p3)
    l7 = model.add_line(p3, p5)

    ll = model.add_curve_loop([l1, l2, l3, l4])
    rec = model.add_plane_surface(ll)

    ll2 = model.add_curve_loop([l5, l6, l7, -l1])
    tri = model.add_plane_surface(ll2)

    # Call gmsh kernel before add physical entities
    model.synchronize()
    model.add_physical(rec, "Volume")
    model.add_physical(tri, "Volume_low")

    gmsh.model.mesh.setSizeCallback(mesh_size)

    geometry.generate_mesh()
    gmsh.write("mesh.msh")
    gmsh.clear()
    geometry.__exit__()

    # Now read the msh file and generate a XDMF
    mesh_from_file = meshio.read("mesh.msh")

    triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
    meshio.write(filename, triangle_mesh)

# tags of the mesh subregions
mesh_tag = 1
obstabcle_tag = 2

# after writing to file, read again the mesh with xdmf parallel dolfin interface
mesh = Mesh(mpi_comm)
mvc2d = MeshValueCollection("size_t", mesh, 2)
with XDMFFile(mpi_comm, filename) as infile:
    infile.read(mesh)
    infile.read(mvc2d, "region")
subregions = cpp.mesh.MeshFunctionSizet(mesh, mvc2d)
print("Read mesh: number of cells: %i.\n" % (mesh.num_cells()))

# tags of the mesh subregions
mesh_tag = 1
obstabcle_tag = 2


class Obstacle(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (-h / 2.0, l_low))


class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], -Lx / 2.0)


class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], Lx / 2.0)


class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], h / 2.0)


class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], -h / 2.0)


obstacle = Obstacle()

subregions.set_all(1)
Obstacle().mark(subregions, 2)
with XDMFFile(mpi_comm, "subregions.xdmf") as fi:
    fi.write(subregions)

domains = MeshFunction("size_t", mesh, 1)
domains.set_all(0)
Left().mark(domains, 1)
Right().mark(domains, 2)
Top().mark(domains, 3)
Bottom().mark(domains, 4)
ds = Measure("ds", domain=mesh, subdomain_data=domains)

# Mesh functions and measures
dx_m = Measure(
    "dx", domain=mesh, subdomain_data=subregions, metadata={"quadrature_degree": 2}
)


# SubDomains for boundary conditions
left = CompiledSubDomain("near(x[0], Lx) && on_boundary", Lx=-Lx / 2.0)
right = CompiledSubDomain("near(x[0], Lx) && on_boundary", Lx=Lx / 2.0)
right2 = CompiledSubDomain("near(x[0], Lx) && on_boundary && x[1] <= -0.1", Lx=Lx / 2.0)
up = CompiledSubDomain("near(x[1], y) && on_boundary", y=h / 2.0)
down = CompiledSubDomain("near(x[1], y) && on_boundary", y=-h / 2.0)
# nodes = np.array([[-Lx / 2.0, 0.0], [Lx / 2.0, 0.0]])
nodes = np.array(
    [
        [Lx / 2.0, 0.0],
    ]
)


# Check that points at the bases are found, printing the values to screen.
# It's a sanity check. This function is used in the definition of the boundary
# conditions
def specified_nodes(x, on_boundary):
    bb = any([np.allclose(x, row, rtol=1e-14) for row in nodes])
    if bb:
        print(f"FOUND: {x}")
    return bb


# Discrete space
V_u = VectorFunctionSpace(mesh, "CG", 1)
V_alpha = FunctionSpace(mesh, "CG", 1)

# Functions. Trial functions. Test functions
u, du, v = Function(V_u), TrialFunction(V_u), TestFunction(V_u)
u.rename("displacement", "displacement")

alpha, dalpha, beta = Function(V_alpha), TrialFunction(V_alpha), TestFunction(V_alpha)
alpha.rename("damage", "damage")

# Boundary conditions
bc_load0 = DirichletBC(V_u.sub(0), Constant(0.0), right)
bc_load1 = DirichletBC(V_u.sub(1), Constant(0.0), specified_nodes, "pointwise")
bc_load1 = DirichletBC(V_u.sub(1), Constant(0.0), right2)
bc_u = [bc_load0, bc_load1]
bc_alpha = [
    DirichletBC(V_alpha, Constant(0.0), left),
    DirichletBC(V_alpha, Constant(0.0), down),
]

# Initial damage
lb = interpolate(Constant("0."), V_alpha)  # lower bound, initialize to initial_alpha
ub = interpolate(Constant("1."), V_alpha)  # upper bound, set to 1

# Stiffness modulation and dissipated energy function
residual_stiffness = Constant(residual_stiffness)
a = lambda alpha: (1.0 - alpha) ** 2 + residual_stiffness
# alpha ** 2 is as in the article of miehe 2010, but it removes the elastic
# before fracture nucleation
# w = lambda alpha: alpha * alpha
w = lambda alpha: alpha


####################################################################################
z = sympy.Symbol("z")
c_w = 4 * sympy.integrate(sympy.sqrt(w(z)), (z, 0, 1))
c_1w = sympy.integrate(sympy.sqrt(1 / w(z)), (z, 0, 1))
tmp = 2 * (sympy.diff(w(z), z) / sympy.diff(1 / a(z), z)).subs({"z": 0})
sigma_c = sympy.sqrt(tmp * Gc * Yf / (c_w * ell))
eps_c = float(sigma_c / Yf)
Nc = sigma_c
Mc = sigma_c * 1 / 6
Tc = sigma_c * 1 / 12
if mpiRank == 0:
    print(f"{Nc=}, {Mc=}, {Tc=}\n")
####################################################################################

# Membrane strain
E = lambda u: sym(grad(u))

# Elastic energy densities (see Miehe, 2010)
Ep = sd_p(E(u))
En = sd_n(E(u))
psi_p = 0.5 * lb0 * pp(tr(E(u))) ** 2 + mu0 * inner(Ep, Ep)
psi_n = 0.5 * lb0 * pn(tr(E(u))) ** 2 + mu0 * inner(En, En)
psi = a(alpha) * psi_p + psi_n
psi_all = 0.5 * lb0 * tr(E(u)) ** 2 + mu0 * inner(E(u), E(u))

# Stress
N = (
    lambda u, alpha: a(alpha) * (lb0 * pp(tr(E(u))) * Identity(2) + 2.0 * mu0 * Ep)
    + lb0 * pn(tr(E(u))) * Identity(2)
    + 2.0 * mu0 * En
)

Np = lambda u, alpha: a(alpha) * (lb0 * pp(tr(E(u))) * Identity(2) + 2.0 * mu0 * Ep)
Nn = lambda u: +lb0 * pn(tr(E(u))) * Identity(2) + 2.0 * mu0 * En

# Initial u
u0_expr = Expression(
    ("0.0", "-(160./3.)*p + 4.0*p*pow(x[0], 2) - p*pow(x[0], 4)/24."),
    p=-1.0e-6,
    degree=2,
)
u.assign(u0_expr)
for bci in bc_u:
    bci.apply(u.vector())

# auxiliary stress boundary conditions
n = FacetNormal(mesh)
stress_TRACTION = Expression("c*1.0/(1.0 - std::pow(nu,2))", c=0.0, nu=nu, degree=2)
stress_BENDING = Expression(
    "c*12.0*x[1]/(1.0 - std::pow(nu,2))", c=0.0, nu=nu, degree=2
)
stress_SHEAR = Expression("c*1.0/(1.0 - std::pow(nu,2))", c=0.0, nu=nu, degree=2)
# neumann contribution
Lwork = inner(stress_TRACTION * n[0], u[0]) * ds(1) + inner(
    stress_SHEAR * n[0], u[1]
) * ds(1)

# Elastic energy
elastic_energy = psi * dx_m

# Dissipated energy
dissipated_energy = Gc / float(c_w) * (
    w(alpha) / ell + ell * dot(grad(alpha), grad(alpha))
) * dx_m(1) + Gc2 / float(c_w) * (
    w(alpha) / ell + ell * dot(grad(alpha), grad(alpha))
) * dx_m(
    2
)

# Total energy
total_energy = elastic_energy + dissipated_energy - Lwork

# First and second directional derivatives wrt u
F_u = derivative(total_energy, u, v)
J_u = derivative(F_u, u, du)

# First and second directional derivatives wrt alpha
F_alpha = derivative(total_energy, alpha, beta)
J_alpha = derivative(F_alpha, alpha, dalpha)

# Displacement variational problem
problem_u = NonlinearVariationalProblem(F_u, u, bc_u, J_u)
solver_u = NonlinearVariationalSolver(problem_u)

# Solver for the u-problem
prm = solver_u.parameters
# prm["newton_solver"]["linear_solver"] = "lu"
prm["newton_solver"]["maximum_iterations"] = 200
prm["newton_solver"]["absolute_tolerance"] = 1e-7
prm["newton_solver"]["relative_tolerance"] = 1e-7
# prm["newton_solver"]["convergence_criterion"] = "residual"
prm["newton_solver"]["error_on_nonconvergence"] = False
prm["newton_solver"]["report"] = True


# Define damage variational problem
class DamageProblem(OptimisationProblem):
    def f(self, x):
        """Function to be minimised"""
        alpha.vector()[:] = x
        return assemble(total_energy)

    def F(self, b, x):
        """Gradient (first derivative)"""
        alpha.vector()[:] = x
        assemble(F_alpha, b)

        for bc_i in self.bcs:
            bc_i.apply(b)

    def J(self, A, x):
        """Hessian (second derivative)"""
        alpha.vector()[:] = x
        assemble(J_alpha, A)

        for bc_i in self.bcs:
            bc_i.apply(A)


# Solver for the damage problem
solver_alpha_tao = PETScTAOSolver()
solver_alpha_tao.parameters.update(
    {
        "method": "tron",
        #    "linear_solver": "cg",
        "maximum_iterations": 10000,
        "line_search": "gpcg",
        #     "linear_solver" : "nash",
        #     "linear_solver" : "stcg",
        "preconditioner": "hypre_amg",
        "gradient_absolute_tol": 1.0e-05,
        "gradient_relative_tol": 1.0e-05,
        "gradient_t_tol": 1.0e-05,
        "monitor_convergence": False,
        "report": False,
    }
)


# Define alternate minimization algorithm
def alternate_minimization(
    alpha,
    tol=1.0e-4,
    maxiter=2000,
    alpha_0=interpolate(Constant("0.0"), V_alpha),
):
    # Initialization
    iter_step = 1
    err_alpha = 1
    alpha_error = Function(V_alpha)
    converged_u = False

    # Iteration loop
    while err_alpha > tol and iter_step < maxiter:
        # Solve elastic problem
        its_u, converged_u = solver_u.solve()
        if not converged_u:
            break
        if MPI.rank(MPI.comm_world) == 0:
            print(
                f"Solver for u converged: {converged_u}. "
                f"Number of iterations: {its_u}."
            )

        # Solve damage problem
        DamageProblem.bcs = bc_alpha  # apply damage boundary conditions
        solver_alpha_tao.solve(
            DamageProblem(), alpha.vector(), lb.vector(), ub.vector()
        )

        # Test error
        alpha_error.vector()[:] = alpha.vector() - alpha_0.vector()
        err_alpha = alpha_error.vector().norm("linf")

        # Monitor the results
        alpha_max = alpha.vector().max()
        if MPI.rank(MPI.comm_world) == 0:
            print(
                "Iteration:  %2d, Error: %2.8g, alpha_max: %.8g"
                % (iter_step, err_alpha, alpha_max)
            )

        # Update iteration
        alpha_0.assign(alpha)
        iter_step = iter_step + 1

    return converged_u


"""
Now setup loading and postprocessing.
"""

# Output folder and files
output_dir = "./"

# Projection spaces
S1 = TensorFunctionSpace(mesh, "P", 1)
S2 = VectorFunctionSpace(mesh, "P", 1)


def postprocessing():
    E_ = project(E(u), S1)
    E_.rename("strain", "strain")
    N_ = project(N(u, alpha), S1)
    N_.rename("stress", "stress")
    return E_, N_


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-nnc",
        "--ndivnc",
        dest="NNc",
        help="compression normalized with critical traction load",
        type=float,
        default=-1.0,
    )

    args = parser.parse_args()
    if MPI.rank(MPI.comm_world) == 0:
        print(f"Running with {args=}\n")

    # Load (imposed displacement)
    initial_angle_displacement = 0.00002
    # rotation and compression/traction displacement have a fixed ratio, positive
    # for traction, negative for compression
    angle_compression_ratio = args.NNc
    delta_l = float(sigma_c / 100.0)
    delta_l_reduce_fact = 1.0 / 2.0

    filename_alpha_u = output_dir + f"alphau_tan{100*angle_compression_ratio:.0f}.xdmf"
    filename_iters = f"energies_tan{100*angle_compression_ratio:.0f}.txt"

    # Solution
    with XDMFFile(MPI.comm_world, filename_alpha_u) as file_alpha_u:
        file_alpha_u.parameters.update(
            {"functions_share_mesh": True, "rewrite_function_mesh": False}
        )

        t = delta_l
        i_t = 0
        alpha_0 = interpolate(Constant(0.0), V_alpha)
        alpha_0_iter = interpolate(Constant(0.0), V_alpha)
        u_0 = interpolate(Constant((0.0, 0.0)), V_u)

        it_cracked = 0
        delta_crack_initial = 0.0
        delta_crack = 0.0
        cracklen_prev = 0.0
        cracklen_last = 0.0

        while True:
            if MPI.rank(MPI.comm_world) == 0:
                print(
                    "\n\n" + "=" * 80 + "\n",
                    f"Now solving load increment {i_t + 1}. Load: {t:.5f}.",
                    "\n" + "=" * 80,
                )

            stress_BENDING.c = t
            stress_SHEAR.c = t
            stress_TRACTION.c = angle_compression_ratio * Nc

            # Solve alternate minimization
            converged = alternate_minimization(
                alpha,
                tol=1.0e-4,
                maxiter=2000,
                alpha_0=alpha_0_iter,
            )
            if not converged:
                if MPI.rank(MPI.comm_world) == 0:
                    print(
                        f"\nNot converged for {delta_l=}, reducing step size "
                        f"to {delta_l_reduce_fact*delta_l}.\n"
                    )
                t -= delta_l
                delta_l *= delta_l_reduce_fact
                t += delta_l
                u.assign(u_0)
                alpha.assign(alpha_0)
                continue
            else:
                u_0.assign(u)
                alpha_0.assign(alpha)
                t += delta_l
                i_t += 1

            ux, uy = u.split(deepcopy=True)

            # updating the lower bound to account for the irreversibility
            lb.vector()[:] = alpha.vector()
            alpha_max = alpha.vector().max()
            if alpha_max == 1.0:
                it_cracked += 1
                cracklen_last = 1.0 / Gcf * assemble(dissipated_energy)
                if it_cracked == 2:
                    delta_crack_initial = cracklen_last - cracklen_prev
                delta_crack = cracklen_last - cracklen_prev
                cracklen_prev = cracklen_last
                if MPI.rank(MPI.comm_world) == 0 and it_cracked > 2:
                    print(
                        f"\n{delta_crack=}, {delta_crack_initial=}, ratio: {delta_crack/delta_crack_initial}. "
                        f"load is angle {t}, displacement {angle_compression_ratio*t}.\n"
                    )

            if MPI.rank(MPI.comm_world) == 0:
                print(f"\nCrack lenght is: {cracklen_last:.4f}.\n")
            # postprocess and write to file
            savestep = 1
            save_to_file = (not i_t % savestep) or alpha_max == 1.0

            E_, N_ = postprocessing()
            # if True, postprocess and write to file for u, alpha, stress fields
            if save_to_file:
                file_alpha_u.write(u, i_t)
                file_alpha_u.write(alpha, i_t)
                file_alpha_u.write(N_, i_t)

            if alpha_max == 1.0:
                break

        print(f"FINAL: {t=}, {i_t=}, {angle_compression_ratio=}, {cracklen_prev=}.\n")
