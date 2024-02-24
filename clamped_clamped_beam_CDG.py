"""
Solve the damage problem for a 1D beam, with two damage descriptors
used along thickness in a thin beam.
"""

from argparse import ArgumentParser

import numpy as np
import pandas as pd
import sympy
import ufl_legacy as ufl
from dolfin import *


# Option2: Lumped-mass projection to CG1, which remains monotone.
def lumpedProject(f, V):
    v = TestFunction(V)
    lhs = assemble(inner(Constant(1.0), v) * dx)
    rhs = assemble(inner(f, v) * dx)
    u = Function(V)
    as_backend_type(u.vector()).vec().pointwiseDivide(
        as_backend_type(rhs).vec(), as_backend_type(lhs).vec()
    )
    return u


# Output folder and files
output_dir = "./"

# Suppress excessive output in parallel:
parameters["std_out_all_processes"] = False

# Get the communicator world
mpi_comm = MPI.comm_world
mpiRank = MPI.rank(mpi_comm)


NLAYERS = 2

# Geometric and constitutive parameters. Units: kN, mm
Gcf = 0.000666666666667
# Stiffness modulation and dissipated energy function
residual_stiffness = 0.2 * Gcf
residual_stiffness = Constant(residual_stiffness)
# Damage parameters
Gc, ell = Constant(Gcf), 0.01

# Geometry and Material
Lx, b, h = 1.0, 1.0, 0.1
mesh_size_ref = 0.5 * ell

# Elastic properties
nu = Constant(0.3)
nuf = float(nu)
Yf = 1
Y = Constant(Yf)

# Critical Epsilon. Notice the (1-nu^2) term, needed if plane strain
epsc = np.sqrt(3.0 / 2.0 * Gcf / ell / Yf / h)
sigmacf = Yf * epsc
print("=" * 80 + "\n" + f"eps_c = {epsc}, sigma_c = {sigmacf}\n" + "=" * 80 + "\n")
epscu = ufl.sqrt(3.0 / 2.0 * Gcf / ell / Y / h)
sigmac = Y * epscu

mesh = UnitIntervalMesh(int(1.0 / mesh_size_ref))

nodes = [0.0, 1.0]

# SubDomains for boundary conditions
left = CompiledSubDomain("near(x[0], 0.0) && on_boundary")
right = CompiledSubDomain("near(x[0], 1.0) && on_boundary")


def specified_nodes(x, on_boundary):
    bb = any([np.allclose(x[0], ni, rtol=1e-14) for ni in nodes])
    return bb


facet_function = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
facet_function.set_all(0)
left.mark(facet_function, 1)
right.mark(facet_function, 2)

ds = Measure("ds", domain=mesh, subdomain_data=facet_function)

# Discrete space
P1 = FiniteElement("P", interval, degree=1)
V_def = FunctionSpace(mesh, "CG", 1)
V_alpha = FunctionSpace(mesh, "CG", 1)
V_sigma = FunctionSpace(mesh, "DG", 0)

# membrane, transverse, rotation
element = MixedElement([P1, P1])

# Function Spaces
Q = FunctionSpace(mesh, element)
q, q_, qt = TrialFunction(Q), Function(Q), TestFunction(Q)
v_, w_ = split(q_)
vu, wu = split(q)
vt, wt = split(qt)
q_.rename("v_w", "v_w")

aphs = [Function(V_alpha) for _ in range(NLAYERS)]
dalpha, beta = (
    TrialFunction(V_alpha),
    TestFunction(V_alpha),
)
for i, aphi in enumerate(aphs):
    stri = f"alpha{i}"
    if i == 0:
        stri = "alpha"
    elif i == 1:
        stri = "beta"
    aphi.rename(stri, stri)

# Boundary conditions: 0 - left,  1 - right
v0be = Expression("-e", e=0.0, degree=2)
v0bc = Expression("-c", c=0.0, degree=2)
v1be = Expression("e", e=0.0, degree=2)
v1bc = Expression("c", c=0.0, degree=2)
precompr = Expression("c", c=0.0, degree=2)

# Will enforce Dirichlet weakly
bc_u = []
bc_alpha = [
    DirichletBC(V_alpha, Constant(0.0), right),
]

# Initial damage
lbs = [
    interpolate(Constant(0.0), V_alpha) for _ in range(NLAYERS)
]  # lower bound, initialize to initial_alpha
ub = interpolate(Constant(2.0), V_alpha)  # upper bound, set to 2!!

# alpha ** 2 is as in the article of miehe 2010, but it removes the elastic
# limit before fracture nucleation
w = lambda alpha: alpha * alpha
w = lambda alpha: alpha

####################################################################################
# Calculation of the normalization factor
a0 = lambda alpha: (1.0 - alpha) ** 2
z = sympy.Symbol("z")
c_w = 4 * sympy.integrate(sympy.sqrt(w(z)), (z, 0, 1))
print(f"\nc_w value is {c_w}\n")
####################################################################################
aph0, aph1 = aphs
# Stiffnesses
A, B, C = (
    0.5 * ((2 - aph0 - aph1) + residual_stiffness) * h * Y,
    1.0
    / 24.0
    * (
        (
            (2 - aph0 - aph1) * (1 + aph0**2 + (-1 + aph1) * aph1 - aph0 * (1 + aph1))
            + residual_stiffness
        )
        * h**3
        * Y
    ),
    -1.0 / 8.0 * ((aph0 - aph1) * (-2 + aph0 + aph1) * h**2 * Y),
)

CLAMP_COEFF = 500.0
A0, B0 = Function(V_alpha), Function(V_alpha)
A0.vector()[:] = CLAMP_COEFF * h * Yf
B0.vector()[:] = CLAMP_COEFF * 1 / 12.0 * h**3 * Yf


eps, wprime = grad(v_), grad(w_)[0]
chi = grad(wprime)

M = B * chi + C * eps
N = A * eps + C * chi

epsu, wprimeu = grad(vu), grad(wu)[0]
chiu = grad(wprimeu)
epst, wprimet = grad(vt), grad(wt)[0]
chit = grad(wprimet)

# Assume the domain in y is the interval [-h/2, h/2]
hint = h / NLAYERS
n = FacetNormal(mesh)
TAU = B0 * Constant(1.0 / mesh_size_ref)
TAU = B * Constant(1.0 / mesh_size_ref)
TAUG = B0  # * Constant(1.0 / mesh_size_ref) * 2
GAMMAG = A0  # * Constant(1.0 / mesh_size_ref)
TAUH = 2 * B * Constant(1.0 / mesh_size_ref)

L_ext = precompr * v_ * dx

wwt = TestFunction(V_def)
uut = TrialFunction(V_def)
Mb = (
    B * chi[0] * wwt * dx
    + C * eps[0] * wwt * dx
    - B * (avg(wwt) * jump(wprime, n)[0]) * dS
    - B * wwt * (wprime - v0bc) * n[0] * ds(1)
    - B * wwt * (wprime - v1bc) * n[0] * ds(2)
)
Nb = (
    A * eps[0] * uut * dx
    + C * chi[0] * uut * dx
    - C * (avg(uut) * jump(wprime, n)[0]) * dS
    - C * uut * (wprime - v0bc) * n[0] * ds(1)
    - C * uut * (wprime - v1bc) * n[0] * ds(2)
)
chirec = (
    chi[0] * wwt * dx
    - (avg(wwt) * jump(wprime, n)[0]) * dS
    - wwt * (wprime - v0bc) * n[0] * ds(1)
    - wwt * (wprime - v1bc) * n[0] * ds(2)
)
MASS = assemble(inner(uut, wwt) * dx)
proj_solver = LUSolver(MASS)
mproj = Function(V_def)
nproj = Function(V_def)
chiproj = Function(V_def)
mproj.rename("mproj", "mproj")
nproj.rename("nproj", "nproj")
chiproj.rename("chiproj", "chiproj")
A0.rename("A0", "A0")
B0.rename("B0", "B0")

# Elastic energy
elastic_energy = (
    (1.0 / 2.0) * inner(N, eps) * dx
    + (1.0 / 2.0) * inner(M, chi) * dx
    - inner(jump(wprime, n), avg(M)) * dS
    + (1.0 / 2.0) * TAU("+") * inner(jump(wprime, n), jump(wprime, n)) * dS
    - inner((wprime - v0bc) * n, M) * ds(1)
    + (1.0 / 2.0) * TAUH * inner((wprime - v0bc) * n, (wprime - v0bc) * n) * ds(1)
    - inner((wprime - v1bc) * n, M) * ds(2)
    + (1.0 / 2.0) * TAUH * inner((wprime - v1bc) * n, (wprime - v1bc) * n) * ds(2)
    + (1.0 / 2.0) * TAUG * (w_ * w_) * ds
    + (1.0 / 2.0) * GAMMAG * (v_ * v_) * ds
) - L_ext

"""
NOTICE THE DISSIPATED ENERGY IS NOW NORMALIZED SO THAT THE
CRITICAL EPSILON OF CLASSIC RESULT OF BAR IN PURE TRACTION IS
RECOVERED
"""
sigma_top_expr = (
    (
        4
        * (
            -6 * (B * chiproj + C * eps[0])
            + (1 + aph0 - 2 * aph1) * h * (A * eps[0] + C * chiproj)
        )
    )
    / ((-2 + aph0 + aph1 + 1e-3) ** 2 * h**2)
    / sigmac
)
sigma_bot_expr = (
    (
        4
        * (
            6 * (B * chiproj + C * eps[0])
            + (1 - 2 * aph0 + aph1) * h * (A * eps[0] + C * chiproj)
        )
    )
    / ((-2 + aph0 + aph1 + 1e-3) ** 2 * h**2)
    / sigmac
)

fsigmaa = -1000 * ufl.tanh(8 * (sigma_top_expr + 0.5)) + 1001
fsigmab = -1000 * ufl.tanh(8 * (sigma_bot_expr + 0.5)) + 1001
fsigmaaf = project(fsigmaa, V_alpha)
fsigmabf = project(fsigmab, V_alpha)
fsigmaaf.rename("fsigmaa", "fsigmaa")
fsigmabf.rename("fsigmab", "fsigmab")


ws = [fsigmaaf, fsigmabf]

# Dissipated energy
diss_ = []
for i, aphi in enumerate(aphs):
    diss_.append(
        Gc
        * 1.0  # * h
        / 1.0  # / 2.0
        / float(c_w)
        * (ws[i] / ell * aphi + ell * dot(grad(aphi), grad(aphi)))
        * dx
    )
dissipated_energy = sum(diss_)

# Total energy
total_energy = elastic_energy + dissipated_energy

# First and second directional derivatives wrt u
F_u = derivative(total_energy, q_, qt)
J_u = derivative(F_u, q_, q)

# First and second directional derivatives wrt alpha
Fas = [derivative(total_energy, aphi, beta) for aphi in aphs]
Jas = []
for i, aphi in enumerate(aphs):
    Jas.append(derivative(Fas[i], aphi, dalpha))


class NonlinearProblemPointSource(NonlinearProblem):
    def __init__(self, L, a, bcs):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
        self.bcs = bcs
        self.P = 0.0

    def F(self, b, x):
        assemble(self.L, tensor=b)
        point_source = PointSource(Q.sub(1), Point(0.5), self.P)
        point_source.apply(b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A, x):
        assemble(self.a, tensor=A)
        for bc in self.bcs:
            bc.apply(A, x)


problem_u = NonlinearProblemPointSource(F_u, J_u, bc_u)
solver_u = NewtonSolver()

# Solver for the u-problem
prm = solver_u.parameters
# prm["newton_solver"]["linear_solver"] = "lu"
prm["maximum_iterations"] = 200
prm["absolute_tolerance"] = 1e-8
prm["relative_tolerance"] = 1e-8
# prm["newton_solver"]["convergence_criterion"] = "residual"
prm["error_on_nonconvergence"] = False
prm["report"] = True


# Define damage variational problem
class DamageProblem(OptimisationProblem):
    def __init__(self, i, alpha):
        super().__init__()
        self.ind = i
        self.alpha = alpha
        self.bcs = bc_alpha

    def f(self, x):
        """Function to be minimised"""
        self.alpha.vector()[:] = x
        return assemble(total_energy)

    def F(self, b, x):
        """Gradient (first derivative)"""
        self.alpha.vector()[:] = x
        assemble(Fas[self.ind], b)
        # print(b.vec()[:])

        for bc_i in self.bcs:
            bc_i.apply(b)

    def J(self, A, x):
        """Hessian (second derivative)"""
        self.alpha.vector()[:] = x
        assemble(Jas[self.ind], A)

        for bc_i in self.bcs:
            bc_i.apply(A)


# Solver for the damage problem
PETScOptions.set("tao_max_funcs", 20000)
solver_alpha_tao = PETScTAOSolver()
solver_alpha_tao.parameters.update(
    {
        "method": "tron",
        "linear_solver": "default",
        "maximum_iterations": 10000,
        "line_search": "gpcg",
        #     "linear_solver" : "nash",
        # "linear_solver" : "stcg",
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
    aphs,
    aph0s=[interpolate(Constant(0.0), V_alpha) for _ in range(NLAYERS)],
    tol=1.0e-5,
    maxiter=2000,
):
    # Initialization
    iter_step = 1
    err_alpha = 1.0
    err_alphas = [1.0 for _ in range(NLAYERS)]
    alpha_error = Function(V_alpha)
    converged_u = False
    alpha_max = 0.0

    while err_alpha > tol and iter_step < maxiter:
        its_u, converged_u = solver_u.solve(problem_u, q_.vector())
        if not converged_u:
            break
        if MPI.rank(MPI.comm_world) == 0:
            print(
                f"Solver for u converged: {converged_u}. "
                f"Number of iterations: {its_u}."
            )

        Epsilonf = project(eps[0], V_def)
        Chif = project(chi[0], V_def)
        proj_solver.solve(chiproj.vector(), assemble(chirec))
        fsigmaaf.vector()[:] = lumpedProject(fsigmaa, V_alpha).vector()
        fsigmabf.vector()[:] = lumpedProject(fsigmab, V_alpha).vector()

        for ii, (ai, a0i, lbi) in enumerate(zip(aphs, aph0s, lbs)):
            i_other = 0 if ii == 1 else 1
            ub_new = interpolate(Constant(2.0), V_alpha)  # upper bound, set to 2!!
            ub_new.vector()[:] -= aphs[i_other].vector().get_local()

            # Solve damage problem
            solver_alpha_tao.solve(
                DamageProblem(ii, ai), ai.vector(), lbi.vector(), ub_new.vector()
            )

            vai = ai.vector().get_local()
            van = ai.vector().norm("l2")
            v0an = a0i.vector().norm("l2")
            print(
                f"alpha_{ii}: max {vai.max()}, min {vai.min()}, norm {van}, norm old {v0an}"
            )

            # Test error
            alpha_error.vector()[:] = ai.vector() - a0i.vector()
            err_alphas[ii] = alpha_error.vector().norm("linf")
            err_alpha = max(err_alphas)
            print(f"error for alpha_{ii}: {err_alphas[ii]}")

            # Monitor the results
            alpha_max = max(alpha_max, vai.max())

            # Update iteration
            a0i.assign(ai)

        if MPI.rank(MPI.comm_world) == 0:
            print(
                "\n" + "-" * 80,
                "\nIteration:  %2d, Error: %2.8g, alpha_max: %.8g"
                % (iter_step, err_alpha, alpha_max),
                "\n" + "-" * 80 + "\n",
            )

        iter_step = iter_step + 1

        A0.vector()[:] = lumpedProject(Constant(CLAMP_COEFF) * A, V_alpha).vector()
        B0.vector()[:] = lumpedProject(Constant(CLAMP_COEFF) * B, V_alpha).vector()
    return converged_u, alpha_max


def postprocessing():
    Epsilonf = project(eps[0], V_def)
    Chif = project(chi[0], V_def)
    Wprime = project(wprime, V_def)
    Normalf = project(A * eps[0] + C * chi[0], V_def)
    Momentf = project(B * chi[0] + C * eps[0], V_def)

    proj_solver.solve(mproj.vector(), assemble(Mb))
    proj_solver.solve(nproj.vector(), assemble(Nb))
    proj_solver.solve(chiproj.vector(), assemble(chirec))

    Wprime.rename("wprime", "wprime")
    Epsilonf.rename("epsilon", "epsilon")
    Chif.rename("chi", "chi")
    Normalf.rename("Normal", "Normal")
    Momentf.rename("Moment", "Moment")

    return Epsilonf, Chif, Wprime, Normalf, Momentf


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--delta", dest="delta", help="step delta", type=float, default=5.0e-4
    )
    parser.add_argument(
        "-c",
        "--compression",
        dest="comp",
        help="beam pre-compression",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()
    if MPI.rank(MPI.comm_world) == 0:
        print(f"Running with {args=}\n")

    delta_l = args.delta
    delta_l_reduce_fact = 1.0 / 2.0
    COMPRESSION = args.comp

    filename_alpha_u = output_dir + f"alphau_compr{100*COMPRESSION:.0f}.xdmf"
    filename_iters = f"energies_compr{100*COMPRESSION:.0f}.txt"

    # Solution
    with XDMFFile(MPI.comm_world, filename_alpha_u) as file_alpha_u:
        file_alpha_u.parameters.update(
            {"functions_share_mesh": True, "rewrite_function_mesh": False}
        )

        t = delta_l
        i_t = 0
        aph0s = [interpolate(Constant(0.0), V_alpha) for _ in range(NLAYERS)]
        q_0 = interpolate(Constant((0.0, 0.0)), Q)

        cracklen_last = 0.0
        outer_step = 0

        csv_data = []
        compression, vertload = 0.0, 0.0

        while True:
            if MPI.rank(MPI.comm_world) == 0:
                print(
                    "\n\n" + "=" * 80 + "\n" + "=" * 80 + "\n",
                    f"Now solving load increment {i_t + 1}. Load: {t:.6f}.",
                    "\n" + "=" * 80 + "\n" + "=" * 80 + "\n",
                )

            precompr.c = COMPRESSION
            problem_u.P = t

            # SAVE COMPRESSION AND VERTICAL LOAD FOR THIS STEP BEFORE UPDATING
            # THEM AFTER THE NUMERICAL SOLUTION
            compression = COMPRESSION
            vertload = t

            outer_step += 1  # for image visualization name
            # Solve alternate minimization
            converged, alpha_max = alternate_minimization(
                aphs,
                # aph0s,
                tol=1.0e-4,
                maxiter=2000,
            )
            if not converged:
                if MPI.rank(MPI.comm_world) == 0:
                    print(
                        f"\nNot converged for {delta_l=}, reducing step size "
                        f"to {delta_l_reduce_fact*delta_l}.\n"
                    )
                # break
                t -= delta_l
                delta_l *= delta_l_reduce_fact
                t += delta_l
                q_.assign(q_0)
                for i, aphi in enumerate(aphs):
                    aphi.assign(aph0s[i])
                continue
            else:
                q_0.assign(q_)
                for i, aphi in enumerate(aphs):
                    aph0s[i].assign(aphi)
                t += delta_l
                i_t += 1

            # updating the lower bound to account for the irreversibility
            for i, aphi in enumerate(aphs):
                lbs[i].vector()[:] = aphi.vector()
            if alpha_max > 0.0:
                # cracklen_last = 1.0 / Gcf * assemble(dissipated_energy)
                cracklen_last = (
                    aphs[0].vector().get_local().max() * h / 2.0
                )  # only alpha breaks

            if MPI.rank(MPI.comm_world) == 0:
                print(f"Crack lenght is: {cracklen_last:.4f}.")
            # postprocess and write to file
            savestep = 1
            save_to_file = (not i_t % savestep) or alpha_max > 0.0

            Epsilonf, Chif, Wprime, Normalf, Momentf = postprocessing()
            print(
                f"Epsilon over epsc {Epsilonf(0.0)/epsc}, "
                f"Chi*h/2 over epsc {Chif(0.0)*h/2.0/epsc}."
            )

            sigma_top = project(sigma_top_expr, V_sigma)
            sigma_bot = project(sigma_bot_expr, V_sigma)
            sigma_top.rename("sigma_top", "sigma_top")
            sigma_bot.rename("sigma_bot", "sigma_bot")
            sigma_top_max, sigma_bot_max = (
                sigma_top.vector().get_local().max(),
                sigma_bot.vector().get_local().max(),
            )  # assuming one processor!
            print(f"\nSIGMA TOP MAX: {sigma_top_max}, BOT MAX: " f"{sigma_bot_max}.\n")
            # if True, postprocess and write to file for u, alpha, stress fields
            if save_to_file:
                file_alpha_u.write(q_, i_t)
                for aphi in aphs:
                    file_alpha_u.write(aphi, i_t)
                file_alpha_u.write(Epsilonf, i_t)
                file_alpha_u.write(Chif, i_t)
                file_alpha_u.write(Wprime, i_t)
                file_alpha_u.write(Normalf, i_t)
                file_alpha_u.write(Momentf, i_t)
                file_alpha_u.write(sigma_top, i_t)
                file_alpha_u.write(sigma_bot, i_t)
                file_alpha_u.write(fsigmaaf, i_t)
                file_alpha_u.write(fsigmabf, i_t)
                file_alpha_u.write(mproj, i_t)
                file_alpha_u.write(A0, i_t)
                file_alpha_u.write(B0, i_t)
                file_alpha_u.write(nproj, i_t)
                file_alpha_u.write(chiproj, i_t)

            vx = project(q_.sub(0), V_alpha)
            vy = project(q_.sub(1), V_alpha)
            csvv = {
                "step": i_t * np.ones_like(vx.vector().get_local()),
                "axial_load": compression,
                "vertical_load": vertload,
                "x": V_alpha.tabulate_dof_coordinates().flatten(),
                "vx": vx.vector().get_local(),
                "vy": vy.vector().get_local(),
                "a": aph0.vector().get_local(),
                "b": aph1.vector().get_local(),
                "M": mproj.vector().get_local(),
                "N": nproj.vector().get_local(),
                "eps": Epsilonf.vector().get_local(),
                "chi": chiproj.vector().get_local(),
            }

            csv_data.append(csvv)
            # Break after first nucleation of crack, or follow its evolution?
            # if alpha_max == 2.0:
            #     break
            if i_t == 5:
                break


    csv_output_dir = output_dir  # + "csv/"
    for i, csvi in enumerate(csv_data):
        df = pd.DataFrame(csvi)
        df.to_csv(csv_output_dir + f"csv_step_{i}")
