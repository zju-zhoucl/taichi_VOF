import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

ti.init(arch=ti.x64, default_fp=ti.f64)

nx = 32  # Number of grid points in the x direction
ny = 32  # Number of grid points in the y direction
res = 32
Lx = 1.0  # The length of the domain
Ly = 1.0  # The width of the domain

# Physical parameters
rho_water = 1.0
rho_air = 0.5
nu_water = 0.001  # coefficient of kinematic viscosity
nu_air = 0.0005

# Direction and magnitude of volume force
gx = 0
gy = -1

# The initial volume fraction of the domain
x1 = 0
x2 = 0.5
y1 = 0.0
y2 = 0.3

# Solution parameters
dt = 0.001

imin = 1
imax = imin + nx - 1
jmin = 1
jmax = jmin + ny - 1

F = ti.field(float, shape=(imax + 2, jmax + 2))
u = ti.field(float, shape=(imax + 2, jmax + 2))
v = ti.field(float, shape=(imax + 2, jmax + 2))
p = ti.field(float, shape=(imax + 1, jmax + 1))
Fn = ti.field(float, shape=(imax + 2, jmax + 2))
x = ti.field(float, shape=imax + 2)
y = ti.field(float, shape=imax + 2)
xm, ym = ti.field(float, shape=imax + 1), ti.field(float, shape=imax + 1)
x.from_numpy(np.hstack((0, np.linspace(0, Lx, nx + 1))))
y.from_numpy(np.hstack((0, np.linspace(0, Ly, ny + 1))))
N = nx * ny
L = ti.linalg.SparseMatrixBuilder(N, N, max_num_triplets=N * 6)
dx = x[imin + 1] - x[imin]
dy = y[jmin + 1] - y[jmin]
dxi = 1 / dx
dyi = 1 / dy

# The variables in the Poisson solver
u_star = ti.field(float, shape=(imax + 2, jmax + 2))
v_star = ti.field(float, shape=(imax + 2, jmax + 2))
R = ti.field(float, shape=((imax - imin + 1) * (imax - imin + 1)))
pv = ti.field(float, shape=((imax - imin + 1) * (imax - imin + 1)))


@ti.kernel
def grid_staggered():
    for i in ti.ndrange(imin, imax + 1):
        xm[i] = 0.5 * (x[i] + x[i + 1])
    for j in ti.ndrange(jmin, jmax + 1):
        ym[j] = 0.5 * (y[j] + y[j + 1])


@ti.kernel
def set_init_F():
    # Sets the initial volume fraction
    for j, i in ti.ndrange(jmax + 1, imax + 1):
        if (xm[i] >= x1) and (xm[i] <= x2) and (ym[j] >= y1) and (ym[j] <= y2):
            F[i, j] = 1


@ti.kernel
def Laplace_operator(A: ti.types.sparse_matrix_builder()):
    for i, j in ti.ndrange(res, res):
        row = i * res + j
        if row != 0:
            center = 0.0
            if j != 0:
                A[row, row - 1] += -1.0 * dxi ** 2
                center += 1.0
            if j != res - 1:
                A[row, row + 1] += -1.0 * dxi ** 2
                center += 1.0
            if i != 0:
                A[row, row - res] += -1.0 * dxi ** 2
                center += 1.0
            if i != res - 1:
                A[row, row + res] += -1.0 * dxi ** 2
                center += 1.0
            A[row, row] += center * dxi ** 2
    for i in ti.ndrange(res * res):
        if i == 0:
            A[0, i] += 1
        else:
            A[0, i] += 0


@ti.kernel
def set_BC():
    # Set the wall as impenetrable and slip
    for i in ti.ndrange(imax + 2):
        # bottom
        u[i, jmin - 1] = u[i, jmin]
        v[i, jmin - 1] = v[i, jmin]
        F[i, jmin - 1] = F[i, jmin]
        # top
        u[i, jmax + 1] = u[i, jmax]
        v[i, jmax + 1] = v[i, jmax]
        F[i, jmax + 1] = F[i, jmax]
    for j in ti.ndrange(jmax + 2):
        # left
        u[imin - 1, j] = u[imin, j]
        v[imin - 1, j] = v[imin, j]
        F[imin - 1, j] = F[imin, j]
        # right
        u[imax + 1, j] = u[imax, j]
        v[imax + 1, j] = v[imax, j]
        F[imax + 1, j] = F[imax, j]


@ti.func
def var(a, b, c):
    # Find the median of a,b, and c
    center = a + b + c - ti.max(a, b, c) - ti.min(a, b, c)
    return center


@ti.kernel
def cal_mu_rho():
    # Calculate density rho and kinematic viscosity Mu
    for j, i in ti.ndrange((jmin - 1, jmax + 2), (imin - 1, imax + 2)):
        rho[i, j] = rho_air * (1 - var(0, 1, F[i, j])) + rho_water * var(0, 1, F[i, j])
        mu[i, j] = (nu_water * rho_water * var(0, 1, F[i, j]) + nu_air * rho_air * rho_air * (1 - var(0, 1, F[i, j]))) / \
                   rho[i, j]


@ti.kernel
def M_Possion():
    # Solving Pressure Poisson Equation Using Projection Method
    for j, i in ti.ndrange((jmin, jmax + 1), (imin + 1, imax + 1)):
        v_here = 0.25 * (v[i - 1, j] + v[i - 1, j + 1] + v[i, j] + v[i, j + 1])
        u_star[i, j] = (u[i, j] + dt *
                        (mu[i, j] * (u[i - 1, j] - 2 * u[i, j] + u[i + 1, j]) * dxi ** 2
                         + mu[i, j] * (u[i, j - 1] - 2 * u[i, j] + u[i, j + 1]) * dyi ** 2
                         - u[i, j] * (u[i + 1, j] - u[i - 1, j]) * 0.5 * dxi
                         - v_here * (u[i, j + 1] - u[i, j - 1]) * 0.5 * dyi
                         + gx))
    for j, i in ti.ndrange((jmin + 1, jmax + 1), (imin, imax + 1)):
        u_here = 0.25 * (u[i, j - 1] + u[i, j] + u[i + 1, j - 1] + u[i + 1, j])
        v_star[i, j] = (v[i, j] + dt *
                        (mu[i, j] * (v[i - 1, j] - 2 * v[i, j] + v[i + 1, j]) * dxi ** 2
                         + mu[i, j] * (v[i, j - 1] - 2 * v[i, j] + v[i, j + 1]) * dyi ** 2
                         - u_here * (v[i + 1, j] - v[i - 1, j]) * 0.5 * dxi
                         - v[i, j] * (v[i, j + 1] - v[i, j - 1]) * 0.5 * dyi
                         + gy))
    for j, i in ti.ndrange((jmin, jmax + 1), (imin, imax + 1)):
        R[i - imin + (j - 1) * (imax + 1 - imin)] = (-rho[i, j] / dt * (
                (u_star[i + 1, j] - u_star[i, j]) * dxi + (v_star[i, j + 1] - v_star[i, j]) * dyi))


@ti.kernel
def update():
    for j, i in ti.ndrange((jmin, jmax + 1), (imin, imax + 1)):
        p[i, j] = pv[i - imin + (j - 1) * (imax + 1 - imin)]
    for j, i in ti.ndrange((jmin, jmax + 1), (imin + 1, imax + 1)):
        u[i, j] = u_star[i, j] - dt / rho[i, j] * (p[i, j] - p[i - 1, j]) * dxi
    for j, i in ti.ndrange((jmin + 1, jmax + 1), (imin, imax + 1)):
        v[i, j] = v_star[i, j] - dt / rho[i, j] * (p[i, j] - p[i, j - 1]) * dyi


@ti.kernel
def solve_F():
    for j, i in ti.ndrange((jmin, jmax + 1), (imin, imax + 1)):
        u_loc = 3 / 8 * (u[i, j] + u[i + 1, j]) + 1 / 16 * \
                (u[i, j + 1] + u[i + 1, j + 1] + u[i, j - 1] + u[i + 1, j - 1])
        v_loc = 3 / 8 * (v[i, j] + v[i, j + 1]) + 1 / 16 * \
                (v[i - 1, j] + v[i - 1, j + 1] + v[i + 1, j] + v[i + 1, j + 1])
        F[i, j] = (F[i, j] - dt *
                   (u_loc * (F[i + 1, j] - F[i - 1, j]) * dxi / 2 +
                    v_loc * (F[i, j + 1] - F[i, j - 1]) * dyi / 2))
        F[i, j] = var(0, 1, F[i, j])


# Calculate the coordinates of the staggered point
grid_staggered()
# Set initial volume fraction
set_init_F()

# initialize rho, mu
rho = ti.field(float, shape=(imax + 2, jmax + 2))
mu = ti.field(float, shape=(imax + 2, jmax + 2))

# Create Laplace operator
Laplace_operator(L)
A = L.build()

istep = 0
istep_max = 50000

nstep = 100
R_limit = 15.0
count = -1
check_mass = np.zeros((int(istep_max / nstep), 1))  # 检查质量
while istep < istep_max:
    # set boundary conditions
    set_BC()
    istep = istep + 1
    # Update rho, mu by F
    cal_mu_rho()

    # Solving Pressure Poisson Equation Using Projection Method
    M_Possion()
    solver = ti.linalg.SparseSolver(solver_type="LU")
    solver.analyze_pattern(A)
    solver.factorize(A)
    pv_init = solver.solve(R)
    pv.from_numpy(pv_init)
    isSuccess = solver.info()
    update()
    solve_F()
    set_BC()
    istep = istep + 1  # time step +1
    if np.mod(istep, nstep) == 0:  # Output data every 100 steps
        count = count + 1
        Fn1 = F.to_numpy()
        check_mass[count] = sum(sum(abs(Fn1[imin:imax + 1, jmin: jmax + 1])))
        print('Number of iterations', str(istep), '\n check mass：', str(check_mass[count]),
              '\n')
        plt.figure(figsize=(5, 5))
        xm1 = xm.to_numpy()
        ym1 = ym.to_numpy()

        X, Y = np.meshgrid(xm1[imin:imax + 1], ym1[jmin:jmax + 1])
        plt.contour(xm1[imin:imax + 1], ym1[jmin:jmax + 1], Fn1[imin:imax + 1, jmin:jmax + 1].T, [0.5], cmap=plt.cm.jet)
        plt.savefig(str(istep) + '.png')
