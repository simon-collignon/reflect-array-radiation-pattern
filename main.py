import numpy as np
import matplotlib
from matplotlib import pyplot as plt

# constants

f = 1e10
a = 1  # side of the square reflectarray, in meter
h = 0.5  # height of the horn antenna, in meter
size = 101  # number of points per dimension
lam = 3e8 / f  # wave length
k = 2 * np.pi / lam  # wave number

points, dx = np.linspace(-a / 2, a / 2, size, retstep=True)

# unit vectors

bs_vec = np.sqrt(2)**-1 * np.array([1, 0, -1])  # broadside unit vector
po_vec = np.sqrt(2)**-1 * np.array([1, 0, 1])  # polarization unit vector
no_vec = np.array([0, 0, 1])  # normal unit vector


def fields():

    N = 8  # exponent of the cos() in the radiation pattern
    xx = np.copy(points) + a / 2  # horn in x = 0, y = 0
    e_field = np.zeros((size, size, 3), dtype=complex)
    h_field = np.zeros((size, size, 3), dtype=complex)

    for i, x in enumerate(xx):
        for j, y in enumerate(points):

            R = np.sqrt(x**2 + y**2 + h**2)  # distance
            u_vec = R**-1 * np.array([x, y, -h])  # u unit vector
            e_vec = po_vec - np.dot(u_vec, po_vec) * u_vec  # e unit vector

            e_field[i, j] = np.exp(-1j * k * R) * R**-1 * np.dot(
                bs_vec, u_vec)**N * e_vec
            h_field[i, j] = np.cross(u_vec, e_field[i, j])

    return e_field, h_field


def electric_current(h_field):

    e_current = np.copy(h_field)
    e_current[:, :] = np.cross(no_vec, h_field[:, :])

    return e_current


def radiation_pattern(h_field):

    rp = np.zeros((size, size, 3), dtype=complex)  # radiation pattern
    rp_tp = np.zeros((size, size, 2), dtype=complex)  # rp with e_t, e_p
    k_hat = np.zeros((size, size, 3), dtype=complex)  # fft of the currents
    g_t = size**2 * np.fft.ifft2(sign_alternate(h_field))  # center u_x and u_y

    u_xy = lam * dx**-1 * (np.arange(0, size) / size - 0.5)

    for r, u_x in enumerate(u_xy):
        for s, u_y in enumerate(u_xy):
            u_2 = u_x**2 + u_y**2
            if u_2 <= 1:
                u_vec = np.array([u_x, u_y, np.sqrt(1 - u_2)])
                k_hat[r, s] = np.cross(no_vec, g_t[r, s])
                rp[r, s] = k_hat[r, s] - np.dot(k_hat[r, s], u_vec) * u_vec

                e_theta = np.sqrt(u_2)**-1 * np.array(
                    [u_x**2, u_y * u_x, -u_2])
                e_theta = normalize(e_theta)
                e_phi = np.cross(u_vec, e_theta)

                rp_tp[r, s, 0] = np.dot(e_theta, rp[r, s])
                rp_tp[r, s, 1] = np.dot(e_phi, rp[r, s])

    return rp[:, :, 0:2], rp_tp


def normalize(a, axis=-1, order=2):

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1

    return a / np.expand_dims(l2, axis)


def scalar_field(field):

    return np.linalg.norm(field, axis=2)


def sign_alternate(a):

    for m, _ in enumerate(a[:, 0]):
        for n, _ in enumerate(a[0, :]):

            a[m, n] *= (-1)**(m + n)

    return a


def plot_current():

    x, y = np.meshgrid(points, points, indexing='ij')
    plt.figure()
    plt.contourf(x, y, scalar_field(current), 1000, cmap="jet")
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.axis('scaled')
    plt.savefig('../report/figures/current.png')

    # plt.figure()
    # plt.quiver(x, y, abs(current[:, :, 0].flatten()),
    #            abs(current[:, :, 1].flatten()))
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    # plt.axis('scaled')


def plot_radiation_pattern(rp, rp_tp):

    pts = np.arange(0, size)
    u_xy = lam * dx**-1 * (pts / size - 0.5)
    x, y = np.meshgrid(u_xy, u_xy, indexing='ij')

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

    ax1.contourf(x, y, scalar_field(rp), 1000, cmap="jet")
    ax1.set_title(r'$\left\Vert \mathbf{F} \right\Vert$')
    ax1.set_ylabel('$u_y$')
    ax1.set_xlabel('$u_x$')
    ax1.set_aspect('equal', 'box')

    ax2.contourf(x, y, abs(rp_tp[:, :, 0]), 1000, cmap="jet")
    ax2.set_title(r'$F_{\theta}$')
    ax2.set_xlabel('$u_x$')
    ax2.set_aspect('equal', 'box')

    ax3.contourf(x, y, abs(rp_tp[:, :, 1]), 1000, cmap="jet")
    ax3.set_title('$F_{\phi}$')
    ax3.set_xlabel('$u_x$')
    ax3.set_aspect('equal', 'box')

    plt.savefig('../report/figures/radiation_pattern.png')


if __name__ == "__main__":

    # If you don't have pdflatex installed, remove the update call below.
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'font.size': 14,
        'toolbar': 'None',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'legend.fancybox': False,
        'legend.shadow': False,
        'figure.figsize': [20, 8]
    })

    e_field, h_field = fields()
    current = electric_current(h_field)

    F, F_tp = radiation_pattern(h_field)
    plot_radiation_pattern(F, F_tp)
    # plot_current()
    # plt.tight_layout()

    plt.show()
