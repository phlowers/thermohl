import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import circmean, circstd, circvar, vonmises_line

from thermohl.distributions import (
    _vonmises_kappa,
    _vonmises_circ_var,
    truncnorm,
    wrapnorm,
    WrappedNormal,
    vonmises,
)

_twopi = 2 * np.pi


def circular_moments_wrapnorm():
    # define n averages/std couples used to create specific distributions
    n = 100
    xx = np.linspace(0, _twopi, n)
    mu = np.sin(xx)
    sg = np.linspace(0, 5, n + 1)[1:]

    # arrays to compute circular momenta instead of classic ones
    cavg = np.zeros_like(sg)
    cvar = np.zeros_like(sg)
    cstd = np.zeros_like(sg)

    # loop on all couples mu/sg, create distribution, sample it and compute
    # circular momenta. We let in comments the analytical formula used to
    # compute those momenta but we prefer to use scipy.stat's functions.
    for i in range(n):
        ns = 9999
        wn = WrappedNormal(mu[i], sg[i], lwrb=-np.pi)
        s = wn.rvs(ns)
        # z = np.exp(1j * s)
        # R = np.mean(z)
        # circ_mean = np.angle(R)
        cavg[i] = circmean(s, high=wn.uprb, low=wn.lwrb)
        # circ_var = 1 - R
        cvar[i] = circvar(s, high=wn.uprb, low=wn.lwrb)
        # circ_std = np.sqrt(-2. * np.log(np.abs(R)))
        cstd[i] = circstd(s, high=wn.uprb, low=wn.lwrb)

    # plot momenta : we observe that circular mean is very close to mu as long
    # as desired std is low; then for larger sg the mean is noisy and the std
    # and the variance tend to pi
    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].plot(sg, cavg, label="circ. mean")
    ax[0].plot(sg, mu, "--", c="gray", label="input mean")
    ax[1].plot(sg, cvar, label="circ. variance")
    ax[1].plot(sg, 1 - np.exp(-0.5 * sg**2), label="analytic")
    ax[2].plot(sg, cstd, label="circ. std")
    ax[2].plot(sg, sg, "--", c="gray")
    ax[2].axhline(y=np.pi, xmin=sg[0], xmax=sg[-1], c="gray", ls="--")
    for i in range(3):
        ax[i].grid(True)
        ax[i].legend()
    ax[2].set_xlabel("Input variance")
    return


def circular_moments_vonmises():
    # define n averages/std couples used to create specific distributions
    n = 50
    xx = np.linspace(0, _twopi, n)
    mu = np.sin(xx)
    sg = np.linspace(0, 5, n + 1)[1:]
    kp = _vonmises_kappa(sg)
    low = -np.pi
    high = +np.pi

    # arrays to compute classic momenta
    savg = np.zeros_like(kp)
    svar = np.zeros_like(kp)
    sstd = np.zeros_like(kp)

    # arrays to compute circular momenta instead of classic ones
    cavg = np.zeros_like(kp)
    cvar = np.zeros_like(kp)
    cstd = np.zeros_like(kp)

    # loop on all couples mu/sg, create distribution, sample it and compute
    # circular momenta. We let in comments the analytical formula used to
    # compute those momenta but we prefer to use scipy.stat's functions.
    for i in range(n):
        ns = 9999
        vm = vonmises_line(kp[i], loc=mu[i])
        s = vm.rvs(ns)
        # z = np.exp(1j * s)
        # R = np.mean(z)
        # circ_mean = np.angle(R)
        cavg[i] = circmean(s, high=high, low=low)
        # circ_var = 1 - R
        cvar[i] = circvar(s, high=high, low=low)
        # circ_std = np.sqrt(-2. * np.log(np.abs(R)))
        cstd[i] = circstd(s, high=high, low=low)

        # same thing but from distribution
        savg[i] = vm.mean()
        svar[i] = vm.var()
        sstd[i] = vm.std()

    plt.figure()
    plt.title("Von mises parameter and input std")
    plt.plot(sg, kp, label="computed inverse")
    plt.plot(sg, 1 / sg**2, "--", c="gray", label="classical approx")
    plt.xlabel("$\sigma$")
    plt.ylabel("$\kappa$")
    plt.grid(True)
    plt.legend()

    fig, ax = plt.subplots(nrows=3, ncols=1)
    # computed circular mean matches distribution's mean() method;
    ax[0].plot(sg, cavg, label="circ. mean")
    ax[0].plot(sg, savg, label="dist. mean")
    ax[0].plot(sg, mu, "--", c="gray", label="input mean")

    # computed circular variance does not match distribution's var() method;
    # analytic formula from wikipedia matches computed circular variance;
    # actually, d.var() = d.std()**2
    ax[1].plot(sg, cvar, label="circ. variance")
    ax[1].plot(sg, svar, label="dist. variance")
    ax[1].plot(sg, _vonmises_circ_var(kp), label="analytic")
    # ax[1].plot(sg, 0.5 / kp, ls='--', c='gray', label='1/$2\kappa$')
    # ax[1].plot(sg, 1.0 / kp, ls='--', c='gray', label='1/$2\kappa$')
    ax[1].set_ylim([0, 1.1])

    # computed circular std matches distribution's std() method;
    ax[2].plot(sg, cstd, label="circ. std")
    ax[2].plot(sg, sstd, label="dist. std")
    ax[2].plot(sg, np.sqrt(-2 * np.log(1 - _vonmises_circ_var(kp))), label="examples")
    ax[2].axhline(y=np.pi, xmin=sg[0], xmax=sg[-1], c="gray", ls="--")
    ax[2].axhline(y=_twopi / np.sqrt(12), xmin=sg[0], xmax=sg[-1], c="gray", ls="--")
    # ax[2].plot(sg, 1 / np.sqrt(kp), ls='--', c='gray', label='$1/\sqrt{\kappa}$')

    for i in range(3):
        ax[i].grid(True)
        ax[i].legend()
    ax[2].set_xlabel("Input variance")
    return


def _common_distrib(
    dist,
    ns: int,
    ci=0.95,
    circular: bool = False,
    plot: bool = False,
    normalize: float = 1.0,
    bounds=None,
):
    # num bins
    nb = max(20, (ns + 1) // 100)
    # sample
    s = dist.rvs(ns)
    if bounds is not None:
        a = bounds[0]
        b = bounds[1]
    else:
        a = s.min()
        b = s.max()
    x = np.linspace(a, b, 201)
    try:
        y = dist.pdf(x)
    except AttributeError:
        y = np.nan * np.zeros_like(x)

    qmin = 0.5 * (1 - ci)
    qmax = 0.5 * (1 + ci)

    if plot:
        fig, ax = plt.subplots()
        ax.hist(s / normalize, bins=nb, density=True, fc="C1")
        ax.plot(x / normalize, y * normalize, "-", c="C0", label="pdf")
        ax.axvline(dist.mean() / normalize, ls="--", c="C0", label="mean")
        ax.grid(True)
        ax.set_xlim([a / normalize, b / normalize])
        if normalize == 1.0:
            ax.set_xlabel("$x$")
        else:
            ax.set_xlabel(f"$x$ (normalized by {normalize:.3f})")
        ax.grid(True)
        ax.legend()

    print(
        f"distrib: "
        f"min={a:+.3E},"
        f" max={b:+.3E},"
        f" avg={dist.mean():+3.2E}, "
        f"std={dist.std():+3.2E}, "
        f"ci={0.5 * (dist.ppf(qmax) - dist.ppf(qmin)) / dist.std():+3.2E}"
        f"({100 * ci:.0f}%, std-norm.)"
    )

    if not circular:
        print(
            f"sampled: "
            f"min={np.min(s):+.3E}, "
            f"max={np.max(s):+.3E}, "
            f"avg={np.mean(s):+.3E}, "
            f"std={np.std(s):+.3E}, "
            f"ci={0.5 * (np.quantile(s, qmax) - np.quantile(s, qmin)) / dist.std():+.3E} "
            f"({100 * ci:.0f}%, std-norm.)"
        )
    else:
        print(
            f"sampled: "
            f"min={np.min(s):+.3E}, "
            f"max={np.max(s):+.3E}, "
            f"avg={circmean(s, high=b, low=a):+.3E}, "
            f"std={circstd(s, high=b, low=a):+.3E}, "
            f"ci={0.5 * (np.quantile(s, qmax) - np.quantile(s, qmin)) / dist.std():+.3E} "
            f"({100 * ci:.0f}%, std-norm.)"
        )

    return


def example_truncnorm(a, b, mu, sigma, n=9999, ci=0.95, plot=False):
    d = truncnorm(a, b, mu, sigma)
    _common_distrib(d, n, ci=ci, circular=False, plot=plot, bounds=(a, b))
    return


def example_wrapnorm(mu, sigma, n=999, ci=0.95, plot=False):
    a = 0.0
    b = _twopi
    d = wrapnorm(mu, sigma)
    _common_distrib(d, n, ci=ci, plot=plot, normalize=np.pi, bounds=(a, b))
    return


def example_vonmises(mu, sigma, n=999, ci=0.95, plot=False):
    a = -np.pi
    b = +np.pi
    d = vonmises(mu, sigma)
    _common_distrib(d, n, ci=ci, plot=plot, normalize=np.pi, bounds=(a, b))
    return


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("TkAgg")
    plt.close("all")

    circular_moments_wrapnorm()
    circular_moments_vonmises()

    example_truncnorm(-1.0, 1.0, 0.318, 0.2, plot=True)
    example_wrapnorm(1.0, 0.2, plot=True)
    example_vonmises(1.0, 0.5, plot=True)
