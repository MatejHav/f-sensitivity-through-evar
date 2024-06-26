import numpy as np
import matplotlib.pyplot as plt


def prior_experiment():
    """Experiment about how each model reacts to a prior of the hidden confounder"""
    e0 = 0.1
    e1 = 0.5
    # Define the scenario applied on the experiment
    propensity = lambda p: e0 * (1 - p) + e1 * p
    odds_ratio = lambda e, p: (e / (1 - e)) / (propensity(p) / (1 - propensity(p)))
    # Define the two parts of the f-sensitivity bound
    f = lambda p: odds_ratio(e0, p) * np.log(odds_ratio(e0, p)) * e0 * (1 - p) + odds_ratio(e1, p) * np.log(
        odds_ratio(e1, p)) * e1 * p
    f_inv = lambda p: 1 / odds_ratio(e0, p) * np.log(1 / odds_ratio(e0, p)) * (1 - e0) * (1 - p) + 1 / odds_ratio(e1,
                                                                                                                  p) * np.log(
        1 / odds_ratio(e1, p)) * (1 - e1) * p
    ps = np.linspace(0, 1, 100)
    msm = []
    f_sensitivity = []
    # For every prior compute the actual bound
    for p in ps:
        f_sensitivity.append(max(f(p), f_inv(p)))
        msm.append(max(odds_ratio(e0, p), odds_ratio(e1, p), 1 / odds_ratio(e0, p), 1 / odds_ratio(e1, p)))
    # Plot the bound against the priors
    plt.plot(ps, msm)
    plt.xlabel('P(U=1)')
    plt.ylabel('Gamma')
    plt.title('MSM bound dependent on hidden confounder distribution')
    plt.show()
    plt.plot(ps, f_sensitivity)
    plt.xlabel('P(U=1)')
    plt.ylabel('Rho')
    plt.title('F-sensitivity bound dependent on hidden confounder distribution')
    plt.show()


def video_figure():
    """
    Creates figures of how the ball of possible distributions selected by f-sensitivity and MSM would look like,
    when we try to find a normal distribution with a standard normal distribution observed.
    """
    font = {"size": 18}
    rhos = [0.1, 0.5, 1, 2, 4]
    lambdas = [1.3, 1.7, 2.5, 4, 5]
    for i, rho in enumerate(rhos):
        f = lambda x, y: 0.5 * (x ** 2 / (y ** 2) + 1 / (y ** 2) - np.log(1 / (y ** 2)) - 1) <= rho
        g = lambda x, y: 0.5 * (x ** 2 + y ** 2 - np.log(y ** 2) - 1) <= rho

        dx = np.linspace(-4, 4, 2000)
        dy = np.linspace(0.001, 6.001, 2000)
        x, y = np.meshgrid(dx, dy)

        plt.imshow((f(x, y) & g(x, y)).astype(int),
                   extent=(x.min(), x.max(), y.min(), y.max()), origin="lower", cmap="Greys")
        plt.xlabel('mu', fontdict=font)
        plt.ylabel('sigma', fontdict=font)
        plt.title(f'Selection based on f-sensitivity with rho={rho}', fontdict=font)
        plt.savefig(f"rho{i + 1}.png")
        plt.show()

    for i, l in enumerate(lambdas):
        f = lambda x, y, w: 1 / y * np.exp(-0.5 * (((w - x) / y) ** 2 - w ** 2)) <= l
        g = lambda x, y, w: 1 / l <= 1 / y * np.exp(-0.5 * (((w - x) / y) ** 2 - w ** 2))

        dx = np.linspace(-4, 4, 2000)
        dy = np.linspace(0.001, 6.001, 2000)
        x, y = np.meshgrid(dx, dy)
        ws = np.linspace(-2.5, 2.5, 100)

        res = [f(x, y, w) & g(x, y, w) for w in ws]

        plt.imshow((np.bitwise_and.reduce(res, 0)).astype(int),
                   extent=(x.min(), x.max(), y.min(), y.max()), origin="lower", cmap="Greys")

        plt.xlabel('mu', fontdict=font)
        plt.ylabel('sigma', fontdict=font)
        plt.title(f'Selection based on the MSM with gamma={l}', fontdict=font)
        plt.savefig(f"lambda{i + 1}.png")
        plt.show()
