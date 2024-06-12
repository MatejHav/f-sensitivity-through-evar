from tqdm import tqdm

from data_generation.Generator import Generator
from experiments.experiment import Experiment
from experiments.reproduction import run_reproduction
from experiments.comparison import video_figure, prior_experiment
from models import *
from data_generation import *


def create_default_generator(u_prob, x_effect, t_effect, y_effect, dim=1, base_x_prob=0.45, base_t_prob=0.5):
    path = f"./csv_files/data_u{int(100 * u_prob)}_x{int(100 * x_effect)}_t{int(100 * t_effect)}_y{int(100 * y_effect)}_k{dim}.csv"
    # General settings
    sizes = {
        "U": 1,
        "X": dim,
        "T": 1,
        "Y": 1
    }

    # Generators
    u_gen = lambda noise: [0 if np.random.rand() >= u_prob else 1]
    x_gen = lambda u, noise: [0 if np.random.rand() >= x_effect * u[0] + base_x_prob else 1 for _ in range(sizes["X"])]
    t_gen = lambda u, x, noise: [0 if np.random.rand() >= t_effect * u[0] + 0.25 * sum(x) / dim + base_t_prob else 1]
    y_gen = lambda u, x, t, noise: [round(sum(x) / dim + y_effect * u[0] + 2 * t[0] + noise, 1)]
    generators = {
        "U": u_gen,
        "X": x_gen,
        "T": t_gen,
        "Y": y_gen
    }
    # Noise generators
    noise = {
        "U": lambda: 0,
        "X": lambda: 0,
        "T": lambda: 0,
        "Y": lambda: 0.1 * np.random.randn() - 1
    }

    generator = Generator(generators=generators, noise_generators=noise, sizes=sizes)
    return generator, path


def example1():
    """Comparison between CP, GD and Lagrangian approaches"""
    rhos = np.linspace(0, 1.3, 13)
    n_rows = 30_000
    n_jobs = 30
    data_generator, path = create_default_generator(0.25, 0, -0.3, 1, dim=1)
    experiment = Experiment(rhos, generator=data_generator)
    experiment.run(path=path, n_jobs=n_jobs, num_rows=n_rows, adjust_with_RF=True)


def example2():
    """Gaussian Mixture model results"""
    rhos = np.linspace(0, 1.3, 13)
    n_rows = 30_000
    n_jobs = 30
    data_generator, path = create_default_generator(0.25, 0, -0.3, 1, dim=1)
    experiment = Experiment(rhos, generator=data_generator)
    experiment.run_gaussian_mixture(path=path, n_jobs=n_jobs, num_rows=n_rows)


def reproduction():
    """Reproduction of the original paper experiment"""
    run_reproduction()


def possible_distributions_visualization():
    """Visualization of the possible distributions for F Sensitivity and the MSM with a normal distribution"""
    video_figure()


def sensitive_to_prior_experiment():
    """Creates a figure used to check how models react to prior of U"""
    prior_experiment()


def dimensionality_experiment():
    """Experiment with multiple confounders"""
    rhos = np.linspace(0, 1.3, 13)
    n_rows = 30_000
    n_jobs = 30
    data_generator, path = create_default_generator(0.25, 0, -0.3, 1, dim=3)
    experiment = Experiment(rhos, generator=data_generator)
    experiment.run(path=path, n_jobs=n_jobs, num_rows=n_rows, adjust_with_RF=True)


def range_experiment():
    """Experiment with a range of values tested on the example 1 experiment. Prints the table in LaTex form."""
    rhos = np.linspace(0, 1, 3)
    ps = [0.05, 0.15, 0.95]
    xs = [0.05, 0.15, 0.95]
    ts = [-0.25, 0.05, 0.15]
    ys = [-0.5, -0.25, 0.05, 0.15, 0.95]
    bar = tqdm(range(len(ps) * len(xs) * len(ts) * len(ys)))
    res = np.zeros((4, 5))
    solver = FSensitivity()
    for i_p, p in enumerate(ps):
        for i_x, x_effect in enumerate(xs):
            for i_t, t_effect in enumerate(ts):
                for i_y, y_effect in enumerate(ys):
                    generator, path = create_default_generator(p, x_effect, t_effect, y_effect, dim=1, base_x_prob=0)
                    data = generator.generate(num_rows=30_000, n_jobs=30, path=path)
                    differences = []
                    for rho in rhos:
                        cp_lower, cp_upper = solver.create_bounds(data, rho, approach="cp")
                        # Gradient Descent f sensitivity
                        gd_lower, gd_upper = solver.create_bounds(data, rho, approach="evar")
                        differences.append(abs(cp_upper - gd_upper))
                        differences.append(abs(cp_lower - gd_lower))
                    max_diff = max(differences)
                    # Probability of U
                    res[0, i_p + 2] = round(max(max_diff, res[0, i_p + 2]), 3)
                    # Probability of X
                    res[1, i_x + 2] = round(max(max_diff, res[1, i_x + 2]), 3)
                    # Probability of T
                    res[2, i_t + 1] = round(max(max_diff, res[2, i_t + 1]), 3)
                    # Outcome
                    res[3, i_y] = round(max(max_diff, res[3, i_y]), 3)
                    bar.update()
    print(f"""
        & -0.5 & -0.25 & 0.01 & 0.5 & 0.99 \\
         u_p & - & - & {res[0, 2]} & {res[0, 3]} & {res[0, 4]}\\
         x_u & - & - & {res[1, 2]} & {res[1, 3]} & {res[1, 4]}\\
         t_u & - & {res[2, 1]} & {res[2, 2]} & {res[2, 3]} & -\\
         y_u & {res[3, 0]} & {res[3, 1]} & {res[3, 2]} & {res[3, 3]} & {res[3, 4]}\\
        """)


if __name__ == '__main__':
    example1()
    example2()
    reproduction()
    possible_distributions_visualization()
    dimensionality_experiment()
    range_experiment()
