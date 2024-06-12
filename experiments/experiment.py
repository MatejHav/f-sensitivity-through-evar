import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_generation.Generator import Generator, DataObject
from models.FSensitivity import FSensitivity
from tqdm import tqdm
from causalml.inference.tree import CausalRandomForestRegressor
from sklearn.mixture import GaussianMixture


class Experiment:

    def __init__(self, rhos: list, generator: Generator):
        self.rhos = rhos
        self.generator = generator
        self.gd_results = {
            "upper": [],
            "lower": []
        }
        self.cp_results = {
            "upper": [],
            "lower": []
        }
        self.aa_results = {
            "upper": [],
            "lower": []
        }
        self.f_solver = FSensitivity()

    def _adjust_data_with_rf(self, data, path):
        new_path = path[:-4] + "_adjusted.csv"
        X = data.data[data.x_features]
        T = data.data["T"]
        Y = data.data["Y"].to_numpy()
        # Train a Random Forest regressor
        random_forest = CausalRandomForestRegressor()
        random_forest.fit(X=X, treatment=T, y=Y)
        # Store predictions for every row
        copy_of_data = pd.DataFrame([], columns=data.data.columns)
        predictions = random_forest.predict(X, with_outcomes=True)
        for index, row in data.data.iterrows():
            u = row['U0']
            x = row['X0']
            t = row['T0']
            y = row['Y0']
            # Store counterfactuals based on predictions, otherwise keep the observed data
            y0 = predictions[index][0] if t == 1 else y
            y1 = predictions[index][1] if t == 0 else y
            # Round the predictions to 1 decimal place
            copy_of_data.loc[len(copy_of_data)] = [u, x, 0, round(y0, 1)]
            copy_of_data.loc[len(copy_of_data)] = [u, x, 1, round(y1, 1)]
        copy_of_data.to_csv(new_path)
        # Use adjusted data
        return DataObject(copy_of_data)

    def run(self, path: str, num_rows: int = 30_000, n_jobs: int = 30, adjust_with_RF: bool = False):
        data = self.generator.generate(num_rows, n_jobs, path)
        # Adjust with random forest
        if adjust_with_RF:
            data = self._adjust_data_with_rf(data, path)

        # Iterate over all possible rhos
        for rho in tqdm(self.rhos):
            cp_lower, cp_upper = self.f_solver.create_bounds(data, rho, approach="cp")
            gd_lower, gd_upper = self.f_solver.create_bounds(data, rho, approach="evar")
            aa_lower, aa_upper = self.f_solver.create_bounds(data, rho, approach="lagr")
            self.cp_results["lower"].append(cp_lower)
            self.gd_results["lower"].append(gd_lower)
            self.aa_results["lower"].append(aa_lower)
            self.cp_results["upper"].append(cp_upper)
            self.gd_results["upper"].append(gd_upper)
            self.aa_results["upper"].append(aa_upper)
        # Plot the results
        # Constraints programming
        plt.plot(self.rhos, self.cp_results["lower"], color='red')
        plt.plot(self.rhos, self.cp_results["upper"], color='red')
        plt.xlabel('ρ')
        plt.ylabel('ATE')
        plt.title('Bound from constraint programing approach')
        plt.show()
        # EVaR approach
        plt.plot(self.rhos, self.gd_results["lower"], color='blue')
        plt.plot(self.rhos, self.gd_results["upper"], color='blue')
        plt.xlabel('ρ')
        plt.ylabel('ATE')
        plt.title('Bound from gradient descent approach')
        # plt.title('Bound from gradient descent approach with a Gaussian Mixture model')
        plt.show()
        # Authors algorithm
        plt.plot(self.rhos, self.aa_results["lower"], color='green')
        plt.plot(self.rhos, self.aa_results["upper"], color='green')
        plt.xlabel('ρ')
        plt.ylabel('ATE')
        plt.title('Bound from authors approach')
        plt.show()

    def run_gaussian_mixture(self, path: str, num_rows: int = 30_000, n_jobs: int = 30):
        data = self.generator.generate(num_rows, n_jobs, path)
        # Train Gaussian Mixture model, assuming the experimental setup with 1 binary confounder
        k = 1
        treated_data0 = data.data[(data.data['X0'] == 0) & (data.data['T'] == 1)]
        treated_gauss0 = GaussianMixture(n_components=k, covariance_type='spherical')
        treated_gauss0.fit(treated_data0['Y'].to_numpy().reshape(-1, 1))
        treated_data1 = data.data[(data.data['X0'] == 1) & (data.data['T'] == 1)]
        treated_gauss1 = GaussianMixture(n_components=k, covariance_type='spherical')
        treated_gauss1.fit(treated_data1['Y'].to_numpy().reshape(-1, 1))
        control_data0 = data.data[(data.data['X0'] == 0) & (data.data['T'] == 0)]
        control_gauss0 = GaussianMixture(n_components=k, covariance_type='spherical')
        control_gauss0.fit(control_data0['Y'].to_numpy().reshape(-1, 1))
        control_data1 = data.data[(data.data['X0'] == 1) & data.data['T'] == 0]
        control_gauss1 = GaussianMixture(n_components=k, covariance_type='spherical')
        control_gauss1.fit(control_data1['Y'].to_numpy().reshape(-1, 1))
        means = {
            'treated': [treated_gauss0.means_[0], treated_gauss1.means_[0]],
            'control': [control_gauss0.means_[0], control_gauss1.means_[0]]
        }
        variances = {
            'treated': [treated_gauss0.covariances_, treated_gauss1.covariances_],
            'control': [control_gauss0.covariances_, control_gauss1.covariances_]
        }
        # Generate data based on learned gaussians
        copy_of_data = pd.DataFrame([], columns=data.data.columns)
        for index, row in data.data.iterrows():
            u = row['U0']
            x = int(row['X0'])
            t = row['T0']
            y = row['Y0']
            y0 = np.sum(variances["control"][x] * np.random.randn(k) + means["control"][x]) if t == 1 else y
            y1 = np.sum(variances["treated"][x] * np.random.randn(k) + means["treated"][x]) if t == 0 else y
            copy_of_data.loc[len(copy_of_data)] = [u, x, 0, round(y0, 3)]
            copy_of_data.loc[len(copy_of_data)] = [u, x, 1, round(y1, 3)]
        copy_of_data.to_csv(path[:-4] + "_gauss_adjusted.csv")
        data = DataObject(copy_of_data)
        # Iterate over all possible rhos
        res_lower = []
        res_upper = []
        cp_lower = []
        cp_upper = []
        for rho in tqdm(self.rhos):
            lower, upper = self.f_solver.create_bounds(data, rho, approach="cp")
            cp_upper.append(upper)
            cp_lower.append(lower)
            lower = self.f_solver.solve_gaussian_mixture_model(data, rho, True, means, variances, k)
            upper = self.f_solver.solve_gaussian_mixture_model(data, rho, False, means, variances, k)
            res_lower.append(lower)
            res_upper.append(upper)
        plt.plot(self.rhos, res_upper, color='red')
        plt.plot(self.rhos, res_lower, color='red')
        plt.xlabel('ρ')
        plt.ylabel('ATE')
        plt.title('Bound from closed form approach with a Gaussian mixture model')
        plt.show()
        plt.plot(self.rhos, self.cp_results["lower"], color='red')
        plt.plot(self.rhos, self.cp_results["upper"], color='red')
        plt.xlabel('ρ')
        plt.ylabel('ATE')
        plt.title('Bound from constraint programing approach used on generated data from Gaussian model')
        plt.show()
