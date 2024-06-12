import pandas as pd
import numpy as np
import torch
import time

from pyomo.core import *
from pyomo.environ import *
from sklearn.linear_model import QuantileRegressor

from data_generation import *

class FSensitivity:

    def create_bounds(self, data: DataObject, rho: float, approach: str) -> tuple:
        """

        :param data:
        :param rho:
        :param approach: "evar" for gradient descent, "lagr" for lagrangian and "cp" for contraints programming
        :return:
        """
        bound_founder = self._solve_evar_formulation if approach == "evar" \
            else (self._solve_lagrangian_formulation if approach == "lagr"
                  else self._solve_constraint_programming_formulation)
        # Compute the bounds
        lower_control_bound = bound_founder(data, rho, 0, True)
        lower_treated_bound = bound_founder(data, rho, 1, True)
        upper_control_bound = bound_founder(data, rho, 0, False)
        upper_treated_bound = bound_founder(data, rho, 1, False)

        # Output lower and upper bounds based on best and worst cases
        if approach == "lagr":
            mu_0_obs = data.data[data["T"] == 0]["Y"].mean()
            mu_1_obs = data.data[data["T"] == 1]["Y"].mean()
            return (mu_1_obs - upper_treated_bound + mu_0_obs - lower_control_bound,
                    mu_1_obs - lower_treated_bound + mu_0_obs - upper_control_bound)
        return lower_treated_bound - upper_control_bound, upper_treated_bound - lower_control_bound

    def _solve_constraint_programming_formulation(self, data: DataObject, rho: float, treatment: int,
                                                  is_lower_bound: bool) -> float:
        """

        :param data:
        :param rho:
        :param treatment:
        :param is_lower_bound:
        :return:
        """
        X = data.discrete_x()
        Y = data.discrete_y()
        p_treated = len(data.data["T"] == 1) / len(data.data)
        r = lambda x: data.propensity_score_index(x) * (1 - p_treated) / (1 - data.propensity_score_index(x) * p_treated)
        model = ConcreteModel(name="FSensitivityModel")
        model.X = RangeSet(0, len(X) - 1)
        model.Y = RangeSet(0, len(Y) - 1)
        model.L = Var(model.X, model.Y, initialize=1, within=NonNegativeReals)

        # Constraint 1: Definition of R
        def r_constraint(model, x):
            if treatment == 0:
                return (1 / r(x)) == sum([model.L[x, y] * data.probability_of_x_index_y_index_given_t(x, y, treatment) for y in model.Y])
            return r(x) == sum([model.L[x, y] * data.probability_of_x_index_y_index_given_t(x, y, treatment) for y in model.Y])

        model.c1 = Constraint(model.X, rule=r_constraint)

        # Constraint 3: MSM assumption
        def f_constraint(model, x):
            if treatment == 0:
                return sum([
                    log(model.L[x, y] * r(x)) * model.L[x, y] * r(x) * data.probability_of_x_index_y_index_given_t(x, y, treatment)
                for y in model.Y]) <= rho
            return sum([
                log(model.L[x, y] / r(x)) * model.L[x, y] / r(x) * data.probability_of_x_index_y_index_given_t(x, y, treatment)
            for y in model.Y]) <= rho

        model.c3 = Constraint(model.X, rule=f_constraint)

        # Objective: E[Y * L(X, Y)]
        def objective_function(model):
            return sum([sum([Y.iloc[y] * model.L[x, y] * data.probability_of_x_index_y_index_given_t(x, y, treatment)
                             for y in model.Y]) for x in model.X])

        model.OBJ = Objective(rule=objective_function,
                              sense=pyomo.core.minimize if is_lower_bound else pyomo.core.maximize)

        opt = SolverFactory('ipopt')
        opt.options['max_iter'] = 1000
        opt.solve(model)
        return model.OBJ()

    def __entropic_value_at_risk(self, exponents: torch.Tensor, alpha: float, is_lower_bound: bool) -> float:
        eps = 1e-6
        N = len(exponents)
        z = torch.nn.Parameter(torch.tensor([-1.0 if is_lower_bound else 1.0], requires_grad=True, dtype=torch.double))
        helper = lambda: (torch.logsumexp(z * exponents, 0) - np.log(N * alpha)) / z
        optimizer = torch.optim.Adam([z], lr=1e-3, maximize=is_lower_bound, weight_decay=0)
        previous = None
        loss = None
        # Until converged
        while previous is None or abs((previous - loss).item()) > eps:
            if (z.item() < 0 and not is_lower_bound) or (z.item() > 0 and is_lower_bound):
                return min(exponents.max().item(), previous.item()) if not is_lower_bound else \
                    max(exponents.min().item(), previous.item())
            previous = loss
            loss = helper()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return min(exponents.max().item(), helper().item()) if not is_lower_bound else \
            max(exponents.min().item(), helper().item())


    def _solve_evar_formulation(self, data: DataObject, rho: float, is_lower_bound: bool) -> float:
        """

        :param data:
        :param rho:
        :param treatment:
        :param is_lower_bound:
        :return:
        """
        alpha = np.exp(-rho)
        X = data.discrete_x()
        p_treated = len(data.data["T"] == 1) / len(data.data)
        result_treated = 0
        result_control = 0
        for index, row in X.iterrows():
            observed_propensity = data.propensity_score(row)
            r_x = (1 - p_treated) * observed_propensity / (p_treated * (1 - observed_propensity))
            # treated
            exponents = torch.DoubleTensor(data.select_x(row, 1)["Y"].to_numpy())
            treated_q = self.__entropic_value_at_risk(exponents, alpha, is_lower_bound)
            result_treated += r_x * data.probability_of_x_index_given_t(index, 1) * treated_q
            # control
            exponents = torch.DoubleTensor(data.select_x(row, 0)["Y"].to_numpy())
            # print("\ncontrol")
            control_q = self.__entropic_value_at_risk(exponents, alpha, not is_lower_bound)
            # print(control_q)
            result_control += 1 / r_x * data.probability_of_x_index_given_t(index, 0) * control_q
        return result_treated - result_control

    def _solve_lagrangian_formulation(self, data: DataObject, rho: float, treatment: int, is_lower_bound: bool) -> float:
        """

        :param data:
        :param rho:
        :param treatment:
        :param is_lower_bound:
        :return:
        """
        # Shuffle the data
        shuffled_data = data.data.sample(frac=1)
        data_splits = np.array_split(shuffled_data, 3)
        bounds = [0, 0, 0]
        eps = 0
        x_features = list(filter(lambda c: 'X' in c, shuffled_data.columns))
        layer_size = len(x_features)
        # For every subset of data
        for i in range(3):
            current_data = data_splits[i]
            next_data = data_splits[(i + 1) % 3]
            next_next_data = data_splits[(i + 2) % 3]
            selected_curr_data = current_data[current_data["T0"] == treatment]
            selected_next_data = next_data[next_data["T0"] == treatment]
            selected_next_next_data = next_next_data[next_next_data["T0"] == treatment]
            # Estimate r(x) based on i+1 dataset
            p = len(next_data[next_data["T0"] == treatment]) / len(next_data)
            X = next_data.groupby(x_features, as_index=False).size()
            Xt = selected_next_data.groupby(x_features, as_index=False).size()
            # Assign each sample a r(x) value
            r = torch.nn.Sequential(torch.nn.Linear(layer_size, layer_size), torch.nn.ReLU(),
                                    torch.nn.Linear(layer_size, 1))
            r_optim = torch.optim.Adam(r.parameters(), weight_decay=1e-3)
            criterion = torch.nn.MSELoss()
            for _ in range(500):
                batch = selected_next_data.sample(frac=0.2)
                x = torch.Tensor(batch[x_features].to_numpy())
                selected_x = (batch[x_features].join(X, on=x_features, how='inner', lsuffix='l', rsuffix='r')[
                    [*x_features, "size"]]).join(Xt, on=x_features, how='inner', lsuffix='l', rsuffix='r')[
                    [*x_features, "sizel", "sizer"]]
                propensity = torch.Tensor((selected_x["sizer"] / selected_x["sizel"]).to_numpy())
                r_optim.zero_grad()
                target = (1 - propensity) * p / ((1 - p) * propensity)
                prediction = r(x)
                loss = criterion(prediction.squeeze(dim=-1), target)
                loss.backward()
                r_optim.step()
            # Estimate the nuisance parameters using a NN
            alpha_model = torch.nn.Sequential(torch.nn.Linear(layer_size, layer_size), torch.nn.ReLU(),
                                              torch.nn.Linear(layer_size, 1))
            eta_model = torch.nn.Sequential(torch.nn.Linear(layer_size, layer_size), torch.nn.ReLU(),
                                            torch.nn.Linear(layer_size, 1))
            alpha_optim = torch.optim.Adam(alpha_model.parameters(), weight_decay=1e-3)
            eta_optim = torch.optim.Adam(eta_model.parameters(), weight_decay=1e-3)
            # Do 200 steps on randomized batches from i+1 data
            for _ in range(10_000):
                batch = selected_next_data.sample(frac=0.2)
                x = torch.Tensor(batch[x_features].to_numpy())
                y = torch.Tensor(batch["Y0"].to_numpy()).unsqueeze(dim=-1)
                alpha_optim.zero_grad()
                eta_optim.zero_grad()
                alpha = alpha_model(x) ** 2 + 0.1
                eta = eta_model(x)
                loss = torch.mean(alpha * torch.exp((y + eta) / (-alpha - eps) - 1) + eta + alpha * rho)
                loss.backward()
                alpha_optim.step()
                eta_optim.step()
            # Use regression to estimate H(X, Y) given X from i+2 dataset
            regressor = torch.nn.Linear(layer_size, 1)
            regressor_optim = torch.optim.Adam(regressor.parameters(), weight_decay=1e-3)
            criterion = torch.nn.MSELoss()
            for _ in range(10_000):
                batch = selected_next_next_data.sample(frac=0.2)
                x = torch.Tensor(batch[x_features].to_numpy())
                y = torch.Tensor(batch["Y0"].to_numpy()).unsqueeze(dim=-1)
                with torch.no_grad():
                    alpha = alpha_model(x) ** 2 + 0.1
                    eta = eta_model(x)
                regressor_optim.zero_grad()
                prediction = regressor(x)
                target = alpha * torch.exp((y + eta) / (-alpha - eps) - 1) + eta + alpha * rho
                loss = criterion(prediction, target)
                loss.backward()
                regressor_optim.step()
            # Compute the expected bound using H(X,Y) and h(X)
            x = torch.Tensor(selected_curr_data[x_features].to_numpy())
            y = torch.Tensor(selected_curr_data["Y0"].to_numpy()).unsqueeze(dim=-1)
            mean_regressor = torch.mean(regressor(x).detach())
            alpha = alpha_model(x) ** 2 + 0.1
            eta = eta_model(x)
            mean_diff = torch.mean(
                r(x) * (alpha * torch.exp((y + eta) / (-alpha - eps) - 1) + eta + alpha * rho - regressor(x)))
            bounds[i] = (mean_diff + mean_regressor).detach().item()
        # Return average of the three estimated bounds
        return -np.mean(bounds) if is_lower_bound else np.mean(bounds)


    def solve_gaussian_mixture_model(self, data: DataObject, rho: float, is_lower_bound: bool, means: dict, variances:dict, k: int) -> float:
        X = data.discrete_x()
        p_treated = len(data.data[data.data["T"] == 1]) / len(data.data)
        result_treated = 0
        result_control = 0
        for index, row in X.iterrows():
            x = int(row[data.x_features].to_numpy()[0])
            observed_propensity = data.propensity_score_index(index)
            r_x = (1 - p_treated) * observed_propensity / (p_treated * (1 - observed_propensity))
            # treated
            treated_q = (-1 if is_lower_bound else 1) * 0.5 * np.sqrt(rho * sum(variances['treated'][x]) / (2 * k ** 2)) \
                        + sum(means['treated'][x]) / k \
                        + (-1 if is_lower_bound else 1) * (np.sqrt(0.5 * rho)) / (2 * k)
            result_treated += r_x * data.probability_of_x_index_given_t(index, 1) * treated_q
            # control
            control_q = (-1 if not is_lower_bound else 1) * 0.5 * np.sqrt(
                rho * sum(variances['control'][x]) / (2 * k ** 2)) \
                        + sum(means['control'][x]) / k \
                        + (-1 if not is_lower_bound else 1) * (np.sqrt(0.5 * rho)) / (2 * k)
            result_control += 1 / r_x * data.probability_of_x_index_given_t(index, 0) * control_q
        return result_treated - result_control