import pandas as pd
import numpy as np
import time

from pyomo.core import *
from pyomo.environ import *
from sklearn.linear_model import QuantileRegressor

from data_generation import *


class MSM:

    def create_bounds(self, data: DataObject, gamma: float, approach: str) -> tuple:
        """
        Method which automatically computes the bound based on selected approach.

        :param data: DataFrame containing all the data
        :param gamma: Gamma representing the bound of the odds-ratio \(\Gamma\)
        :param approach: String representation of the approach, default is the constraint programming implementation.
        Possible approaches include "cp" constraint programming or "cvar" cvar.
        :return: Tuple representing the lower and upper bound
        """
        bound_founder = self._solve_cvar_formulation if approach == "cvar" \
            else self._solve_constraint_programming_formulation
        # Compute the bounds
        lower_control_bound = bound_founder(data, gamma, 0, True)
        lower_treated_bound = bound_founder(data, gamma, 1, True)
        upper_control_bound = bound_founder(data, gamma, 0, False)
        upper_treated_bound = bound_founder(data, gamma, 1, False)

        # Output lower and upper bounds based on best and worst cases
        return lower_treated_bound - upper_control_bound, upper_treated_bound - lower_control_bound

    def _solve_constraint_programming_formulation(self, data: DataObject, gamma: float, treatment: int,
                                                  is_lower_bound: bool) -> float:
        """
        Constraint programming formulation of the MSM based on Tan 2006. To solve this the GLPK solver is used. GLPK
        is a linear programming solver, so using it is better than a constraint programming one like ipopt.

        :param data: Data Object holding the entire dataset inside it. Used for propensity and prior calculations.
        :param gamma: Bound of the odds-ratio \(\Gamma\)
        :param treatment: Integer representing the treatment. Binary treatment is assumed.
        :param is_lower_bound: Boolean representing whether to compute upper or lower bound.
        :return: The computed bound.
        """
        X = data.discrete_x()
        Y = data.discrete_y()
        model = ConcreteModel(name="MarginalSensitivityModel")
        model.X = RangeSet(0, len(X) - 1)
        model.Y = RangeSet(0, len(Y) - 1)
        model.lam = Var(model.X, model.Y, bounds=(1 / gamma, gamma), initialize=1)

        # Constraint 1: Lambda is a distribution of Y | X, T
        def distribution_constraint(model):
            return sum([sum(model.lam[x, y] * data.probability_of_x_index_y_index_given_t(x, y, treatment)
                            for y in model.Y)
                        for x in model.X]) == 1

        model.c1 = Constraint(rule=distribution_constraint)

        # Constraint 2: Propensity scores remains unchanged
        def propensity_constraint(model):
            return sum(
                [sum([data.propensity_score_index(x) * model.lam[x, y] * data.probability_of_x_index_y_index_given_t(x,
                                                                                                                     y,
                                                                                                                     treatment)
                      for y in model.Y]) for x in model.X]) == sum(
                [sum([data.propensity_score_index(x) * data.probability_of_x_index_y_index_given_t(x, y, treatment)
                      for y in model.Y]) for x in model.X])

        model.c2 = Constraint(rule=propensity_constraint)

        # Constraint 3: Distribution of Y unchanged
        def p_constraint(model):
            return sum(
                [sum([data.probability_of_x_index_y_index_given_t(x, y, treatment) * model.lam[x, y] * X.iloc[x]["size"]
                      for x in model.X]) / sum(X["size"]) for y in model.Y]) == \
                sum([sum([data.probability_of_x_index_y_index_given_t(x, y, treatment) * X.iloc[x]["size"]
                          for x in model.X]) / sum(X["size"]) for y in model.Y])

        model.c3 = Constraint(rule=p_constraint)

        # Objective: Find the bound
        def objective_function(model):
            return sum([sum([Y.iloc[y] * model.lam[x, y] * data.probability_of_x_index_y_index_given_t(x, y, treatment)
                             for y in model.Y]) for x in model.X])

        model.OBJ = Objective(rule=objective_function,
                              sense=pyomo.core.minimize if is_lower_bound else pyomo.core.maximize)

        opt = SolverFactory('glpk')
        opt.options['tmlim'] = 100
        opt.solve(model)
        return model.OBJ()

    def _solve_cvar_formulation(self, data: DataObject, gamma: float, treatment: int, is_lower_bound: bool) -> float:
        """
        Uses CVaR formulation of the MSM problem. The Quantile Regressor used has a linear solver within which can take
        quite some time to compute.

        :param data: Data Object holding the entire dataset inside it. Used for propensity and prior calculations.
        :param gamma: Bound of the odds-ratio \(\Gamma\)
        :param treatment: Integer representing the treatment. Binary treatment is assumed.
        :param is_lower_bound: Boolean representing whether to compute upper or lower bound.
        :return: The computed bound.
        """
        tau = gamma / (gamma + 1)
        if is_lower_bound:
            tau = 1 - tau
        X = data.discrete_x()
        result = 0
        for x_index in range(len(X)):
            selection = data.select_x_index(x_index, treatment)
            x = selection[data.x_features]
            y = selection["Y"].to_numpy()
            regressor = QuantileRegressor(quantile=tau, solver='highs')
            regressor.fit(x, y)
            pred = regressor.predict(x)
            res = 1 / gamma * y + (1 - 1 / gamma) * (pred + 1 / (1 - tau) * (y - pred))
            result += res.mean() * data.probability_of_x_index_given_t(x_index, treatment)
        return result
