import pandas as pd
import numpy as np

from functools import cache

class DataObject:
    """
    Object holding data. Comes with often called methods about propensity and probabilities. The results of each call
    are stored in cache to increase efficiency.
    """

    def __init__(self, data: pd.DataFrame):
        # Save dataframe
        self.data = data
        # Save different names of columns for quick use
        self.x_features = list(filter(lambda col: "X" in col, data.columns))
        self.xt_features = self.x_features.copy()
        self.xt_features.append("T")
        self.xy_features = self.x_features.copy()
        self.xy_features.append("Y")
        # Save discrete sets of X and Y
        self.X = data.groupby(self.x_features, as_index=False).size()
        self.Y = data.groupby(["Y"], as_index=False).size()
        self.XY = data.groupby(self.xy_features, as_index=False).size()
        # Discrete X for each observed treatment
        self.X_0 = data[data["T"] == 0].groupby(self.x_features, as_index=False).size()
        self.X_1 = data[data["T"] == 1].groupby(self.x_features, as_index=False).size()
        # Precompute propensity scores
        data_frequency = data.groupby(self.x_features, as_index=False).size()
        treated_data_frequency = data[data["T"] == 1].groupby(self.x_features, as_index=False).size()
        self.propensity_scores = pd.merge(data_frequency, treated_data_frequency, on=self.x_features, how='left')
        self.propensity_scores.fillna(0, inplace=True)  # Fill non-treated X
        self.propensity_scores["propensity_score"] = self.propensity_scores["treated_size"] / self.propensity_scores[
            "size"]

    def discrete_x(self) -> pd.DataFrame:
        return self.X[self.x_features]

    def discrete_y(self) -> pd.Series:
        return self.Y["Y"]

    def get_all_x(self) -> pd.DataFrame:
        return self.data[self.x_features]

    def get_all_y(self) -> pd.Series:
        return self.data["Y"]

    @cache
    def select_x(self, x: pd.Series, treatment: int) -> pd.DataFrame:
        selection = self.data[
            np.all(self.data[self.x_features] == x[self.x_features], axis=1) &
            self.data["T"] == treatment
        ]
        return selection

    def select_x_index(self, x_index: int, treatment: int) -> pd.DataFrame:
        x = self.X.iloc[x_index]
        return self.select_x(x, treatment)

    @cache
    def probability_of_x(self, x: pd.Series) -> float:
        selection = self.X[np.all(self.X[self.x_features] == x[self.x_features], axis=1)]
        assert len(selection) > 0, f"No features with values {x} exits."
        return selection["size"] / sum(self.X["size"])

    def probability_of_x_index(self, x_index: int) -> float:
        x = self.X.iloc[x_index]
        return self.probability_of_x(x)

    @cache
    def probability_of_x_given_t(self, x: pd.Series, treatment: int) -> float:
        assert treatment == 0 or treatment == 1, f"Binary treatment is assumed, but {treatment} passed as treatment."
        if treatment == 0:
            selection = self.X_0[np.all(self.X_0[self.x_features] == x[self.x_features], axis=1)]
            if len(selection) == 0:
                return 0.0
            return selection["size"] / sum(self.X_0["size"])
        selection = self.X_1[np.all(self.X_1[self.x_features] == x[self.x_features], axis=1)]
        if len(selection) == 0:
            return 0.0
        return selection["size"] / sum(self.X_1["size"])

    def probability_of_x_index_given_t(self, x_index: int, treatment: int) -> float:
        x = self.X.iloc[x_index]
        return self.probability_of_x_given_t(x, treatment)
    @cache
    def propensity_score(self, x: pd.Series) -> float:
        return self.propensity_scores[
            np.all(self.propensity_scores[self.x_features] == x[self.x_features], axis=1)
        ]["propensity_score"].iloc[0]

    def propensity_score_index(self, x_index: int) -> float:
        x = self.X.iloc[x_index]
        return self.propensity_score(x)

    @cache
    def probability_of_x_y_given_t(self, x: pd.Series, y: pd.Series, treatment: int) -> float:
        given_t = self.XY[self.XY["T"] == treatment]
        selection = self.XY[
            np.all(self.XY[self.x_features] == x[self.x_features], axis=1) &
            np.all(self.XY[["Y"]] == y[["Y"]], axis=1) &
            np.all(self.XY["T"] == treatment)
        ]
        if len(selection) == 0:
            return 0.0
        return selection["size"] / sum(given_t["size"])

    def probability_of_x_index_y_index_given_t(self, x_index: int, y_index: int, treatment: int) -> float:
        x = self.X.iloc[x_index]
        y = self.Y.iloc[y_index]
        return self.probability_of_x_y_given_t(x, y, treatment)