import threading
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_generation import DataObject


class Generator:
    """
    Class to generate causal data with a hidden confounder.
    """

    def __init__(self, generators: dict, noise_generators: dict, sizes: dict):
        assert np.all([c in generators and c in noise_generators and c in sizes for c in ["U", "X", "T", "Y"]]), "All inputs require U, X, T and Y"
        self.size = sizes
        self.generators = generators
        self.noise_generators = noise_generators

    def generate(self, num_rows: int, n_jobs: int, path: str, verbose: int=0) -> DataObject:
        """
        Generate data based on generators provided to the class.
        :param num_rows: Number of samples.
        :param n_jobs: Number of threads to paralelize the process. Should be smaller than num_rows.
        :param path: Path to save the dataframe in. If it already exists, it will be overwritten.
        :param verbose: Include tqdm progress bars.
        :return: DataObject holding the dataframe.
        """
        assert num_rows <= num_rows >= n_jobs, "Number of samples must be greater or equal than number of jobs."
        data = []
        def _generator_helper(k):
            bar = range(k)
            if verbose > 0:
                bar = tqdm(bar)
            for _ in bar:
                U = self.generators["U"](self.noise_generators["U"]())
                X = self.generators["X"](U, self.noise_generators["X"]())
                T = self.generators["T"](U, X, self.noise_generators["T"]())
                Y = self.generators["Y"](U, X, T, self.noise_generators["Y"]())
                data.append([*U, *X, *T, *Y])

        threads = []
        for i in range(n_jobs):
            thread = threading.Thread(target=_generator_helper, args=[num_rows // n_jobs])
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        df = pd.DataFrame(data, columns=[*[f"U{i}" for i in range(self.size["U"])],
                                         *[f"X{i}" for i in range(self.size["X"])],
                                         "T", "Y"])
        df.to_csv(path, index=False, columns=df.columns)
        return DataObject(df), path
