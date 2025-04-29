import numpy as np
import pandas as pd
from algorithmic_attribute_classification import AttributeClassification

import numpy as np
import pandas as pd

class Uncertainty:

    def __init__(self, data, SAs):
        self.data = data
        self.SAs = SAs

    def entropy(self, series):
        counts = series.value_counts(normalize=True)
        return -np.sum(counts * np.log2(counts))

    def max_entropy(self, series):
        n = series.nunique()
        return np.log2(n) if n > 0 else 0

    def normalized_entropy(self, series):
        h = self.entropy(series)
        h_max = self.max_entropy(series)
        return h / h_max if h_max != 0 else 0

    def min_entropy(self, series):
        p_max = series.value_counts(normalize=True).max()
        return -np.log2(p_max) if p_max > 0 else 0

    def uncertainty_calculate_all(self):
        if not self.SAs:
            print("No sensitive attributes found.")
            return 0, 0, 0

        entropy_list = []
        min_entropy_list = []
        normalized_entropy_list = []

        for sa in self.SAs:
            series = self.data[sa]
            entropy_list.append(self.entropy(series))
            min_entropy_list.append(self.min_entropy(series))
            normalized_entropy_list.append(self.normalized_entropy(series))

        avg_entropy = np.mean(entropy_list)
        avg_min_entropy = np.mean(min_entropy_list)
        avg_normalized_entropy = np.mean(normalized_entropy_list)

        return avg_entropy, avg_min_entropy, avg_normalized_entropy

if __name__ == "__main__":
    FILE_PATH = "./data/test/healthcare_dataset.csv"
    DATA = pd.read_csv(FILE_PATH)

    attributes = AttributeClassification(DATA, FILE_PATH).run_on_csv()
    SAs = attributes['SAs']

    uncertainty = Uncertainty(DATA, SAs)
    avg_entropy, avg_min_entropy, avg_normalized_entropy = uncertainty.uncertainty_calculate_all()

    print("Average Entropy:", avg_entropy)
    print("Average Min Entropy:", avg_min_entropy)
    print("Average Normalized Entropy:", avg_normalized_entropy)
