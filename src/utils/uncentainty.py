import numpy as np
import pandas as pd
from algorithmic_attribute_classification import AttributeClassification


class Uncertainty:

    def __init__(self, data, SA):
        self.data = data
        self.series = SA


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



if __name__ == "__main__":

    FILE_PATH = "./data/test/healthcare_dataset.csv"
    DATA = pd.read_csv(FILE_PATH)

    attributes =  AttributeClassification(DATA, FILE_PATH).run_on_csv()
    SA = attributes['SAs']

    uncertainty= Uncertainty(DATA, SA)

    entropy_list = []
    min_entropy_list = []
    normalized_entropy_list = []

    if not SA:
        avg_entropy = 0
        avg_min_entropy = 0
        avg_normalized_entropy = 0

    else:
        for sa in attributes['SAs']:
            series = DATA[sa]
            entropy_list.append(uncertainty.entropy(series))
            min_entropy_list.append(uncertainty.min_entropy(series))
            normalized_entropy_list.append(uncertainty.normalized_entropy(series))

        avg_entropy = np.mean(entropy_list)
        avg_min_entropy = np.mean(min_entropy_list)
        avg_normalized_entropy = np.mean(normalized_entropy_list)

    print("Average Entropy:", avg_entropy)
    print("Average Min Entropy:", avg_min_entropy)
    print("Average Normalized Entropy:", avg_normalized_entropy)


