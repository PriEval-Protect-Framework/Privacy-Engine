import numpy as np
import pandas as pd
from algorithmic_attribute_classification import AttributeClassification


class Uncertainty:

    def __init__(self, data, SA):
        self.data = data
        self.series = SA

    def entropy(self, series):
        probs = series.value_counts(normalize=True)
        return -np.sum(probs * np.log2(probs))


    def min_entropy(self, series):
        p_max = series.value_counts(normalize=True).max()
        return -np.log2(p_max)


    def max_entropy(self, series):
        n = series.nunique()
        return np.log2(n)

    def normalized_entropy(self, series):
        h = self.entropy(series)
        h_max = self.max_entropy(series)
        return h / h_max if h_max != 0 else 0


if __name__ == "__main__":

    FILE_PATH = "./data/test/healthcare_dataset.csv"
    DATA = pd.read_csv(FILE_PATH)

    attributes =  AttributeClassification(DATA, FILE_PATH).run_on_csv()
    SA = attributes['SAs']

    uncertainty= Uncertainty(DATA, SA)

    print("Entropy: ", uncertainty.entropy(DATA[SA]))
    print("Min Entropy: ", uncertainty.min_entropy(DATA[SA]))
    print("Normalized Entropy: ", uncertainty.normalized_entropy(DATA[SA]))


