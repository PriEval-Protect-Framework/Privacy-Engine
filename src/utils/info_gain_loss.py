from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd
from src.utils.algorithmic_attribute_classification import AttributeClassification


class InformationGainLoss:

    def __init__(self, data, QIs, SAs):
        self.DATA = data
        self.QIs = QIs
        self.SAs = SAs

    def calculate_mutual_information(self):
        if not self.QIs or not self.SAs:
            print("No QIs or SAs provided. Returning MI = 0.")
            return 0

        encoder = OrdinalEncoder()
        mi_values = []

        for qi in self.QIs:
            for sa in self.SAs:
                try:
                    encoded = encoder.fit_transform(self.DATA[[qi, sa]])
                    mi = mutual_info_score(encoded[:, 0], encoded[:, 1])
                    mi_values.append(mi)
                except Exception as e:
                    print(f"Skipping MI({qi}, {sa}) due to error: {e}")

        return np.mean(mi_values) if mi_values else 0

    def calculate_privacy_score(self):
        if not self.QIs or not self.SAs:
            print("No QIs or SAs provided. Returning Privacy Score = 0.")
            return 0

        entropy_values = []

        for sa in self.SAs:
            grouped = self.DATA.groupby(self.QIs)
            total_records = len(self.DATA)
            weighted_entropy = 0

            for _, group in grouped:
                group_weight = len(group) / total_records
                counts = group[sa].value_counts(normalize=True)
                entropy = -np.sum(counts * np.log2(counts + 1e-10))  
                weighted_entropy += group_weight * entropy

            entropy_values.append(weighted_entropy)

        return np.mean(entropy_values) if entropy_values else 0


if __name__ == "__main__":
    
    FILE_PATH = "./data/test/healthcare_dataset.csv"
    DATA = pd.read_csv(FILE_PATH)

    attributes =  AttributeClassification(DATA, FILE_PATH).run_on_csv()
    QI= attributes['QIDs']
    SA = attributes['SAs']

    if not QI or not SA:
        print("No QIDs or SAs found in the dataset.")
        exit(1)

    info_gain_loss = InformationGainLoss(DATA, QI, SA)

    print("Mutual Information: ", info_gain_loss.calculate_mutual_information())
    print("Relative entropy: ", info_gain_loss.calculate_privacy_score())

