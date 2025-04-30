import pandas as pd
from algorithmic_attribute_classification import AttributeClassification


class AdversarySuccessMetrics:
    def __init__(self, df, quasi_identifiers):
        self.df = df
        self.qi = quasi_identifiers

    def adversary_success_rate(self):
        grouped = self.df.groupby(self.qi)
        rates = [1 / len(g) for _, g in grouped if len(g) > 0]
        return {
            "average_success_rate": round(sum(rates) / len(rates), 4) if rates else 0,
            "max_success_rate": round(max(rates), 4) if rates else 0,
            "min_success_rate": round(min(rates), 4) if rates else 0,
            "num_equivalence_classes": len(rates)
        }

    def delta_presence(self, original_df):
        intersection = pd.merge(self.df, original_df, how='inner')
        delta = len(intersection) / len(self.df) if len(self.df) > 0 else 0
        return {
            "delta_presence": round(delta, 4),
            "shared_records": len(intersection),
            "published_records": len(self.df)
        }


if __name__ == "__main__":

    FILE_PATH = "./data/test/healthcare_dataset.csv"
    DATA = pd.read_csv(FILE_PATH)
    attributes = AttributeClassification(DATA, FILE_PATH).run_on_csv()
    QI = attributes['QIDs']

    adversary = AdversarySuccessMetrics(DATA, QI)

    print("Adversary Success Rate: ", adversary.adversary_success_rate())
    print("Delta Presence: ", adversary.delta_presence(DATA))

