import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import uuid
from itertools import combinations
from pycanon import anonymity 
import numpy as np

class AttributeClassification:
    """
    Class to classify attributes based on their types [QIDs, SAs, NSs]
    QIDs: Quasi-Identifiers
    SAs: Sensitive Attributes
    NSs: Non-Sensitive Attributes
    """
    def __init__(self, dataframe: pd.DataFrame, filename: str,  beta=(0.3, 0.8), alpha=(0.8, 1.0)):
        self.df = dataframe
        self.filename = filename
        self.columns = dataframe.columns
        self.beta = beta
        self.alpha = alpha
        self.qids = []
        self.sas = []
        self.nss = []
        self.risk_scores = {}

    def compute_g_distinct_matrix(self):
        """Compute g-distinct for all attributes (1 / freq of value)."""
        attr_dg_matrix = defaultdict(list)

        for attr in self.columns:
            value_counts = self.df[attr].value_counts().to_dict()

            for val in self.df[attr]:
                freq = value_counts.get(val, 1)
                g_value = 1.0 / freq
                attr_dg_matrix[attr].append(g_value)

        return attr_dg_matrix

    def compute_reidentification_risk(self, attr_dg_matrix):
        """Compute risk for each attribute as sum of its g-distinct values."""
        risk_scores = {}
        for attr, g_values in attr_dg_matrix.items():
            risk_scores[attr] = sum(g_values) / len(g_values)  # Normalize
        return risk_scores

    def classify_by_thresholds(self, risk_scores):
        """Classify each attribute based on risk."""
        for attr, risk in risk_scores.items():
            if self.beta[0] <= risk < self.beta[1]:
                self.qids.append(attr)
            elif self.alpha[0] <= risk <= self.alpha[1]:
                self.sas.append(attr)
            else:
                self.nss.append(attr)

    def classify_attributes(self):
        dg_matrix = self.compute_g_distinct_matrix()
        self.risk_scores = self.compute_reidentification_risk(dg_matrix)
        self.classify_by_thresholds(self.risk_scores)

    def get_classification(self):
        return {
            "QIDs": self.qids,
            "SAs": self.sas,
            "NSs": self.nss,
            "Rrisk": self.risk_scores
        }
    

    def run_on_csv(self, beta=(0.5, 0.8), alpha=(0.85, 1.0), visualize=True):
        df = self.df
        print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
        self.classify_attributes()
        result = self.get_classification()


        print("\n--- Classification Result ---")
        print("QIDs:", result["QIDs"])
        print("SAs:", result["SAs"])
        print("NSs:", result["NSs"])

        if visualize:
            print("\n--- Risk Scores ---")
            
            for attr, score in result["Rrisk"].items():
                print(f"{attr}: {score:.3f}")

            chart_dir = "src/charts"
            os.makedirs(chart_dir, exist_ok=True)

            # Generate UUID-based filename
            name = os.path.splitext(os.path.basename(self.filename))[0]
            out_path = os.path.join(chart_dir, f"{name}_{uuid.uuid4().hex[:8]}.png")

            # Plot and save
            plt.figure(figsize=(10, 5))
            plt.bar(result["Rrisk"].keys(), result["Rrisk"].values(), color='skyblue')
            plt.axhline(beta[0], color='orange', linestyle='--', label='QID threshold min')
            plt.axhline(beta[1], color='orange', linestyle='--', label='QID threshold max')
            plt.axhline(alpha[0], color='red', linestyle='--', label='SA threshold min')
            plt.axhline(alpha[1], color='red', linestyle='--', label='SA threshold max')
            plt.xticks(rotation=45)
            plt.ylabel("Re-identification Risk Score")
            plt.title("Privacy Risk Scores per Attribute")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()

            print(f"📊 Chart saved to: {out_path}")

        QID_optimal= self.identify_optimal_qid_dimension(k=3)
        result["QIDs"]= QID_optimal

        return result
    
    def compute_uniqueness(self, df, cols):
        return (df[cols].duplicated(keep=False) == False).sum() / len(df)

    def compute_nue(self, df, cols):
        """Compute Non-Uniform Uniqueness Entropy (NUE) for the given columns."""
        nue = 0
        for col in cols:
            counts = df[col].value_counts(normalize=True)
            entropy = -np.sum(counts * np.log2(counts))
            nue += entropy
        return nue / len(cols)

    def compute_pg(self, orig_uniqueness, anon_uniqueness):
        """Compute Privacy Gain (PG) as the difference between original and anonymized uniqueness."""
        return orig_uniqueness - anon_uniqueness



    def identify_optimal_qid_dimension(self, k=3):
        if not self.qids or len(self.qids) == 1:
            print("Not enough QIDs to evaluate combinations.")
            return self.qids

        original_uniqueness = self.compute_uniqueness(self.df, self.qids)
        best_pg = -np.inf
        best_nue = -np.inf
        best_combination = None

        for i in range(1, len(self.qids) + 1):
            for subset in combinations(self.qids, i):
                subset = list(subset)
                try:
                    k_val = anonymity.k_anonymity(self.df, subset)
                except Exception as e:
                    print(f"Error checking k-anonymity for {subset}: {e}")
                    continue

                if k_val < k:
                    continue  # skip if doesn't meet desired anonymity

                anon_uniqueness = self.compute_uniqueness(self.df, subset)
                nue = self.compute_nue(self.df, subset)
                pg = self.compute_pg(original_uniqueness, anon_uniqueness)

                if pg > best_pg and nue > best_nue:
                    best_pg = pg
                    best_nue = nue
                    best_combination = subset

        if best_combination:
            print(f"Best QID dimension found: {best_combination} with PG={best_pg:.3f}, NUE={best_nue:.3f}")
        else:
            print("No suitable QID dimension found that satisfies the k-anonymity condition.")
            return self.qids

        return best_combination




if __name__ == "__main__":

    print("Running Attribute Classification...")
    path = "./data/test/healthcare_dataset.csv"
    df= pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    filename= path.split("/")[-1]
    attribute_classifier = AttributeClassification(df, filename)

    #retrurn reslt as a dict in run_on_csv {}
    result = attribute_classifier.run_on_csv()
    
    print("final ", result)
