import pandas as pd
from algorithmic_attribute_classification import AttributeClassification
import numpy as np
from scipy.spatial import distance
from concurrent.futures import ProcessPoolExecutor

class DataSimilarity:
    def __init__(self, DATA, QI, SA):
        self.DATA = DATA
        self.QI = QI
        self.SA = SA

    def k_anonymity(self):
        if not self.QI:
            return 1 #worst case
        return self.DATA.groupby(self.QI).size().min()

    def alpha_k_anonymity(self):
        if not self.QI:
            return 1,1
        group_sizes = self.DATA.groupby(self.QI).size()
        alpha = group_sizes.mean()
        k = group_sizes.min()
        return alpha, k

    def l_diversity(self):

        if not self.QI:
            return len(self.DATA[self.SA].nunique())
        
        grouped = self.DATA.groupby(self.QI)
        
        entropies = []
        for _, group in grouped:
            counts = group[self.SA].value_counts(normalize=True).values
            nonzero_counts = counts[counts > 0]
            entropy = -np.sum(nonzero_counts * np.log2(nonzero_counts))
            entropies.append(entropy)
        
        return np.mean(entropies) if entropies else 0



    def compute_group_t_distance(group_df, sa_col, global_values, global_array):
        group_counts = group_df[sa_col].value_counts(normalize=True)
        
        group_dist = pd.Series(0, index=global_values)
        group_dist.update(group_counts)
        
        t_distance = np.abs(global_array - group_dist.values).max()
        return t_distance


    def t_closeness_parallel(self, num_workers=4):
        global_dist = self.DATA[self.SA].value_counts(normalize=True)
        global_values = global_dist.index.tolist()
        global_array = global_dist.values
        
        grouped = list(self.DATA.groupby(self.QI))  # materialize groups
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    self.compute_group_t_distance,
                    group,
                    self.SA,
                    global_values,
                    global_array
                )
                for _, group in grouped
            ]
            t_distances = [f.result() for f in futures]
        
        return max(t_distances)

if __name__ == "__main__":
    # FILE_PATH = "./data/test/COVID-19_Treatments_20250222.csv"
    FILE_PATH = "./data/test/healthcare_dataset.csv"

    DATA = pd.read_csv(FILE_PATH)

    attributes =  AttributeClassification(DATA, FILE_PATH).run_on_csv()
    QI= attributes['QIDs']
    SA = attributes['SAs']


    
    data_similarity = DataSimilarity(DATA, QI, SA)
    print("k-anonymity: ", data_similarity.k_anonymity())
    alpha, k = data_similarity.alpha_k_anonymity()
    print(f"alpha-k-anonymity: (alpha={float(alpha):.4f}, k={int(k)})")
    print("l-diversity: ", data_similarity.l_diversity())
    # print("t-closeness: ", data_similarity.t_closeness()) # ! takes alot of time
