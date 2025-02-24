
from scipy.spatial import distance # EMD Distance
import pandas as pd


class DatasetScores:
    """
    Class to calculate privacy scores for a dataset:
    - k-anonymity
    - l-diversity
    - t-closeness
    - re-identification risk
    Note: The privacy risk assessment is only meaningful for QIs, since PIs are assumed to be already handled/ removed.
    """
    
    def k_anonymity(self, df, quasi_identifiers):
        """
        Ensures that each record is indistinguishable from at least k-1 others.
        A dataset satisfies k-anonymity if every group of quasi-identifiers appears at least k times.
        Formula:
            k=min(group size for all quasi-identifier combinations)

        """
        group_counts = df.groupby(quasi_identifiers).size()
        return group_counts.min()
    

    def l_diversity(self, df, quasi_identifiers, sensitive_column):
        """
        Ensures that each quasi-identifier group has at least l different sensitive values.
        Protects against attribute disclosure.
        Formula:
            l=min(distinct sensitive values per group)
        """

        diversity_counts = df.groupby(quasi_identifiers)[sensitive_column].nunique()
        return diversity_counts.min()

    def t_closeness(self, df, quasi_identifiers, sensitive_column):
         
        """
        Ensures that the distribution of sensitive values in each equivalence class is close to the overall distribution of sensitive values.
        Protects against attribute disclosure.
        Formula:
            t=max(abs(overall_dist - dist_in_group))

            Ensures the distribution of sensitive values in each quasi-identifier group is similar to the overall distribution.
            Uses Earth Mover’s Distance (EMD) to compare distributions.
            Formula:
                t=max(D(P(Q),P(D)))
            D(P(Q),P(D)) is the distance between the sensitive attribute distribution in each group Q and the overall dataset D.

        """
        overall_distribution = df[sensitive_column].value_counts(normalize=True)
    
        max_distance = 0
        for _, group in df.groupby(quasi_identifiers):
            group_distribution = group[sensitive_column].value_counts(normalize=True)
            # Align the group distribution with the overall distribution's index
            group_distribution = group_distribution.reindex(overall_distribution.index, fill_value=0)
            dist = distance.jensenshannon(overall_distribution, group_distribution)
            max_distance = max(max_distance, dist)

        return round(max_distance, 4)
    

    def reidentification_risk(self, df, quasi_identifiers):
        """
        Calculates the average re-identification risk over all records.
        For each equivalence class (group of records sharing the same QIs),
        each record has a risk of 1/(group size).
        The overall risk is the weighted average of these risks.
        """
        group_counts = df.groupby(quasi_identifiers).size()
        
        # Compute risk for each group: risk per record is 1 / (group size)
        # Then, weight that risk by the number of records in that group.
        total_risk = 0
        for group_size in group_counts:
            total_risk += group_size * (1 / group_size)  
            
        # The average risk per record is the number of groups divided by the total number of records.
        avg_risk = total_risk / len(df)
        return round(avg_risk, 4)

# Test
if __name__ == "__main__":

    # dummy data
    data = {
        'age': [25, 25, 30, 30, 30, 35, 35, 40, 40, 40],
        'zipcode': ['12345', '12345', '23456', '23456', '23456', '34567', '34567', '45678', '45678', '45678'],
        'disease': ['Flu', 'Cold', 'Flu', 'Flu', 'Cold', 'Cancer', 'Cancer', 'Diabetes', 'Diabetes', 'Flu']
    }
    df = pd.DataFrame(data)

    # Define the quasi-identifiers and the sensitive column
    quasi_identifiers = ['age', 'zipcode']
    sensitive_column = 'disease'
    
    ds = DatasetScores()
    
    k_anonymity_score = ds.k_anonymity(df, quasi_identifiers)
    print("k-anonymity score:", k_anonymity_score)
    
    l_diversity_score = ds.l_diversity(df, quasi_identifiers, sensitive_column)
    print("l-diversity score:", l_diversity_score)
    
    t_closeness_score = ds.t_closeness(df, quasi_identifiers, sensitive_column)
    print("t-closeness score:", t_closeness_score)
    
    reid_risk = ds.reidentification_risk(df, quasi_identifiers)
    print("Re-identification risk:", reid_risk)


