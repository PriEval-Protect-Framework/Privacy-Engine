
from scipy.spatial import distance # EMD Distance



class DatasetScores:
    """
        privacy risk assessment is only meaningful for QIs, since PIs are assumed to be already handled/ removed.
    """
    
    def k_anonymity(df, quasi_identifiers):
        """
        Ensures that each record is indistinguishable from at least k-1 others.
        A dataset satisfies k-anonymity if every group of quasi-identifiers appears at least k times.
        Formula:
            k=min(group size for all quasi-identifier combinations)

        """
        group_counts = df.groupby(quasi_identifiers).size()
        return group_counts.min()
    

    def l_diversity(df, quasi_identifiers, sensitive_column):
        """
        Ensures that each quasi-identifier group has at least l different sensitive values.
        Protects against attribute disclosure.
        Formula:
            l=min(distinct sensitive values per group)
        """

        diversity_counts = df.groupby(quasi_identifiers)[sensitive_column].nunique()
        return diversity_counts.min()

    def t_closeness(df, quasi_identifiers, sensitive_column):
         
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
            dist = distance.jensenshannon(overall_distribution, group_distribution)
            max_distance = max(max_distance, dist)

        return round(max_distance, 4)
    

    def reidentification_risk(df, quasi_identifiers, k=2):
        """
        Measures the risk of re-identifying an individual from the dataset.
        The higher the risk, the lower the privacy.
        Formula:
            1 - (records that share quasi-identifiers with at least k others / total_records)
        Note: Here we took threshold k = 2 
        
        """

        group_counts = df.groupby(quasi_identifiers).size()
        shared_records = sum(group_counts[group_counts >= k])  # Records that share QI
        total_records = len(df)
        return round(1 - (shared_records / total_records), 4)

