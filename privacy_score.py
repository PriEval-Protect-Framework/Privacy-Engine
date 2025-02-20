import numpy as np

class PrivacyScore:

    @staticmethod
    def privacy_score(
        personal_identifiers=0,
        quasi_identifiers=0,
        sensitive_attributes=0,
        reidentification_risk=0,
        l_diversity=0,
        k_anonymity=0,
        t_closeness=0,
        access_control=0,
        regulation="HIPPA",
        localisation="EU",
        hipaa_compliance=0,
        gdpr_compliance=0,
        data_distribution="centralized",
    ):
        """
        Computes a privacy score based on given parameters.
        params:
            personal_identifiers: int, number of personal identifiers such as name, email, etc.
            quasi_identifiers: int, number of quasi identifiers such as age, gender, ect.
            sensitive_attributes: int, number of sensitive attributes such as record history, health data, disease etc.
            reidentification_risk: float between 0 and 1, probability of risk of re-identification of individuals in the dataset.
            l_diversity: int, minimum L in which each group has at least L different sensitive values.
            k_anonymity: int, the higher the better
            t_closeness: float, how much the distribution of sensitive values in each equivalence class is close to the overall dataset distribution.
            regulation: string, ['HIPPA', 'GDPR', 'ISO 27799'], regulation to be followed.
            localisation: string, data localization from a country to another can impose stricter privacy.
            hippa_compliance: float, how much the data is compliant with HIPPA regulation.
            gdpr_compliance: float, how much the data is compliant with GDPR regulation.
            data_distribution: string, ['centralized', 'federated'], how the data is distributed.
        returns:
            privacy_percentage: float, privacy score in percentage: pourcentage of privacy needed in the dataset. [0-100%]
        """

        
        # Assigning weights to each factor (TO BE FURTHER ADJUSTED)
        weights = {
            "personal_identifiers": -0.2,  # Higher personal identifiers decrease privacy
            "quasi_identifiers": -0.1,  # More quasi-identifiers = more risk
            "sensitive_attributes": -0.2,  # More sensitive attributes decrease privacy
            "reidentification_risk": -0.15,  # Higher risk lowers privacy
            "l_diversity": 0.1,  # Higher l-diversity improves privacy
            "k_anonymity": 0.1,  # Higher k-anonymity improves privacy
            "t_closeness": 0.1,  # Higher t-closeness improves privacy
            "hipaa_compliance": 0.1,  # Compliance boosts privacy score
            "gdpr_compliance": 0.1,  # Compliance boosts privacy score
        }
        
        # Adjusting score based on regulation strictness
        regulation_scores = {
            "HIPAA": 0.8,
            "GDPR": 0.9,
            "ISO 27799": 0.85,
        }
        regulation_score = regulation_scores.get(regulation, 0.7)
        
        # Adjusting for data distribution
        distribution_penalty = -0.1 if data_distribution == "centralized" else 0.05
        
        # Compute weighted sum
        score = sum(
            weights[key] * value
            for key, value in locals().items()
            if key in weights
        ) + (0.1 * regulation_score) + distribution_penalty
        
        # Normalize score to percentage (0-100)
        privacy_percentage = np.clip(score * 100, 0, 100)
        
        return round(privacy_percentage, 2)

# Example usage
privacy = PrivacyScore.privacy_score(
    personal_identifiers=0.9,
    quasi_identifiers=0.8,
    sensitive_attributes=0.7,
    reidentification_risk=0.2,
    l_diversity=0.6,
    k_anonymity=0.8,
    t_closeness=0.7,
    regulation="GDPR",
    hipaa_compliance=0.95,
    gdpr_compliance=0.9,
    data_distribution="federated",
)

print(f"Estimated Privacy Score: {privacy}%")