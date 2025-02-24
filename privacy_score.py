import numpy as np
import pandas as pd
import os
from utils.detect_identifiers import DetectIdentifiers
from utils.dataset_scores import DatasetScores
from utils.hipaa_compliance import HIPAACalculator
from utils.gdpr_compliance import GDPRComplianceCalculator

class PrivacyScore:
    """
        Computes a privacy score based on given parameters.
        params:
            
        returns:
            privacy_percentage: float, privacy score in percentage: pourcentage of privacy needed in the dataset. [0-100%]
    """

    def __init__(self, regulation, localisation, data_distribution, encryption):
        self.regulation = regulation
        self.localisation = localisation
        self.data_distribution = data_distribution
        self.encryption = encryption
        self.weights = self.set_initial_weights()

    def set_initial_weights(self):
        # Base weights (adjustable)
        weights = {
            'personal_identifiers': 1.5,  # Higher weight because more identifiers increase risk
            'quasi_identifiers': 1.2,
            'sensitive_attributes': 1.0,
            'reidentification_risk': -1.2,  # Negative because lower risk is better
            'l_diversity': -1.0,            # Negative because higher diversity means better privacy
            'k_anonymity': -1.0,            
            't_closeness': -1.0,            
            'access_control': 0.9, # encryption strength, no encryption is penalized by high weight
            'hipaa_compliance': 0.6 if self.regulation == 'HIPAA' else 0.3,
            'gdpr_compliance': 0.9 if self.regulation == 'GDPR' else 0.3,
            'iso_27799_compliance': 0.7 if self.regulation == 'ISO 27799' else 0.3,
            'distribution_penalty': 0.05,
        }

        # Localization adjustments
        if self.localisation == 'EU':
            weights['gdpr_compliance'] *= 1.5  # More focus on GDPR compliance in EU
        elif self.localisation == 'US':
            weights['hipaa_compliance'] *= 1.5

        if self.data_distribution == 'centralized':
            weights['distribution_penalty'] = 0.1 # Penalize centralized data distribution
        
        # encryption [None, 'Symmetric', 'Advanced Symmetric', 'Asymmetric', 'Hybrid', 'Homomorphic'] in ascending order of security
        encryption_weights = {
            'None': 0.9,
            'Symmetric': 0.8,
            'Advanced Symmetric': 0.7,
            'Asymmetric': 0.5,
            'Hybrid': 0.3,
            'Homomorphic': 0.1
        }
        weights['access_control'] = encryption_weights.get(self.encryption, 0.9)  # Default to most penalizing if unknown

        return weights
    

    def calculate_score(self, **metrics):
        """
        Calculate the privacy score based on given metrics and weights.
        How much privacy is needed in the dataset.
        """
        total_score = 0
        for metric, value in metrics.items():
            weight = self.weights.get(metric, 0)  # Default to 0 if metric not recognized
            total_score += value * weight
        
        # Normalize score to be between 0 and 100%
        normalized_score = np.clip(total_score, 0, 100)
        return normalized_score

if __name__ == "__main__":
    privacy_calc = PrivacyScore(regulation='GDPR', localisation='EU', data_distribution='centralized', encryption='Homomorphic')

    # number of identifiers
    detector = DetectIdentifiers()
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir, '.', 'data', 'test', 'healthcare_dataset.csv')
    df=pd.read_csv(data_path, )
    column_names = list(df.columns)
    identifiers , nb_personal_identifiers, nb_quasi_identifiers, nb_sensitive_attributes = detector.detect_identifiers(column_names)

    # dataset scores
    ds = DatasetScores()
    k_anonymity_score = ds.k_anonymity(df, identifiers['quasi_identifiers'])
    l_diversity_score = ds.l_diversity(df, identifiers['quasi_identifiers'], identifiers['sensitive_attributes'])
    t_closeness_score = ds.t_closeness(df, identifiers['quasi_identifiers'], identifiers['sensitive_attributes'])
    reid_risk = ds.reidentification_risk(df, identifiers['quasi_identifiers'])

    # compliance scores
    compliance_calculator = GDPRComplianceCalculator()

    # Example scores for each GDPR principle (0-100)
    gdpr_scores = {
        "Lawfulness, Fairness, and Transparency": 80,
        "Purpose Limitation": 90,
        "Data Minimization": 70,
        "Accuracy": 100,
        "Storage Limitation": 60,
        "Integrity and Confidentiality": 90,
        "Accountability": 80
    }
    gdpr_compliance_score = compliance_calculator.calculate_compliance_score(gdpr_scores)

    # Example scores for each HIPAA principle (0-100)
    hipaa_scores = {
        "Data Encryption": 100,
        "Access Controls": 75,
        "Audit Logs": 90,
        "Training": 80,
        "Risk Assessments": 70,
        "Incident Response": 100,
    }

    calculator = HIPAACalculator()

    calculator.set_scores(hipaa_scores)
    hipaa_compliance_score=calculator.calculate_compliance_score()

    # final metrics
    metrics = {
        'personal_identifiers': nb_personal_identifiers,
        'quasi_identifiers': nb_quasi_identifiers,
        'sensitive_attributes': nb_sensitive_attributes,
        'reidentification_risk': reid_risk,
        'l_diversity': l_diversity_score,
        'k_anonymity': k_anonymity_score,
        't_closeness': t_closeness_score,
        'access_control': 60,
        'hipaa_compliance': hipaa_compliance_score,  
        'gdpr_compliance': gdpr_compliance_score,
    }
    score = privacy_calc.calculate_score(**metrics)
    print("Privacy Score: ", score)