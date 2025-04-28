import numpy as np
import pandas as pd
import os
from utils.detect_identifiers import DetectIdentifiers
from utils.dataset_scores import DatasetScores
from utils.hipaa_compliance import HIPAACalculator
from utils.gdpr_compliance import GDPRComplianceCalculator

class PrivacyScore:
    """
        Computes a privacy score based on given parameters and metrics.            
        returns:
            privacy_percentage: float, privacy score in percentage: pourcentage of privacy needed in the dataset. [0-100%]
    """

    def __init__(self, regulation, localisation, data_distribution, encryption):
        self.regulation = regulation
        self.localisation = localisation
        self.data_distribution = data_distribution
        self.encryption = encryption
        self.distribution_factor = self.get_distribution_factor()
        self.encryption_factor = self.get_encryption_factor()
        self.weights = self.set_initial_weights()
        
    def get_distribution_factor(self):
        # Centralized data has higher privacy needs (higher risk)
        if self.data_distribution == 'centralized':
            return 1.2
        elif self.data_distribution == 'federated':
            return 0.9
        elif self.data_distribution == 'decentralized':
            return 0.7
        else:  # Default or unknown
            return 1.0
            
    def get_encryption_factor(self):
        # Map encryption types to factor values (lower is better for privacy)
        # symmetric / asymmetric only!
        encryption_factors = {
            'None': 1.5,
            'Symmetric': 1.2,
            'Advanced Symmetric': 1.0,
            'Asymmetric': 0.8,
            'Hybrid': 0.6,
            'Homomorphic': 0.3
        }
        return encryption_factors.get(self.encryption, 1.0)  # Default to 1.0 if unknown

    def set_initial_weights(self):
        # Base weights - positive weights for risk factors, negative for protective factors
        weights = {
            'personal_identifiers': 7.0,   
            'quasi_identifiers': 5.0,     
            'sensitive_attributes': 6.0,  
            'reidentification_risk': 8.0, 
            'l_diversity': -3.0,           # Negative because higher diversity means better privacy
            'k_anonymity': -3.0,          
            't_closeness': -3.0,          
            'hipaa_compliance': -4.0 if self.regulation == 'HIPAA' else -2.0,  # Better compliance reduces privacy needs
            'gdpr_compliance': -4.0 if self.regulation == 'GDPR' else -2.0,    
        }

        # Localization adjustments
        if self.localisation == 'EU':
            weights['gdpr_compliance'] *= 1.5 
        elif self.localisation == 'US':
            weights['hipaa_compliance'] *= 1.5
            
        return weights

    def calculate_score(self, **metrics):
        """
        Calculate the privacy score based on given metrics and weights.
        How much privacy is needed in the dataset.
        Returns a value between 0-100% where:
        - Higher value means MORE privacy measures are needed
        - Lower value means FEWER privacy measures are needed
        """
        base_score = 50  
        
        for metric, value in metrics.items():
            if metric not in self.weights:
                continue
                
            weight = self.weights[metric]
            
            if metric == 'k_anonymity':
                # Convert k-anonymity to privacy need (smaller k means more privacy needed)
                k_factor = self.evaluate_k_anonymity(value)
                base_score += k_factor * (-weight)  # We invert the weight effect
                
            if metric == 'l_diversity':
                l_factor = self.evaluate_l_diversity(value)
                base_score += l_factor * (-weight)  
                
            if metric == 't_closeness':
                t_factor = self.evaluate_t_closeness(value)
                base_score += t_factor * (-weight)
                
            if metric == 'reidentification_risk':
                # Scale from 0-1 to stronger impact on score
                risk_factor = value * 10.0  # Scale up for stronger impact
                base_score += risk_factor * weight
                
            if metric in ['hipaa_compliance', 'gdpr_compliance']:
                compliance_factor = (100 - value) / 10.0  # Invert so lower compliance = higher need
                base_score += compliance_factor * (-weight)
                
            if metric in ['personal_identifiers', 'quasi_identifiers', 'sensitive_attributes']:
                base_score += value * weight
        
        base_score *= self.distribution_factor
        base_score *= self.encryption_factor
        
        # Normalize score between 0-100
        print("before clip", base_score)
        final_score = np.clip(base_score, 0, 100)
        return final_score
    
    def evaluate_k_anonymity(self, value):
        """Evaluate k-anonymity value into a privacy need factor"""
        if value == float('nan'):
            return 0
        if value <= 1:
            return 10.0  # Very poor anonymity
        elif value <= 5:
            return 8.0   
        elif value <= 10:
            return 5.0   
        elif value <= 20:
            return 2.0   
        else:
            return 0.5   # Excellent anonymity
            
    def evaluate_l_diversity(self, value):
        """Evaluate l-diversity value into a privacy need factor"""
        if value==float('inf'):
            return 0 # nothing to mesure
        
        elif value <= 1:
            return 10.0  # Very poor diversity
        elif value <= 2:
            return 7.0  
        elif value <= 4:
            return 4.0  
        elif value <= 6:
            return 2.0   
        else:
            return 0.5   # Excellent diversity
            
    def evaluate_t_closeness(self, value):
        """Evaluate t-closeness value into a privacy need factor"""
        if value >= 0.5:
            return 10.0  # Very poor closeness
        elif value >= 0.3:
            return 7.0  
        elif value >= 0.2:
            return 4.0  
        elif value >= 0.1:
            return 2.0 
        else:
            return 0.5   # Excellent closeness

if __name__ == "__main__":
    # privacy_calc = PrivacyScore(regulation='GDPR', localisation='EU', data_distribution='centralized', encryption='Homomorphic')
    # privacy_calc = PrivacyScore(regulation='HIPAA', localisation='US', data_distribution='decentralized', encryption='Homomorphic')
    privacy_calc = PrivacyScore(regulation='GDPR', localisation='EU', data_distribution='federated', encryption='Homomorphic')


    # number of identifiers
    detector = DetectIdentifiers()
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir, '.', 'data', 'test', 'healthcare_dataset.csv')
    # data_path = os.path.join(script_dir, 'data', 'test', 'COVID-19_Treatments_20250222.csv')
    # data_path = os.path.join(script_dir, '.', 'data', 'test', 'ASPR_Treatments_Locator_20250222.csv')

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
        'hipaa_compliance': hipaa_compliance_score,  
        'gdpr_compliance': gdpr_compliance_score,
    }
    score = privacy_calc.calculate_score(**metrics)
    print(f"Privacy Score: {score:.2f}%")
    print("Interpretation: Privacy measures needed for this dataset")
    
    print("\nContributing factors:")
    print(f"- Personal identifiers: {nb_personal_identifiers}")
    print(f"- Quasi-identifiers: {nb_quasi_identifiers}")
    print(f"- Sensitive attributes: {nb_sensitive_attributes}")
    print(f"- K-anonymity: {k_anonymity_score}")
    print(f"- L-diversity: {l_diversity_score}")
    print(f"- T-closeness: {t_closeness_score}")
    print(f"- Re-identification risk: {reid_risk}")
    print(f"- GDPR compliance: {gdpr_compliance_score}%")
    print(f"- HIPAA compliance: {hipaa_compliance_score}%")