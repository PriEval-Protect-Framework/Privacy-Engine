
class GDPRComplianceCalculator:
    """
    A class to calculate GDPR compliance scores for a medical dataset.
    The GDPR compliance score is calculated based on the following principles:
    
        Lawfulness, fairness and transparency.
        Purpose limitation.
        Data minimisation.
        Accuracy.
        Storage limitation.
        Integrity and confidentiality (security)
        Accountability.

    """

    def __init__(self):
        # Define GDPR principles and their weights (adjustable)
        self.gdpr_principles = {
            "Lawfulness, Fairness, and Transparency": 0.20,
            "Purpose Limitation": 0.15,
            "Data Minimization": 0.15,
            "Accuracy": 0.10,
            "Storage Limitation": 0.10,
            "Integrity and Confidentiality": 0.20,
            "Accountability": 0.10
        }

    def calculate_compliance_score(self, scores):
        """
        Calculate the GDPR compliance score based on weighted principles.

        :param scores: A dictionary with GDPR principles as keys and scores % as values.
        :return: Total GDPR compliance score percentage.
        """
        total_score = 0.0

        # Validate input scores
        for principle in self.gdpr_principles:
            if principle not in scores:
                raise ValueError(f"Missing score for principle: {principle}")
            if not (0 <= scores[principle] <= 100):
                raise ValueError(f"Score for {principle} must be between 0 and 100.")

        # Calculate weighted score
        for principle, weight in self.gdpr_principles.items():
            total_score += scores[principle] * weight

        return total_score

    def interpret_score(self, score):
        """
        Interpret the GDPR compliance score.

        :param score: The compliance score (0-100).
        :return: A string interpretation of the score.
        """
        if score >= 90:
            return "Highly compliant"
        elif 70 <= score < 90:
            return "Mostly compliant but needs improvement"
        else:
            return "Significant gaps in compliance"

    def generate_report(self, scores):
        """
        Generate a GDPR compliance report.

        :param scores: A dictionary with GDPR principles as keys and scores % as values.
        """
        try:
            compliance_score = self.calculate_compliance_score(scores)
            interpretation = self.interpret_score(compliance_score)

            print("\n--- GDPR Compliance Report ---")
            for principle, score in scores.items():
                print(f"{principle}: {score}%")
            print(f"\nOverall Compliance Score: {compliance_score:.2f}%")
            print(f"Interpretation: {interpretation}")
        except ValueError as e:
            print(f"Error: {e}")


# Test 
if __name__ == "__main__":

    compliance_calculator = GDPRComplianceCalculator()

    # Example scores for each GDPR principle (0-100)
    scores = {
        "Lawfulness, Fairness, and Transparency": 80,
        "Purpose Limitation": 90,
        "Data Minimization": 70,
        "Accuracy": 100,
        "Storage Limitation": 60,
        "Integrity and Confidentiality": 90,
        "Accountability": 80
    }


    print("compliance score: ", compliance_calculator.calculate_compliance_score(scores), "%")

    compliance_calculator.generate_report(scores)