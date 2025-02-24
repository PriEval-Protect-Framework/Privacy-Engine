

class HIPAACalculator:
    
    """
    Key Compliance Metrics for HIPAA:

        Data Encryption: Is PHI encrypted at rest and in transit?

        Access Controls: Are there role-based access controls (RBAC) to limit who can access PHI?

        Audit Logs: Are access and modifications to PHI logged and monitored?

        Training: Are employees trained on HIPAA compliance annually?

        Risk Assessments: Are regular risk assessments conducted?

        Incident Response: Is there a documented process for handling PHI breaches?
    """


    def __init__(self):
        # Define metrics and their weights (adjustable)
        self.metrics = {
            "Data Encryption": {"weight": 0.20, "score": 0},
            "Access Controls": {"weight": 0.20, "score": 0},
            "Audit Logs": {"weight": 0.15, "score": 0},
            "Training": {"weight": 0.10, "score": 0},
            "Risk Assessments": {"weight": 0.15, "score": 0},
            "Incident Response": {"weight": 0.20, "score": 0},
        }

    def set_scores(self, scores):
        # Set scores for each metric
        for metric, score in scores.items():
            if metric in self.metrics:
                if 0 <= score <= 100:
                    self.metrics[metric]["score"] = score
                else:
                    raise ValueError(f"Score for {metric} must be between 0 and 100.")
            else:
                raise KeyError(f"Invalid metric: {metric}")

    def calculate_compliance_score(self):
        # Calculate the weighted compliance score
        total_score = 0
        for metric, data in self.metrics.items():
            total_score += data["score"] * data["weight"]
        return total_score

    def interpret_score(self, score):
        if score >= 90:
            return "Highly compliant"
        elif 70 <= score < 90:
            return "Mostly compliant but needs improvement"
        else:
            return "Significant gaps in compliance"

    def generate_report(self):
        compliance_score = self.calculate_compliance_score()
        interpretation = self.interpret_score(compliance_score)

        print("\n--- HIPAA Compliance Report ---")
        for metric, data in self.metrics.items():
            print(f"{metric}: {data['score']}%")
        print(f"\nOverall Compliance Score: {compliance_score:.2f}%")
        print(f"Interpretation: {interpretation}")


# Test
if __name__ == "__main__":
    # Example HIPAA compliance scores
    scores = {
        "Data Encryption": 100,
        "Access Controls": 75,
        "Audit Logs": 90,
        "Training": 80,
        "Risk Assessments": 70,
        "Incident Response": 100,
    }

    calculator = HIPAACalculator()

    calculator.set_scores(scores)
    calculator.calculate_compliance_score()
    calculator.generate_report()