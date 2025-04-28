from pycanon import anonymity, report
import pandas as pd
from attribute_classification import AttributeClassification

class DataSimilarity: 

    def __init__(self, DATA, QI, SA):
        self.DATA = DATA
        self.QI = QI
        self.SA = SA
        

    def k_anonymity(self, DATA, QI):
        """
        returns the k for k-anonymity
        """
        return anonymity.k_anonymity(DATA, QI)

    def alpha_k_anonymity(self):
        """
        returns the alpha, k for alpha-k-anonymity
        """
        return anonymity.alpha_k_anonymity(self.DATA, self.QI, self.SA)


    def l_diversity(self):
        """
        returns the entropy l for l-diversity
        """
        return anonymity.l_diversity(self.DATA, self.QI, self.SA)

    def t_closeness(self):
        """
        returns the t for t-closeness
        """
        return anonymity.t_closeness(self.DATA, self.QI, self.SA)

    def full_report(self):
        """
        returns the full report
        """
        return report.print_report(self.DATA, self.QI, self.SA)
    
    def report_json(self):
        """
        returns the report in json format
        """
        return report.get_report_values(self.DATA, self.QI, self.SA)
    
    def delta_disclosure(self):
        """
        returns the delta disclosure
        """
        return anonymity.delta_disclosure(self.DATA, self.QI, self.SA)


if __name__ == "__main__":
    FILE_PATH = "./data/test/healthcare_dataset.csv"
    DATA = pd.read_csv(FILE_PATH)

    attributes =  AttributeClassification(DATA, FILE_PATH).run_on_csv()
    QI= attributes['QIDs']
    SA = attributes['SAs']


    
    data_similarity = DataSimilarity(DATA, QI, SA)
    print("k-anonymity: ", data_similarity.k_anonymity(DATA, QI))
    print("alpha-k-anonymity: ", data_similarity.alpha_k_anonymity())
    print("l-diversity: ", data_similarity.l_diversity())

    
    # print("delta-disclosure: ", data_similarity.delta_disclosure())

    # SA = attributes['SAs'][0] if attributes['SAs'] else None

    # if SA:
    #     print("t-closeness: ", data_similarity.t_closeness())
    # print("full report: ", data_similarity.full_report())
    # print("report json: ", data_similarity.report_json())
