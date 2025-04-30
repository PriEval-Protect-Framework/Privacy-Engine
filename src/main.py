from fastapi import FastAPI, File, UploadFile
import pandas as pd
from src.utils.algorithmic_attribute_classification import AttributeClassification
from src.utils.adversary_success import AdversarySuccessMetrics
from src.utils.data_similarity import DataSimilarity
from src.utils.info_gain_loss import InformationGainLoss
from src.utils.uncentainty import Uncertainty
import tempfile

app = FastAPI()

@app.post("/calcul")
async def calculate_privacy_metrics(file: UploadFile = File(...)):
    try:
        # Save uploaded file to a temp location
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        contents = await file.read()
        temp.write(contents)
        temp.close()

        df = pd.read_csv(temp.name)
    except Exception as e:
        return {"error": f"Failed to read uploaded file: {e}"}

    attributes = AttributeClassification(df, file.filename).run_on_csv()
    QIDs = attributes.get("QIDs", [])
    SAs = attributes.get("SAs", [])

    result = {"attribute_classification": attributes}

    if not QIDs or not SAs:
        result["error"] = "Insufficient QIDs or SAs for full metric computation."
        return result

    adversary = AdversarySuccessMetrics(df, QIDs)
    result["adversary_success_rate"] = adversary.adversary_success_rate()
    result["delta_presence"] = adversary.delta_presence(df)

    data_similarity = DataSimilarity(df, QIDs, SAs[0])
    result["k_anonymity"] = int(data_similarity.k_anonymity())
    alpha, k = data_similarity.alpha_k_anonymity()
    result["alpha_k_anonymity"] = {"alpha": round(float(alpha), 4), "k": int(k)}
    result["l_diversity"] = float(data_similarity.l_diversity())

    info_gain = InformationGainLoss(df, QIDs, SAs)
    result["mutual_information"] = float(round(info_gain.calculate_mutual_information(), 4))
    result["privacy_score_entropy"] = float(round(info_gain.calculate_privacy_score(), 4))

    uncertainty = Uncertainty(df, SAs)
    entropy, min_entropy, norm_entropy = uncertainty.uncertainty_calculate_all()
    result["uncertainty_metrics"] = {
        "avg_entropy": float(round(entropy, 4)),
        "avg_min_entropy": float(round(min_entropy, 4)),
        "avg_normalized_entropy": float(round(norm_entropy, 4))
    }

    return result
