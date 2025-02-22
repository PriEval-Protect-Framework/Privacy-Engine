import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import ast
import re

load_dotenv()

class DetectIdentifiers:
    """
    This class detects and classifies identifiers from dataset column names using the Gemini-Pro AI model.
    
    Categories:
        - Personal identifiers (e.g., name, phone number, email address)
        - Quasi-identifiers (e.g., age, gender)
        - Sensitive attributes (e.g., disease, medical history, financial data)
        - Neither of the above
    """

    def __init__(self):
        GEMINI_PRO_API_KEY = os.getenv('GEMINI_PRO_API_KEY')

        if not GEMINI_PRO_API_KEY:
            raise ValueError("GOOGLE_API_KEY is missing from environment variables.")
        genai.configure(api_key=GEMINI_PRO_API_KEY)
        self.model = genai.GenerativeModel('gemini-pro')

    def clean_json_string(self, json_string):
        pattern = r'^```json\s*(.*?)\s*```$'
        cleaned_string = re.sub(pattern, r'\1', json_string, flags=re.DOTALL)
        return cleaned_string.strip()


    def detect_identifiers(self, column_names):
        """
        Detects identifiers from column names using an LLM API.
        
        Params:
            column_names: list of strings, list of column names.
        
        Returns:
            dict: Dictionary containing detected identifiers categorized into four classes.
        """
        if not column_names:
            return {"error": "No column names provided."}

        prompt = (
            "You are an expert in data classification and privacy protection. Given a list of dataset column names, "
            "categorize each into one of the following classes: \n"
            "1. Personal Identifiers (Directly identify an individual, e.g., full name, phone number, SSN, passport number)\n"
            "2. Quasi-Identifiers (Do not directly identify an individual but can be used in combination to do so, e.g., age, gender, zip code, job title, IP address)\n"
            "3. Sensitive Attributes (Highly confidential data that can cause harm or discrimination if exposed, e.g., medical history, financial data, criminal record, political views)\n"
            "4. Neither (Columns that do not fall into any of the above categories)\n"
            "Return the output as a valid JSON object with double quotes, formatted as follows: \n"
            "{ \"personal_identifiers\": [...], \"quasi_identifiers\": [...], \"sensitive_attributes\": [...], \"neither\": [...] }\n"
            f"Column names: {column_names}"
        )

        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 500,
            }
        )

        try:
            # Try parsing response as JSON
            cleaned=self.clean_json_string(response.text)
            identifiers = json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                # Fallback: Use ast.literal_eval() for improperly formatted JSON
                identifiers = ast.literal_eval(response.text)
                print(identifiers)
            except Exception as e:
                identifiers = {"error": f"Failed to parse response from AI: {str(e)}"}

        return identifiers


if __name__ == "__main__":
    
    detector = DetectIdentifiers()
    column_names = ["name", "email", "age", "gender", "medical_history", "account_balance", "purchase_history", "blood pressure", "code postal"]
    identifiers = detector.detect_identifiers(column_names)
    print(json.dumps(identifiers, indent=4))
