from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer
import json
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modifie ici pour spécifier des domaines spécifiques
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Définir le chemin vers votre modèle Hugging Face
model_name = "MrFrijo/LiAPI"

# Charger le modèle et le tokenizer depuis Hugging Face
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Initialiser FastAPI
app = FastAPI()

# Schémas pour les requêtes
class TranslationRequest(BaseModel):
    text: str
    src_lang: str  # Langue source ("fr" ou "li")
    target_lang: str  # Langue cible ("li" ou "fr")

class EvaluationRequest(BaseModel):
    translated_text: str
    feedback: str  # "positive" ou "negative"

# Endpoint de test
@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API Lingala !"}

# Endpoint pour la traduction
@app.post("/translate/")
def translate(request: TranslationRequest):
    # Validation des langues source et cible
    if request.src_lang not in ["fr", "li"] or request.target_lang not in ["li", "fr"]:
        raise HTTPException(status_code=400, detail="Les langues doivent être 'fr' ou 'li'.")
    if request.src_lang == request.target_lang:
        raise HTTPException(status_code=400, detail="Les langues source et cible ne peuvent pas être identiques.")

    # Vérification si le texte est vide
    input_text = request.text.strip()
    if not input_text:
        raise HTTPException(status_code=400, detail="Le texte à traduire est vide.")

    # Tokeniser le texte
    tokenized_text = tokenizer(input_text, return_tensors="pt")

    # Effectuer la traduction
    translated = model.generate(**tokenized_text)

    # Convertir les tokens traduits en texte
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    return {
        "source_text": input_text,
        "translated_text": translated_text,
        "source_language": request.src_lang,
        "target_language": request.target_lang,
    }

# Endpoint pour l'évaluation
@app.post("/evaluate/")
def evaluate(request: EvaluationRequest):
    # Vérifier le feedback
    if request.feedback not in ["positive", "negative"]:
        raise HTTPException(status_code=400, detail="Le feedback doit être 'positive' ou 'negative'.")

    # Construire une entrée de feedback
    feedback_data = {
        "translated_text": request.translated_text,
        "feedback": request.feedback,
    }

    # Enregistrer les évaluations dans un fichier JSON
    try:
        with open("feedback.json", "a") as f:
            f.write(json.dumps(feedback_data) + "\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'enregistrement : {str(e)}")

    return {"message": "Merci pour votre évaluation !"}

# Exemple d'utilisation
# Pour lancer l'API : `uvicorn nom_du_fichier:app --reload`
