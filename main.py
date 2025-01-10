from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer
from fastapi.middleware.cors import CORSMiddleware
import json
import os

# Initialiser l'application FastAPI
app = FastAPI()

# Ajouter les configurations CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modifie ici pour spécifier des domaines spécifiques
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle et le tokenizer
try:
    model_name = "MrFrijo/LiAPI"  # Remplace par le bon chemin si nécessaire
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle : {e}")

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
    try:
        # Validation des langues source et cible
        if request.src_lang not in ["fr", "li"] or request.target_lang not in ["li", "fr"]:
            raise HTTPException(status_code=400, detail="Les langues doivent être 'fr' ou 'li'.")
        if request.src_lang == request.target_lang:
            raise HTTPException(status_code=400, detail="Les langues source et cible ne peuvent pas être identiques.")

        # Vérification si le texte est vide
        input_text = request.text.strip()
        if not input_text:
            raise HTTPException(status_code=400, detail="Le texte à traduire est vide.")

        # Ajout de préfixe pour MarianMT
        input_text = f">>{request.target_lang}<< {input_text}"
        tokenized_text = tokenizer(input_text, return_tensors="pt")

        # Effectuer la traduction
        translated = model.generate(**tokenized_text)

        # Convertir les tokens traduits en texte
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        texte = translated_text.replace('â', 'á').replace('ô', 'ó')
        # Remplacement spécifique de "bongó" par "bongô"
        translated_text = texte.replace('mbonte', 'mbónte')
        translated_text = texte.replace('bongó', 'bongô')
        

        return {
            "source_text": request.text.strip(),
            "translated_text": translated_text,
            "source_language": request.src_lang,
            "target_language": request.target_lang,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la traduction : {str(e)}")

# Endpoint pour l'évaluation
@app.post("/evaluate/")
def evaluate(request: EvaluationRequest):
    try:
        # Vérifier le feedback
        if request.feedback not in ["positive", "negative"]:
            raise HTTPException(status_code=400, detail="Le feedback doit être 'positive' ou 'negative'.")

        # Construire une entrée de feedback
        feedback_data = {
            "translated_text": request.translated_text,
            "feedback": request.feedback,
        }

        # Enregistrer les évaluations dans un fichier JSON
        feedback_file = "feedback.json"
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_data, ensure_ascii=False) + "\n")

        return {"message": "Merci pour votre évaluation !"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'enregistrement : {str(e)}")
