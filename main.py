from fastapi import FastAPI
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer

# Définir le chemin vers votre modèle
model_name = 'https://drive.google.com/drive/folders/12vxuoCVSp3JsQoav1O87tINsTrxpOqIz?usp=sharing'

# Charger le modèle et le tokenizer
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Initialiser FastAPI
app = FastAPI()

# Schéma pour les requêtes
class TranslationRequest(BaseModel):
    text: str
    src_lang: str  # Langue source (fr ou li)
    target_lang: str  # Langue cible (li ou fr)

# Endpoint de test
@app.get("/")
def home():
   
    return {"message": "Bienvenue sur l'API Lingala!"}, {"text sortie":f"{translated_text}"}

# Endpoint pour la traduction
@app.post("/translate/")
def translate(request: TranslationRequest):
    if request.src_lang not in ["fr", "li"] or request.target_lang not in ["fr", "li"]:
        return {"error": "Les langues doivent être 'fr' (Français) ou 'li' (Lingala)."}

    if request.src_lang == request.target_lang:
        return {"error": "Les langues source et cible ne peuvent pas être identiques."}

    # Tokeniser le texte
    tokenized_text = tokenizer(request.text, return_tensors="pt")

    # Effectuer la traduction
    translated = model.generate(**tokenized_text)

    # Convertir les tokens traduits en texte
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    # Retourner le résultat
    return {
        "source_text": request.text,
        "translated_text": translated_text,
        "source_language": request.src_lang,
        "target_language": request.target_lang,
    }
