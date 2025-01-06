import requests

# Définir l'URL de l'API
url = "127.0.0.1:8000/translate/"

# Contenu de la requête
payload = {
    "text": "elle frappe son mari",
    "src_lang": "fr",
    "target_lang": "li"
}

# Envoyer la requête POST
response = requests.post(url, json=payload)

# Afficher la réponse
print("Réponse de l'API :")
print(response.json())
