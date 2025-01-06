from transformers import MarianMTModel, MarianTokenizer

def load_model(model_path):
    """
    Charger le modèle et le tokenizer à partir du chemin spécifié.
    """
    model = MarianMTModel.from_pretrained(model_path, local_files_only=True)
    tokenizer = MarianTokenizer.from_pretrained(model_path, local_files_only=True)
    return model, tokenizer
