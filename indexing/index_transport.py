import json
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Charger les variables d'environnement du fichier .env
load_dotenv()

JSON_PATH = os.path.join(os.path.dirname(__file__), "../produitService_dataset.json")
QDRANT_DIR = os.path.join(os.path.dirname(__file__), "../qdrant_data")
COLLECTION_NAME = "transport"

print("--- Initialisation de l'indexation Transports ---")

client = QdrantClient(path=QDRANT_DIR)
model = SentenceTransformer("intfloat/multilingual-e5-large")

if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
    print(f"Collection '{COLLECTION_NAME}' créée.")
else:
    print(f"La collection '{COLLECTION_NAME}' existe déjà.")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    produits = json.load(f) # Correction ici : produits est directement la liste complète !

points = []

print("Extraction et traitement des données de Transport...")
for idx, prod in enumerate(produits):
    # Sécurisation du filtrage selon le type ou la catégorie
    type_p = prod.get("typeProduit", "")
    cat_p = prod.get("categorieProduit", "")
    
    # On vérifie si c'est un dictionnaire ou une chaîne pour éviter les bugs si c'est un ID
    cat_str = cat_p if isinstance(cat_p, str) else ""

    if "transport" in type_p.lower() or "transport" in cat_str.lower():
        
        lang_fr = {}
        if "translations" in prod and isinstance(prod["translations"], list):
            for trans in prod["translations"]:
                if trans.get("language") == "fr":
                    lang_fr = trans
                    break
                    
        designation = lang_fr.get("designationProduit", "")
        descriptif = lang_fr.get("descriptifProduit", "")
        slogan = lang_fr.get("slogan", "")
        
        texte_complet = f"Désignation: {designation}. Slogan: {slogan}. Description: {descriptif}"
        texte_a_vectoriser = f"passage: {texte_complet}"
        
        embedding = model.encode(texte_a_vectoriser).tolist()
        
        payload = {
            "id": prod.get("_id", {}).get("$oid", str(idx)) if isinstance(prod.get("_id"), dict) else str(idx),
            "designation": designation,
            "descriptif": descriptif,
            "type": type_p,
            "localisation": "Maroc"
        }
        points.append(PointStruct(id=idx, vector=embedding, payload=payload))

if points:
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Indexation réussie : {len(points)} éléments de transport ajoutés avec succès !")
else:
    print("Aucun élément de transport trouvé dans le dataset.")