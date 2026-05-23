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
COLLECTION_NAME = "loisirs"

print("--- Initialisation de l'indexation du domaine Loisirs (Activities & Events) ---")

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
    produits = json.load(f)

points = []

print("Combinaison et traitement des Activities et Events...")
for idx, prod in enumerate(produits):
    type_p = prod.get("typeProduit", "")
    
    # CORRECTION DES FILTRES : On utilise exactement les mots du JSON
    if type_p in ["Activitie", "Event"]:
        
        lang_fr = {}
        if "translations" in prod and isinstance(prod["translations"], list):
            for trans in prod["translations"]:
                if trans.get("language") == "fr":
                    lang_fr = trans
                    break
                    
        designation = lang_fr.get("designationProduit", "")
        descriptif = lang_fr.get("descriptifProduit", "")
        slogan = lang_fr.get("slogan", "")
        
        # Sécurité si les champs principaux sont vides
        if not designation:
            designation = lang_fr.get("designation", "")
        if not descriptif:
            descriptif = lang_fr.get("descriptif", "")
            
        texte_complet = f"Nom: {designation}. Slogan: {slogan}. Détails: {descriptif}"
        texte_a_vectoriser = f"passage: {texte_complet}"
        
        embedding = model.encode(texte_a_vectoriser).tolist()
        
        payload = {
            "id": prod.get("_id", {}).get("$oid", str(idx)) if isinstance(prod.get("_id"), dict) else str(idx),
            "designation": designation,
            "descriptif": descriptif,
            "type": type_p, # Gardera "Activitie" ou "Event"
            "localisation": "Maroc"
        }
        
        points.append(PointStruct(id=len(points), vector=embedding, payload=payload))

if points:
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Indexation réussie : {len(points)} éléments ajoutés dans la collection '{COLLECTION_NAME}' !")
else:
    print("⚠️ Aucun élément trouvé. Vérifie l'orthographe de 'Activitie' ou 'Event' dans le JSON.")