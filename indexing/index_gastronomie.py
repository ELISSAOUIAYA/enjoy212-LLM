import os
import json
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# 1. Charger les variables d'environnement
load_dotenv()

# 2. Connexion à Qdrant en mode local
print("Connexion à Qdrant...")
qdrant_client = QdrantClient(path="./qdrant_data")  # On réutilise le MÊME dossier de stockage !

COLLECTION_NAME = "gastronomie"

# 3. Initialiser le modèle d'embeddings (taille 1024)
print("Chargement du modèle de vectorisation...")
encoder = SentenceTransformer("intfloat/multilingual-e5-large")

# 4. Créer ou recréer la collection dans Qdrant proprement
print(f"Configuration de la collection '{COLLECTION_NAME}'...")
if qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
    qdrant_client.delete_collection(collection_name=COLLECTION_NAME)

qdrant_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)

# 5. Charger les données du fichier JSON
json_path = os.path.join(os.getcwd(), "produitService_dataset.json")

if not os.path.exists(json_path):
    print(f"Erreur : Le fichier {json_path} est introuvable à la racine !")
    exit()

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Fichier JSON chargé. Traitement des données de gastronomie...")

points = []
count = 0

for index, item in enumerate(data):
    # On filtre uniquement sur la restauration cette fois
    if item.get("typeProduit") == "Restauration":
        # Extraire la traduction en français
        translations = item.get("translations", [])
        fr_translation = next((t for t in translations if t.get("language") == "fr"), None)
        
        if fr_translation:
            designation = fr_translation.get("designationProduit", "")
            descriptif = fr_translation.get("descriptifProduit", "")
            slogan = fr_translation.get("slogan", "")
            
            # Construire le texte textuel complet pour la restauration
            texte_a_vectoriser = f"Type: Restauration | Nom: {designation} | Description: {descriptif} | Slogan: {slogan}"
            input_text = f"passage: {texte_a_vectoriser}"
            
            # Créer le vecteur
            embedding = encoder.encode(input_text).tolist()
            
            # Préparer les métadonnées (payload) pour Qdrant
            payload = {
                "id_produit": item.get("_id", {}).get("$oid", str(index)),
                "type": "Restauration",
                "nom": designation,
                "description": descriptif,
                "prix_defaut": item.get("tarifUHTPardefaut", 0),
                "image": item.get("imageProduit", "")
            }
            
            # Ajouter le point structuré
            points.append(
                PointStruct(
                    id=count,
                    vector=embedding,
                    payload=payload
                )
            )
            count += 1

# 6. Envoyer les données par lots dans Qdrant
if points:
    print(f"Envoi de {len(points)} restaurants vers Qdrant...")
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print("🎉 Indexation de la Gastronomie réussie avec succès !")
else:
    print("Aucune donnée de restauration trouvée dans le fichier JSON.")