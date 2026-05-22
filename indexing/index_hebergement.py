import os
import json
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# 1. Charger les variables d'environnement (notre fameux fichier .env)
load_dotenv()

# 2. Connexion à Qdrant en mode local (pas besoin de serveur allumé !)
print("Connexion à Qdrant...")
qdrant_client = QdrantClient(path="./qdrant_data")

COLLECTION_NAME = "hebergement"

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
# Note : Assure-toi que produitService_dataset.json est bien à la racine de ton projet
json_path = os.path.join(os.getcwd(), "produitService_dataset.json")

if not os.path.exists(json_path):
    print(f"Erreur : Le fichier {json_path} est introuvable à la racine !")
    exit()

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Fichier JSON chargé. Traitement des données d'hébergement...")

points = []
count = 0

# Types de produits considérés comme de l'hébergement
types_hebergement = ["Hotel", "Riad", "Gite"]

for index, item in enumerate(data):
    # On filtre uniquement sur les hébergements
    if item.get("typeProduit") in types_hebergement:
        # Extraire la traduction en français
        translations = item.get("translations", [])
        fr_translation = next((t for t in translations if t.get("language") == "fr"), None)
        
        if fr_translation:
            designation = fr_translation.get("designationProduit", "")
            descriptif = fr_translation.get("descriptifProduit", "")
            slogan = fr_translation.get("slogan", "")
            
            # Construire le texte textuel complet que l'IA va lire pour chercher
            texte_a_vectoriser = f"Type: {item.get('typeProduit')} | Nom: {designation} | Description: {descriptif} | Slogan: {slogan}"
            
            # Recommandation pour le modèle E5 : les passages doivent commencer par "passage: "
            input_text = f"passage: {texte_a_vectoriser}"
            
            # Créer le vecteur (embedding)
            embedding = encoder.encode(input_text).tolist()
            
            # Préparer les métadonnées (payload) pour Qdrant
            payload = {
                "id_produit": item.get("_id", {}).get("$oid", str(index)),
                "type": item.get("typeProduit"),
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
    print(f"Envoi de {len(points)} hébergements vers Qdrant...")
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print("🎉 Indexation de l'Hébergement réussie avec succès !")
else:
    print("Aucune donnée d'hébergement trouvée dans le fichier JSON.")