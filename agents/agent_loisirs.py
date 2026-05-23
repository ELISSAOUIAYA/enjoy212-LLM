import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# Charger les variables d'environnement du fichier .env
load_dotenv()

QDRANT_DIR = os.path.join(os.path.dirname(__file__), "../qdrant_data")
COLLECTION_NAME = "loisirs"

def interroger_agent_loisir(question_touriste: str) -> str:
    qdrant_client = QdrantClient(path=QDRANT_DIR)
    embed_model = SentenceTransformer("intfloat/multilingual-e5-large")
    groq_client = Groq()
    
    # Contrainte E5 : Préfixe "query: "
    question_vectorisee = f"query: {question_touriste}"
    query_vector = embed_model.encode(question_vectorisee).tolist()
    
    # Appel Qdrant
    recuperations = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3
    )
    
    # REGLAGE ICI : Même chose, on ajoute .points pour boucler correctement
    contexte = ""
    for p in recuperations.points:
        pay = p.payload
        contexte += f"- [{pay.get('type')}] {pay.get('designation')} : {pay.get('descriptif')}\n"
        
    prompt_systeme = (
        "Tu es un agent expert en loisirs, animations, culture et événements pour l'application touristique 'Enjoy 212'. "
        "Ton but est de faire découvrir les activités et les festivals/événements à faire au Maroc. "
        "Réponds avec enthousiasme, de manière chaleureuse, et utilise le contexte fourni pour guider le touriste."
    )
    
    prompt_utilisateur = f"Contexte disponible :\n{contexte}\n\nQuestion du touriste : {question_touriste}"
    
    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": prompt_systeme},
            {"role": "user", "content": prompt_utilisateur}
        ],
        temperature=0.6
    )
    return completion.choices[0].message.content