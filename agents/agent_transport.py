import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# Charger les variables d'environnement du fichier .env
load_dotenv()

QDRANT_DIR = os.path.join(os.path.dirname(__file__), "../qdrant_data")
COLLECTION_NAME = "transport"

def interroger_agent_transport(question_touriste: str) -> str:
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
    
    # REGLAGE ICI : On ajoute .points sur l'objet recuperations dans la boucle for
    contexte = ""
    for p in recuperations.points:
        pay = p.payload
        contexte += f"- {pay.get('designation')} : {pay.get('descriptif')}\n"
        
    prompt_systeme = (
        "Tu es un agent expert en transports pour l'application touristique 'Enjoy 212'. "
        "Ton but est d'aider les touristes à comprendre comment se déplacer au Maroc. "
        "Réponds de manière très chaleureuse, polie, et utilise le contexte fourni pour donner des détails précis."
    )
    
    prompt_utilisateur = f"Contexte extrait de notre base :\n{contexte}\n\nQuestion du touriste : {question_touriste}"
    
    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": prompt_systeme},
            {"role": "user", "content": prompt_utilisateur}
        ],
        temperature=0.6
    )
    
    return completion.choices[0].message.content