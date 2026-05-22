import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. Configuration et Connexion
load_dotenv()
qdrant_client = QdrantClient(path="./qdrant_data")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
encoder = SentenceTransformer("intfloat/multilingual-e5-large")

def interroger_expert_gastronomie(question_utilisateur):
    print(f"\n[Agent Gastronomie] Recherche de restaurants pour : '{question_utilisateur}'...")
    
    # Recommandation E5
    query_text = f"query: {question_utilisateur}"
    query_vector = encoder.encode(query_text).tolist()
    
    # 2. Recherche dans Qdrant collection 'gastronomie'
    recuperation = qdrant_client.query_points(
        collection_name="gastronomie",
        query=query_vector,
        limit=3
    )
    
    # 3. Construire le contexte
    contexte_restos = ""
    for hit in recuperation.points:
        p = hit.payload
        contexte_restos += f"- Nom: {p.get('nom')} | Description: {p.get('description')} | Slogan: {p.get('slogan')}\n"
    
    # 4. Prompt pour Groq
    prompt_systeme = (
        "Tu es l'agent expert en Gastronomie et Restauration pour le guide Enjoy 212. "
        "Utilise les informations fournies dans le contexte pour recommander des restaurants et donner faim à l'utilisateur. "
        "Réponds de manière chaleureuse en français."
    )
    
    prompt_utilisateur = f"Contexte des restaurants disponibles :\n{contexte_restos}\n\nQuestion du touriste : {question_utilisateur}"
    
    # 5. Appel à Groq (avec le bon modèle !)
    reponse = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt_systeme},
            {"role": "user", "content": prompt_utilisateur}
        ],
        model="llama-3.1-8b-instant",
        temperature=0.7
    )
    
    return reponse.choices[0].message.content

# Bloc de test local
if __name__ == "__main__":
    test_question = "Je cherche un bon endroit pour manger traditionnel"
    resultat = interroger_expert_gastronomie(test_question)
    print("\n--- Réponse de l'Agent Expert ---")
    print(resultat)