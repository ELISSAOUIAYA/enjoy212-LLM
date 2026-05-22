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

def interroger_expert_hebergement(question_utilisateur):
    print(f"\n[Agent Hébergement] Recherche d'hôtels pour : '{question_utilisateur}'...")
    
    # Recommandation E5 : les requêtes de recherche doivent commencer par "query: "
    query_text = f"query: {question_utilisateur}"
    query_vector = encoder.encode(query_text).tolist()
    
    # 2. Recherche dans Qdrant avec query_points (compatible mode local)
    recuperation = qdrant_client.query_points(
        collection_name="hebergement",
        query=query_vector,
        limit=3 # On récupère les 3 meilleurs résultats
    )
    
    # 3. Construire le contexte pour le LLM
    contexte_hotels = ""
    for hit in recuperation.points:
        p = hit.payload
        contexte_hotels += f"- Nom: {p.get('nom')} | Description: {p.get('description')} | Prix indicatif: {p.get('prix_defaut')} DH\n"
    
    # 4. Prompt pour Groq
    prompt_systeme = (
        "Tu es l'agent expert en Hébergement pour le guide touristique Enjoy 212. "
        "Utilise les informations fournies dans le contexte ci-dessous pour répondre de manière chaleureuse, "
        "précise et professionnelle à l'utilisateur. Si tu ne trouves pas de réponse pertinente dans le contexte, dis-le poliment."
    )
    
    prompt_utilisateur = f"Contexte des hôtels disponibles :\n{contexte_hotels}\n\nQuestion du touriste : {question_utilisateur}"
    
    # 5. Appel à Groq
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
    test_question = "Je cherche un hôtel ou un riad confortable à Tanger"
    resultat = interroger_expert_hebergement(test_question)
    print("\n--- Réponse de l'Agent Expert ---")
    print(resultat)