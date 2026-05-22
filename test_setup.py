# test_setup.py
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()

# Test Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "system", 
            "content": "Tu es un assistant virtuel expert pour le projet touristique Enjoy212. Tu dois t'exprimer uniquement en Darija marocain (en lettres latines/arabizi), de manière chaleureuse et accueillante."
        },
        {
            "role": "user", 
            "content": "Dis bonjour et souhaite la bienvenue"
        }
    ]
)
print("Groq OK :", response.choices[0].message.content)

# Test Qdrant
qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
print("Qdrant OK :", qdrant.get_collections())

# Test Embedding
model = SentenceTransformer("intfloat/multilingual-e5-large")
vecteur = model.encode("test enjoy 212")
print("Embedding OK : taille =", len(vecteur))