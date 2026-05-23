import os
from dotenv import load_dotenv
from agents.agent_transport import interroger_agent_transport
from agents.agent_loisirs import interroger_agent_loisir

# 1. Charger les clés API présentes dans ton fichier .env
load_dotenv()

print("=" * 60)
print("🤖 APPLICATION ENJOY 212 - TEST DES AGENTS EXPERTS (BINÔME B)")
print("=" * 60)

# 2. TEST DE L'AGENT TRANSPORT
print("\n🚖 [Test] Question à l'Agent Transport : 'Quels choix de transport proposez-vous ?'")
try:
    # On appelle ta fonction
    reponse_transport = interroger_agent_transport("Quels choix de transport proposez-vous ?")
    print(f"\n✍️ Réponse de l'Agent Transport :\n{reponse_transport}")
except Exception as e:
    print(f"❌ Erreur lors du test Transport : {e}")

print("-" * 60)

# 3. TEST DE L'AGENT LOISIRS (Activités + Événements)
print("\n🎭 [Test] Question à l'Agent Loisirs : 'Quelles sont les activités et événements disponibles ?'")
try:
    # On appelle ta fonction
    reponse_loisir = interroger_agent_loisir("Quelles sont les activités et événements disponibles ?")
    print(f"\n✍️ Réponse de l'Agent Loisirs :\n{reponse_loisir}")
except Exception as e:
    print(f"❌ Erreur lors du test Loisirs : {e}")

print("=" * 60)
print("🎉 Fin du test des agents !")