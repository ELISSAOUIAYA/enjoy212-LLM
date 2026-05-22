
# Enjoy 212 LLM

Composante IA du projet Enjoy212 (PFE). Architecture multi-agents comprenant un LLM Global orchestrateur et 5 à 6 LLMs spécifiques par domaine (Hébergement, Transport, Activités...) avec intégration RAG.

## 🏗️ Architecture du Système
* **LLM Global :** Point d'entrée unique de l'utilisateur. Il répond aux requêtes générales et l'oriente intelligemment vers un agent spécifique lorsqu'il demande des informations pointues sur un domaine.
* **LLMs Spécifiques :** Modèles experts alimentés par la taxonomie et dédiés à l'approfondissement de chaque secteur (Hébergement, Transport, Gastronomie, Activités...).

---

## 🚀 Setup (à faire une seule fois)

1. **Cloner le repo**
```bash
git clone [https://github.com/ELISSAOUIAYA/enjoy212-LLM.git](https://github.com/ELISSAOUIAYA/enjoy212-LLM.git)

```

2. **Installer les dépendances**

```bash
pip install -r requirements.txt

```

3. **Créer ton fichier de configuration .env**

```bash
cp .env.example .env

```

> 💡 Remplir le fichier `.env` avec les vraies clés API (demande à Aya EL ISSAOUI).

4. **Vérifier que l'étape 1 fonctionne**

```bash
python test_setup.py

```

---

## 👥 Répartition du Travail & Dossiers

Les scripts d'indexation et d'interrogation pour chaque domaine sont séparés pour éviter les conflits sur Git.

* **Binôme A :** Focus sur les domaines de l'hébergement et de la restauration.
* 📁 Fichiers : `indexing/index_hebergement.py` + `indexing/index_gastronomie.py`


* **Binôme B :** Focus sur les domaines des transports et des activités touristiques.
* 📁 Fichiers : `indexing/index_transport.py` + `indexing/index_activites.py`



```

```
