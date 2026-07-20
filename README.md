# Projet 8 — API Flask de segmentation d'images pour voiture autonome

API de démonstration pour le projet de segmentation sémantique d'images urbaines (véhicule autonome).  
Développée avec Flask + Gunicorn, containerisée via Docker, déployée sur Azure App Service.

---

## Contexte

Ce dépôt contient uniquement le **backend** (API Flask).  
Le frontend Streamlit est maintenu dans le dépôt `P8_front_app`.  
Le code d'entraînement des modèles est maintenu dans le dépôt `P8_Segmentation_Images`.

---

## Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| `GET` | `/health` | Vérification que l'API est démarrée — retourne `{"status": "ok"}` instantanément |
| `GET` | `/preload` | Charge le modèle + les images de test, retourne la liste disponible (index + chemin) |
| `POST` | `/select_img` | Retourne l'image originale et le masque réel pour un index donné |
| `POST` | `/predict` | Génère le masque prédit par le modèle pour un index donné |

### Exemples d'appels

```bash
# Vérifier que l'API répond
curl https://projet8-api-e8a2apgjhhehekc4.swedencentral-01.azurewebsites.net/health

# Précharger le modèle + lister les images disponibles
curl https://projet8-api-e8a2apgjhhehekc4.swedencentral-01.azurewebsites.net/preload

# Sélectionner l'image index 0
curl -X POST https://projet8-api-e8a2apgjhhehekc4.swedencentral-01.azurewebsites.net/select_img \
  -H "Content-Type: application/json" \
  -d '{"image_index": 0}'

# Lancer une prédiction sur l'image index 0
curl -X POST https://projet8-api-e8a2apgjhhehekc4.swedencentral-01.azurewebsites.net/predict \
  -H "Content-Type: application/json" \
  -d '{"image_index": 0}'
```

> `/predict` peut prendre 30 à 60 secondes sur le plan B1 (CPU uniquement).

---

## Architecture

```
P8_api_app/
│
├── api/
│   ├── api.py              # Application Flask — routes et logique métier
│   ├── classe_dataset.py   # Générateur de données (ImageSegmentationDataset)
│   ├── custom_object.py    # DiceFocalLoss + DiceMetric (objets Keras custom)
│   └── utils_p8.py         # Fonctions utilitaires (visualisation des masques)
│
├── Dockerfile              # Image Docker Python 3.10-slim
├── Procfile                # Commande gunicorn pour Azure App Service
└── requirements.txt        # Dépendances Python
```

---

## Chargement des ressources (lazy loading)

Le modèle (~17 Mo) et les images de test sont **téléchargés au premier appel** (pas au démarrage), afin d'éviter un timeout de démarrage sur Azure App Service :

- **Modèle** : téléchargé depuis Azure Blob Storage (`stprojet8seg`)
- **Images de test** : téléchargées depuis Google Drive (11 images Frankfurt, Cityscapes)

Le endpoint `/health` répond immédiatement sans déclencher ce chargement — utilisé par Azure pour les health checks.

---

## Déploiement Azure

| Ressource | Détail |
|-----------|--------|
| App Service | `projet8-api` (Sweden Central) |
| Plan | B1 (CPU, 1,75 Go RAM) |
| Container Registry | `acrprojet8.azurecr.io` |
| Image Docker | `acrprojet8.azurecr.io/projet8-api:latest` |
| Blob Storage | `stprojet8seg` — conteneur `models` (modèle .keras) |

### Build et déploiement

```bash
# Build et push de l'image vers Azure Container Registry
az acr build \
  --registry acrprojet8 \
  --image projet8-api:latest .

# Redémarrer l'App Service pour prendre en compte la nouvelle image
az webapp restart --name projet8-api --resource-group rg-projet8
```

### Gestion des crédits Azure

> **État actuel :** l'App Service est **arrêtée** (économies — coût Azure même sans trafic). Redémarrage en moins d'une minute avant démonstration.

```bash
# Redémarrer avant démonstration
az webapp start --name projet8-api --resource-group rg-projet8

# Remettre en pause après démonstration
az webapp stop --name projet8-api --resource-group rg-projet8
```

---

## Lancer l'API en local

```bash
# Installer les dépendances
pip install -r requirements.txt

# Démarrer l'API
gunicorn "api.api:app" --bind 0.0.0.0:4444 --timeout 300 --workers 1
```

L'API est alors accessible sur `http://127.0.0.1:4444`.

---

## Dépendances principales

```
flask
gunicorn
tensorflow-cpu==2.16.2
keras==3.10.0
numpy==1.26.4
pillow
opencv-python-headless==4.10.0.84
albumentations
azure-storage-blob
gdown
matplotlib
```
