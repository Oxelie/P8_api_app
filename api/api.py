from flask import Flask, jsonify, request
import os
import numpy as np
from tensorflow import keras
import io
from PIL import Image   
import tensorflow as tf
from keras.applications.mobilenet_v3 import preprocess_input as mnv3_preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from .custom_object import DiceFocalLoss
from .classe_dataset import ImageSegmentationDataset
import pathlib
from .utils_p8 import labels
import matplotlib.pyplot as plt
import base64
import requests
import time
import gdown
from azure.storage.blob import BlobServiceClient

# MAIN_DIR = "https://drive.google.com/drive/folders/1xR3fMt_t8op1NqHP6UAOczNtEXi0FdUA"
# TEST_DIR = "./test"
# # Charge le modèle entraîné
# MODEL_PATH = "model_ResNet50_UNet.keras"
# # MODEL_PATH = "https://drive.google.com/file/d/1oe94iBbXN2Gdt7wGNwciwDKq4rgiF8IA"

# # code à voir pour télécharger le modèle depuis google drive (si besoin) et le charger ensuite

# def download_large_file(file_id, destination):
#     session = requests.Session()
#     url = f"https://drive.google.com/uc?export=download&id={file_id}"
#     # response = session.get(url, stream=True)

#     # # Récupère le token de confirmation si présent
#     # token = None
#     # for key, value in response.cookies.items():
#     #     if key.startswith("download_warning"):
#     #         token = value

#     # if token:
#     #     response = session.get(url, params={"confirm": token}, stream=True)

#     # with open(destination, "wb") as f:
#     #     for chunk in response.iter_content(32768):
#     #         f.write(chunk)
    
#     resp = session.get(url, stream=True)
#     if not resp.ok:
#         raise RuntimeError(f"download failed: {resp.status_code}")
#     content_type = resp.headers.get("Content-Type","")
#     if "html" in content_type.lower():
#         raise RuntimeError("Drive returned HTML instead of the model file")
#     # écrire dans temp puis renommer
#     with open(destination + ".tmp", "wb") as f:
#         for chunk in resp.iter_content(32768):
#             f.write(chunk)
#     os.replace(destination + ".tmp", destination)
            
# download_large_file("1oe94iBbXN2Gdt7wGNwciwDKq4rgiF8IA", "./model_ResNet50_UNet.keras")

# def download_from_drive(file_id, destination, max_retries=3):
#     """
#     Télécharge un fichier depuis Google Drive en gérant la confirmation.
    
#     Args:
#         file_id: ID du fichier Drive
#         destination: chemin de destination local
#         max_retries: nombre de tentatives en cas d'erreur
    
#     Raises:
#         RuntimeError: si le téléchargement échoue
#     """
#     session = requests.Session()
#     url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
#     for attempt in range(max_retries):
#         try:
#             print(f"[Tentative {attempt + 1}/{max_retries}] Téléchargement depuis Drive...")
            
#             # Première requête
#             response = session.get(url, stream=True, timeout=30)
            
#             # Vérifier le statut
#             if response.status_code != 200:
#                 raise RuntimeError(f"HTTP {response.status_code}: {response.reason}")
            
#             # Chercher le token de confirmation dans les cookies
#             token = None
#             for key, value in response.cookies.items():
#                 if key.startswith("download_warning"):
#                     token = value
#                     break
            
#             # Si token trouvé, relancer la requête avec confirmation
#             if token:
#                 print("  → Token de confirmation trouvé, confirmation en cours...")
#                 response = session.get(
#                     url, 
#                     params={"confirm": token},
#                     stream=True,
#                     timeout=30
#                 )
#                 if response.status_code != 200:
#                     raise RuntimeError(f"Erreur après confirmation: HTTP {response.status_code}")
            
#             # Vérifier le Content-Type
#             content_type = response.headers.get("Content-Type", "").lower()
#             if "html" in content_type or "text" in content_type:
#                 # Chercher un indice dans le texte
#                 text_sample = response.text[:500]
#                 if "login" in text_sample.lower() or "signin" in text_sample.lower():
#                     raise RuntimeError("Accès refusé: le fichier n'est peut-être pas public ou vous êtes déconnecté")
#                 else:
#                     raise RuntimeError(f"Drive a retourné du HTML au lieu du fichier (Content-Type: {content_type})")
            
#             # Vérifier la taille du fichier
#             content_length = response.headers.get("Content-Length")
#             if content_length:
#                 size_mb = int(content_length) / (1024 * 1024)
#                 print(f"  → Taille: {size_mb:.1f} MB")
            
#             # Écrire dans un fichier temporaire
#             tmp_destination = destination + ".tmp"
#             bytes_downloaded = 0
            
#             with open(tmp_destination, "wb") as f:
#                 for chunk in response.iter_content(chunk_size=32768):
#                     if chunk:
#                         f.write(chunk)
#                         bytes_downloaded += len(chunk)
            
#             # Vérifier que le fichier n'est pas vide
#             if os.path.getsize(tmp_destination) == 0:
#                 raise RuntimeError("Fichier téléchargé vide (0 bytes)")
            
#             # Vérifier que c'est un ZIP valide (magic bytes pour .keras)
#             with open(tmp_destination, "rb") as f:
#                 magic = f.read(2)
            
#             if magic != b"PK":
#                 raise RuntimeError(f"Fichier téléchargé n'est pas un archive ZIP valide (magic bytes: {magic})")
            
#             # Remplacer le fichier final
#             os.replace(tmp_destination, destination)
            
#             print(f"✓ Téléchargement réussi: {destination}")
#             return True
            
#         except requests.exceptions.Timeout:
#             print(f"  ✗ Timeout - nouvelle tentative dans 5s...")
#             time.sleep(5)
#         except requests.exceptions.RequestException as e:
#             print(f"  ✗ Erreur réseau: {e}")
#             time.sleep(5)
#         except RuntimeError as e:
#             print(f"  ✗ {e}")
#             if attempt < max_retries - 1:
#                 time.sleep(5)
#             else:
#                 raise
    
#     raise RuntimeError(f"Impossible de télécharger après {max_retries} tentatives")

# download_from_drive("1oe94iBbXN2Gdt7wGNwciwDKq4rgiF8IA", "./model_ResNet50_UNet.keras")

# actuellement dans le dossier /Users/stephanieduhem/Documents/_DIPLOMES_CURSUS_/MASTER_AI_ENGINEER/openclassroom/projet_8/P8_Segmentation_Images/mlf_1/models 
# + sous dossiers aux noms des modèles REsNet50_Unet_50epochs_data_augm

# def ensure_model_downloaded():
#     """Télécharge le modèle depuis Drive s'il n'existe pas localement."""
#     if not os.path.exists(MODEL_PATH):
#         print(f"Modèle non trouvé. Téléchargement depuis Drive...")
#         try:
#             gdown.download(
#                 MAIN_DIR,
#                 MODEL_PATH,
#                 quiet=False
#             )
#             print(f"✓ Modèle téléchargé: {MODEL_PATH}")
#         except Exception as e:
#             raise RuntimeError(f"Impossible de télécharger le modèle: {e}")
#     else:
#         print(f"✓ Modèle trouvé localement: {MODEL_PATH}")

# def ensure_test_images_downloaded():
#     """Télécharge les images de test depuis Drive s'il n'existe pas localement."""
#     if not os.path.exists(TEST_DIR) or len(os.listdir(TEST_DIR)) == 0:
#         print(f"Images de test non trouvées. Téléchargement...")
#         os.makedirs(TEST_DIR, exist_ok=True)
#         try:
#             # Remplacer par l'ID du dossier/fichier Drive contenant les images
#             gdown.download_folder(
#                 "https://drive.google.com/drive/folders/1wC6tviUc-8NjkgVbOLqjRxVwednPRhDS",
#                 output=TEST_DIR,
#                 quiet=False
#             )
#             print(f"✓ Images de test téléchargées: {TEST_DIR}")
#         except Exception as e:
#             print(f"⚠ Impossible de télécharger les images: {e}")
#     else:
#         print(f"✓ Images de test trouvées localement: {TEST_DIR}")

# # Appeler au démarrage du module (avant de charger le modèle)
# ensure_model_downloaded()
# ensure_test_images_downloaded()

# # définir le nom du modèle - en récupérant le nom du modèle à partir du nom du fichier du modèle chargé
# model_name = os.path.basename(MODEL_PATH).split(".")[0].lower() 
# print("model_name:", model_name)

# # Puis charger le modèle normalement
# model = tf.keras.models.load_model(
#     MODEL_PATH,
#     custom_objects={"DiceFocalLoss": DiceFocalLoss},
#     compile=False
# )



# # Taille attendue par le modèle
# TARGET_SIZE = (256, 512)


# # path vers les nouveaux dossier train et test
# test_dir = pathlib.Path("../test")


# # récupération des chemins pour les images et les masques du test
# image_paths_test = sorted(list(test_dir.glob("*leftImg8bit.png")))
# mask_paths_test = sorted(list(test_dir.glob("*labelIds.png")))
# test_paths = list(zip(image_paths_test, mask_paths_test))

# test_dataset = ImageSegmentationDataset(
#     paths=test_paths,
#     labels=labels,
#     batch_size=4,
#     augmentations=False,        # Pas d'augmentation pour le test
#     normalize=True,
#     shuffle=False,
#     label_onehot=False,
#     sample_weights=True,        # Poids de classes activés
#     model_name="resnet50_unet") #pour normalisation adpatée au modèle


# Chemin absolu du répertoire du module API
API_DIR = pathlib.Path(__file__).parent.absolute()

# Dossier de cache local pour les fichiers téléchargés
CACHE_DIR = API_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = CACHE_DIR / "model_MobileNetV3_UNet_50epochs.keras"
TEST_DIR = CACHE_DIR / "test"

# Google Drive — images de test uniquement
TEST_DRIVE_URL = "https://drive.google.com/drive/folders/16P7bh3J9Cj5Vdrt1Zk3jUWccYO7fKs_l"

# Azure Blob Storage — modèle
AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_CONTAINER_NAME = "models"
AZURE_BLOB_NAME = "model_MobileNetV3_UNet_50epochs.keras"

TARGET_SIZE = (256, 512)


def ensure_model_downloaded():
    """Télécharge le modèle depuis Azure Blob Storage s'il n'existe pas en cache local."""
    if MODEL_PATH.exists():
        print(f"✓ Modèle trouvé en cache: {MODEL_PATH}")
        return
    if not AZURE_CONNECTION_STRING:
        raise RuntimeError(
            "Variable d'environnement AZURE_STORAGE_CONNECTION_STRING manquante."
        )
    print("Modèle non trouvé. Téléchargement depuis Azure Blob Storage...")
    try:
        blob_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING) \
            .get_blob_client(container=AZURE_CONTAINER_NAME, blob=AZURE_BLOB_NAME)
        with open(MODEL_PATH, "wb") as f:
            f.write(blob_client.download_blob().readall())
        print(f"✓ Modèle téléchargé: {MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(f"Impossible de télécharger le modèle depuis Azure: {e}") from e


def ensure_test_images_downloaded():
    """Télécharge les images de test depuis Google Drive si absentes du cache local."""
    if TEST_DIR.exists() and len(list(TEST_DIR.glob("*.png"))) > 0:
        print(f"✓ Images de test trouvées localement: {TEST_DIR}")
        return
    print("Images de test non trouvées. Téléchargement depuis Google Drive...")
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    try:
        gdown.download_folder(TEST_DRIVE_URL, output=str(TEST_DIR), quiet=False)
        print(f"✓ Images de test téléchargées: {TEST_DIR}")
    except Exception as e:
        print(f"⚠ Impossible de télécharger les images: {e}")


app = Flask(__name__)

# Ressources chargées une seule fois à la première requête (lazy loading)
_model = None
_test_dataset = None


def _load_resources():
    """Télécharge et charge le modèle + dataset au premier appel."""
    global _model, _test_dataset
    if _model is not None and _test_dataset is not None:
        return _model, _test_dataset

    ensure_model_downloaded()
    ensure_test_images_downloaded()

    model_name = MODEL_PATH.stem.lower()
    print("model_name:", model_name)

    _model = tf.keras.models.load_model(
        str(MODEL_PATH),
        custom_objects={"DiceFocalLoss": DiceFocalLoss},
        compile=False
    )

    image_files = sorted(list(TEST_DIR.glob("*leftImg8bit.png")))
    image_paths_test = []
    mask_paths_test = []
    for img_path in image_files:
        mask_path = img_path.parent / str(img_path.name).replace(
            "_leftImg8bit.png", "_gtFine_labelIds.png"
        )
        if mask_path.exists():
            image_paths_test.append(img_path)
            mask_paths_test.append(mask_path)

    test_paths = list(zip(image_paths_test, mask_paths_test))
    _test_dataset = ImageSegmentationDataset(
        paths=test_paths,
        labels=labels,
        batch_size=4,
        augmentations=False,
        normalize=True,
        shuffle=False,
        label_onehot=False,
        sample_weights=True,
        model_name="mobilenetv3small_unet"
    )
    return _model, _test_dataset

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/preload", methods=["GET"])
def list_images():
    _, test_dataset = _load_resources()
    return jsonify([{"index": i, "path": str(p).replace("../test/", "")} for i, p in enumerate(test_dataset.image_paths)])



# sélectionner une image de test parmi la liste des images de test disponibles (endpoint /preload)
@app.route("/select_img", methods=["POST"])
def select_image():
   
    # # récupérer en plus le mask vérité terrain _gtFine_color.png  
    # test_dataset.image_paths[request.json["image_index"]] # chemin de l'image sélectionnée
    # test_dataset.mask_paths[request.json["image_index"]] # chemin du masque de vérité terrain correspondant à l'image sélectionnée
    
     # pour charger l'image et le masque de vérité terrain sélectionnés dans le dataset (pour affichage dans l'app streamlit)
    
    _, test_dataset = _load_resources()
    _, mask, paths = test_dataset.get_image_and_mask(request.json["image_index"])
    orig_img = np.array(Image.open(test_dataset.image_paths[request.json["image_index"]]).convert("RGB").resize((TARGET_SIZE[1], TARGET_SIZE[0]), Image.BILINEAR)
)

    
    num_classes= 8
    cmap = plt.get_cmap("tab10", num_classes)

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))

    axs[0].imshow(orig_img)
    axs[0].set_title("Original Image")
    axs[1].imshow(mask, cmap=cmap, vmin=0, vmax=num_classes-1)
    axs[1].set_title("Ground Truth Mask")
    
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    

    bbox_orig = axs[0].get_tightbbox(fig.canvas.get_renderer())
    bbox_orig = bbox_orig.transformed(fig.dpi_scale_trans.inverted())
    
    bbox_mask = axs[1].get_tightbbox(fig.canvas.get_renderer())
    bbox_mask = bbox_mask.transformed(fig.dpi_scale_trans.inverted())
    
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', 
                bbox_inches=bbox_orig
                )   
    buf.seek(0)
    
    buf_mask = io.BytesIO()
    fig.savefig(buf_mask, format='png', 
                bbox_inches=bbox_mask
                )   
    buf_mask.seek(0)
    
    encoded_orig = base64.b64encode(buf.read()).decode('utf-8')
    encoded_mask = base64.b64encode(buf_mask.read()).decode('utf-8')

    return jsonify({
        "image": encoded_orig,
        "mask": encoded_mask
    })  
    

# faire un 2eme API pour la prédiction du masque (endpoint /predict) 
@app.route("/predict", methods=["POST"])
def predict_mask():    
    
    model, test_dataset = _load_resources()
    img_mask_pred = test_dataset.show_prediction(model, request.json["image_index"])
    
    return jsonify({"img_mask_pred": img_mask_pred})
    
    # global selected_img
    # # ici on va recevoir le nom de l'image de test à prédire
    # # selected_img = jsonify(usable_img)[0] 
    # # charger l'image de test correspondante depuis le dossier data/test
    # img_path = os.path.join("../data/test", selected_img)
    # open_img = Image.open(img_path).convert("RGB")
    # # preprocess de l'image pour la mettre au format attendu par le modèle (taille, normalisation)
    # resized_img = open_img.resize(TARGET_SIZE)
    # prep_img = preprocess_img(resized_img)    
    # # ensuite on va faire la prédiction du masque pour cette image
    # # et enfin on va retourner le masque prédit (sous forme d'image ou de tableau numpy)
    # mask_pred = model.predict(np.expand_dims(prep_img, axis=0))
    # mask_pred = np.argmax(mask_pred.squeeze(), axis=-1)
    # return mask_pred 

# sachant que  l'affichage des images de test + image du mask réel + image du mask prédit est dans une autre app (streamlit)
# on va créer une classe à part pour l'affichage des images et des masques dans l'app


if __name__ == "__main__":
    app.run(port = 4444, debug=True)