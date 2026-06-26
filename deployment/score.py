import os
import json
import base64
import io
import numpy as np
import joblib
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def init():
    global resnet, scaler, pca, classifier, mlb, transform

    model_dir = os.path.join(os.getenv("AZUREML_MODEL_DIR", "."), "model_assets")

    resnet = models.resnet18(weights="IMAGENET1K_V1")
    resnet.fc = nn.Identity()
    resnet.eval()

    scaler = joblib.load(os.path.join(model_dir, "feature_scaler.pkl"))
    pca = joblib.load(os.path.join(model_dir, "pca_model.pkl"))
    classifier = joblib.load(os.path.join(model_dir, "best_model_final.pkl"))
    mlb = joblib.load(os.path.join(model_dir, "mlb_encoder.pkl"))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Model i preprocessing objekti uspjesno ucitani.")


def run(raw_data):
    try:
        data = json.loads(raw_data)
        image_b64 = data["image"]

        image_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            features = resnet(img_tensor).squeeze().numpy().reshape(1, -1)

        # ISPRAVAN REDOSLIJED: PCA prvo (512->150), zatim StandardScaler (150->150)
        features_pca = pca.transform(features)
        features_scaled = scaler.transform(features_pca)

        probs_per_label = classifier.predict_proba(features_scaled)
        probs = np.array([p[:, 1] if p.shape[1] == 2 else p[:, 0] for p in probs_per_label]).T[0]

        predictions = classifier.predict(features_scaled)[0]
        predicted_genres = [mlb.classes_[i] for i, val in enumerate(predictions) if val == 1]

        result = {
            "predicted_genres": predicted_genres,
            "probabilities": {mlb.classes_[i]: float(probs[i]) for i in range(len(mlb.classes_))}
        }
        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": str(e)})
