from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
import io
import base64
from PIL import Image

app = Flask(__name__)

# Load model
MODEL_PATH = "model/best.pt"
print(f"🔍 Loading model from {MODEL_PATH}")
assert os.path.exists(MODEL_PATH), f"Model file not found: {MODEL_PATH}"
model = YOLO(MODEL_PATH)
print("✅ Model loaded")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/detect", methods=["POST"])
def detect():
    try:
        if "image" in request.files:
            img = Image.open(request.files["image"].stream).convert("RGB")
        elif request.is_json and "image_base64" in request.json:
            img_data = base64.b64decode(request.json["image_base64"])
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
        else:
            return jsonify({"error": "No image provided"}), 400

        results = model.predict(img, conf=0.25)
        detections = []
        result = results[0]

        if hasattr(result, "boxes") and result.boxes is not None:
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy.tolist()[0]

                detections.append({
                    "id": i,
                    "class_id": cls_id,
                    "class_name": result.names[cls_id],
                    "confidence": conf,
                    "bbox": {
                        "x1": float(x1), "y1": float(y1),
                        "x2": float(x2), "y2": float(y2),
                        "width": float(x2 - x1), "height": float(y2 - y1)
                    }
                })

        return jsonify({
            "detections": detections,
            "image_width": img.width,
            "image_height": img.height
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
