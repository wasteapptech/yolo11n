from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
import io
import base64
from PIL import Image

app = Flask(__name__)

class ObjectDetector:
    def __init__(self, model_path="best.pt"):
        print(f"🔍 Loading model from {model_path}")
        assert os.path.exists(model_path), f"Model file not found: {model_path}"
        self.model = YOLO(model_path)
        print("✅ Model loaded")
    
    def detect_objects(self, image, confidence_threshold=0.25):
        try:
            results = self.model.predict(image, conf=confidence_threshold)
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
            
            return detections, image.width, image.height
        except Exception as e:
            print(f"Error during object detection: {e}")
            return [], None, None

# Load the object detection model
detector = ObjectDetector(model_path="best.pt")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/api/yolo/detect", methods=["POST"])
def detect():
    try:
        if "image" in request.files:
            img = Image.open(request.files["image"].stream).convert("RGB")
        elif request.is_json and "image_base64" in request.json:
            img_data = base64.b64decode(request.json["image_base64"])
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
        else:
            return jsonify({"error": "No image provided"}), 400
        
        detections, image_width, image_height = detector.detect_objects(img)
        
        return jsonify({
            "detections": detections,
            "image_width": image_width,
            "image_height": image_height
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=15016)
