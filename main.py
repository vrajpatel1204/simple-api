from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import io
import json

app = FastAPI()

IMG_SIZE = 300

# Load TFLite model once
interpreter = tflite.Interpreter(model_path="model.tflite")

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("layer1_class_indices.json", "r") as f:
    labels = {int(k): v for k, v in json.load(f).items()}

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img, dtype=np.float32)
    img = (img / 127.5) - 1  # EfficientNet preprocess
    img = np.expand_dims(img, axis=0)
    return img

def predict(img_tensor):
    interpreter.set_tensor(input_details[0]["index"], img_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]

    pred_idx = int(np.argmax(output))
    pred_class = labels[pred_idx]
    confidence = float(output[pred_idx])

    return pred_class, confidence

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = preprocess(image_bytes)
        breed, conf = predict(img)

        return {
            "success": True,
            "predicted_class": breed,
            "confidence": conf
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
