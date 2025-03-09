# app.py
import io
import json
from typing import Any

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI(title="Dog Breed Classifier API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model: tf.lite.Interpreter | None = None
label_map: dict[str, str] | None = None
input_details: list[dict[str, Any]] | None = None
output_details: list[dict[str, Any]] | None = None


@app.on_event("startup")
async def startup_event() -> None:
    global model, label_map, input_details, output_details
    # Load the TF Lite model
    interpreter = tf.lite.Interpreter(model_path="dog_breed_model_quantized.tflite")
    interpreter.allocate_tensors()
    model = interpreter

    # Get input and output tensors
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    # Load the label map
    with open("label_map.json") as f:
        label_map = json.load(f)


def preprocess_image(
    image: Image.Image, target_size: tuple[int, int] = (224, 224)
) -> np.ndarray:
    """Preprocess the image for model prediction."""
    # Resize image
    image = image.resize(target_size)

    # Convert to array and normalize
    image_array = np.array(image)
    # Handle grayscale images
    if len(image_array.shape) == 2:
        image_array = np.stack((image_array,) * 3, axis=-1)
    elif image_array.shape[2] == 4:
        # Remove alpha channel
        image_array = image_array[:, :, :3]

    # Normalize to [0,1]
    image_array = image_array.astype(np.float32) / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


@app.post("/predict/")
async def predict(file: UploadFile = None) -> dict[str, list[dict[str, Any]]]:
    """
    Predict the dog breed from an uploaded image.
    """
    if file is None:
        raise HTTPException(status_code=400, detail="No file provided")
    if (
        model is None
        or input_details is None
        or output_details is None
        or label_map is None
    ):
        raise HTTPException(status_code=500, detail="Model not initialized")

    # Read and process the image
    image_content = await file.read()
    image = Image.open(io.BytesIO(image_content))
    processed_image = preprocess_image(image)

    # Make prediction using TF Lite
    model.set_tensor(input_details[0]["index"], processed_image)
    model.invoke()
    predictions = model.get_tensor(output_details[0]["index"])

    # Get top 3 predictions
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    top_3_predictions = [
        {"breed": label_map[str(idx)], "confidence": float(predictions[0][idx])}
        for idx in top_3_indices
    ]

    return {"predictions": top_3_predictions}


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Welcome to the Dog Breed Classifier API"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)
