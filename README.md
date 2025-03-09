# Dog Breed Classifier

This project is a complete machine learning pipeline for dog breed classification using deep learning techniques. It includes data preparation, model training, and deployment as a RESTful API service.

## Project Structure

```
|-- deployment
  |-- cloudbuild.yaml            # Cloud Build configuration for CI/CD
|-- data_collection
  |-- organizer.py               # Script to organize dataset into train/val/test splits
|-- modeling
  |-- model.py                   # Model architecture and training pipeline
|-- backend
  |-- app.py                     # FastAPI server implementation
  |-- dockerfile                 # Docker configuration for containerization
  |-- dog_breed_model_quantized.tflite  # Optimized TensorFlow Lite model
  |-- label_map.json             # Mapping between class indices and breed names
  |-- requirements.txt           # Python dependencies
  |-- test_api.py                # Script to test API endpoints
  |-- test_script.sh             # Shell script for testing
  |-- test_ui.html               # Simple UI for visual testing
  |-- test_images/               # Sample images for testing
    |-- *.jpg
```

## Data Processing

The project uses the Dog Breed dataset from Kaggle, which contains images of various dog breeds. The `organizer.py` script:

- Organizes the original dataset into train, validation, and test splits
- Maintains proper directory structure for TensorFlow's data generators
- Uses a 70/15/15 split by default for training/validation/test
- Preserves class balance across splits

## Model Architecture

The model architecture (`model.py`) employs transfer learning using MobileNetV2 as a base model:

- Pre-trained MobileNetV2 (trained on ImageNet) serves as the feature extractor
- Custom classification layers are added on top
- Two-phase training:
  1. Train only the custom classification layers with the base model frozen
  2. Fine-tune the last 20 layers of the base model with a reduced learning rate
- Data augmentation is applied during training to improve generalization
- Early stopping and model checkpointing to prevent overfitting
- Model is exported in both TensorFlow SavedModel and quantized TensorFlow Lite formats

## API Backend

The backend implements a RESTful API service using FastAPI:

- `/predict` endpoint accepts image uploads and returns top breed predictions
- Input images are preprocessed for model inference
- Quantized TFLite model for efficient inference
- CORS middleware for browser compatibility
- Prediction results include breed names and confidence scores

## Testing

Testing components include:

- `test_api.py`: Tests the API with various dog images and measures response time
- `test_script.sh`: Shell script for automated testing
- `test_ui.html`: Simple HTML interface for visual testing of API functionality
- Sample dog images to validate predictions

## Deployment

The project includes cloud deployment configuration with:

- Docker containerization for portable deployment
- Cloud Build configuration for CI/CD pipelines

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- FastAPI
- Pillow
- NumPy
- scikit-learn

### Installation

1. Clone the repository
2. Install the required packages:
```bash
pip install -r backend/requirements.txt
```

### Training a Model

1. Organize your dataset:
```bash
python data_collection/organizer.py
```

2. Train the model:
```bash
python modeling/model.py
```

### Running the API Locally

```bash
cd backend
uvicorn app:app --host 0.0.0.0 --reload
```

The API will be available at `http://localhost:8000`

### Testing the API

```bash
cd backend
python test_api.py
```

## Acknowledgments

- Dog Breed dataset from Kaggle
- TensorFlow and Keras for model training
- FastAPI for the backend service
