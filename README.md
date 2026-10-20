🦎 ZooTag: Museum Specimen Image Classification

ZooTag is a deep learning and transfer learning image classification system developed for the Evolutionsmuseet Zoology Section (Uppsala University). The model classifies visitor photographs of museum animal specimens into seven broad taxonomic categories in real time to power an interactive mobile visitor engagement application.

📌 Project Overview

Museum exhibits present unique computer vision challenges, including glass reflections, ambient lighting variations, fluid-preserved wet jars, pinned entomology displays, and partial views captured by rushing visitors.

To solve this cost-effectively and accurately, we built a two-stage fine-tuned MobileNetV2 convolutional neural network (ZooTag_MobileNetV2) exported to TensorFlow Lite (.tflite) for lightweight mobile deployment, paired with an OpenAPI 3.0 RESTful web service specification.

Supported Taxonomic Categories (Exact Output Labels)

AMPHIBIAN

BIRD

FISH

INSECT

MAMMAL

MARINE_INVERTEBRATE (jellyfish, sea cucumbers, mussels, brittle stars/starfish, marine slugs/snails)

REPTILE

📂 Repository Structure

├── notebooks/
│   └── ZooTag_Transfer_Learning.ipynb   # End-to-end data processing, training, evaluation & TFLite export
├── models/
│   └── zootag_model.tflite              # Final optimized TensorFlow Lite model for mobile deployment
├── api/
│   └── zootag_openapi.yaml              # OpenAPI 3.0.3 REST API specification
├── docs/
│   ├── ZooTag_Model_Card.pdf            # Formal Model Card reporting performance & limitations
│   └── confusion_matrix.png             # Confusion matrix on the isolated 20% test set
└── README.md


📊 Dataset & Preprocessing Pipeline

Data Acquisition: 1,218 raw specimen images captured on-site at the Evolutionsmuseet Zoology Section (Villavägen 9, Uppsala) across multiple smartphone cameras and formats (.HEIC, .JPG, .PNG).

Format Standardization & 80/20 Split:

All images (including Apple .HEIC files via pillow-heif) were standardized to 3-channel RGB .JPG format.

Divided into an isolated 80% TRAINING/ split and 20% TESTING/ (247 unseen images) split prior to augmentation to prevent data leakage.

Class Imbalance Mitigation:

Offline Minority Oversampling: Minority classes in TRAINING/ were augmented (random horizontal flips, $\pm 20^\circ$ rotations, zooms, and brightness/contrast shifts) to reach a minimum baseline of 200 training images per class (1,541 balanced training images).

Balanced Class Weights & Label Smoothing: Combined with inverse-frequency class_weight penalties and label_smoothing=0.05 during training.

🧠 Model Architecture & Training Strategy

Input Contract: Accepts raw image tensors of shape (224, 224, 3) with float32 pixel values in the range [0.0, 255.0] to integrate directly with automated evaluation pipelines.

Internal Preprocessing: Embedded tf.keras.layers.Rescaling(scale=1./127.5, offset=-1.0, name="internal_rescaling") inside the model graph to scale inputs to [-1.0, 1.0] for MobileNetV2.

Backbone: Pre-trained MobileNetV2 (weights='imagenet', include_top=False).

Classification Head:
GlobalAveragePooling2D $\rightarrow$ BatchNormalization $\rightarrow$ Dropout(0.4) $\rightarrow$ Dense(256, activation='relu') $\rightarrow$ BatchNormalization $\rightarrow$ Dropout(0.3) $\rightarrow$ Dense(7, activation='softmax')

Two-Stage Training:

Phase 1 (Feature Extraction): Trained the custom classification head for 15 epochs with the MobileNetV2 base frozen (Adam, lr=1e-3).

Phase 2 (Fine-Tuning): Unfroze the top 40 layers of MobileNetV2 (keeping BatchNormalization layers frozen) and fine-tuned with a low learning rate (Adam, lr=1e-4, ReduceLROnPlateau, and EarlyStopping).

📈 Evaluation & TFLite Verification Results

The final model was evaluated on the 247 isolated, unaugmented TESTING/ images:

Metric

Result

Unseen Test Set Accuracy (Keras .h5)

90.69% (224 / 247)

Converted TFLite Model Accuracy (.tflite)

90.69% (224 / 247)

Keras vs. TFLite Prediction Match Rate

100.00% (247 / 247)

Input Tensor Shape & Dtype

[1, 224, 224, 3] (float32 in [0, 255])

🚀 Quickstart: Running the TFLite Model in Python

You can run inference using zootag_model.tflite with raw [0, 255] float32 images:

import numpy as np
import tensorflow as tf
from PIL import Image

CLASS_NAMES = [
    'AMPHIBIAN', 'BIRD', 'FISH', 'INSECT',
    'MAMMAL', 'MARINE_INVERTEBRATE', 'REPTILE'
]

# 1. Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="zootag_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. Load and resize an image to (224, 224, 3) with float32 values in [0, 255]
img = Image.open("sample_specimen.jpg").convert("RGB").resize((224, 224))
input_tensor = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

# 3. Run inference
interpreter.set_tensor(input_details[0]['index'], input_tensor)
interpreter.invoke()

probs = interpreter.get_tensor(output_details[0]['index'])[0]
pred_idx = np.argmax(probs)

print(f"Predicted Tag: {CLASS_NAMES[pred_idx]} ({probs[pred_idx] * 100:.2f}% confidence)")

