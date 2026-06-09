# 🌿 AI Plant Disease Diagnostics Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-FF4B4B.svg)](https://streamlit.io/)

An industry-grade, production-ready Computer Vision application that uses Deep Learning and Transfer Learning to classify plant leaf diseases. 

Instead of wrapping heavy neural networks directly inside the user interface, this project utilizes a modern **Decoupled Microservice Architecture**: a high-performance **FastAPI REST API** handles heavy model inference on the backend, while a lightweight **Streamlit dashboard** serves as the user-facing frontend.

---

## 🏗️ System Architecture

The application is split into distinct layers to ensure high scalability and separation of concerns:

1. **Deep Learning Pipeline (`src/train.py`):** Uses a pre-trained **ResNet18** architecture via Transfer Learning. It features real-time data augmentation and advanced checkpoint tracking to save only the highest-performing weights based on validation accuracy.
2. **Inference Microservice (`src/api.py`):** A production-level **FastAPI** backend that exposes a `/predict` POST endpoint. It loads model tensors into memory and processes multi-part image file data stream uploads.
3. **Frontend Presentation (`app.py`):** A clean **Streamlit** dashboard that communicates with the API backend over asynchronous HTTP requests, keeping frontend memory usage extremely low.

---

## 🛠️ Tech Stack & Dependencies

- **Core Framework:** PyTorch, Torchvision
- **Backend API:** FastAPI, Uvicorn
- **Frontend UI:** Streamlit, Pillow (PIL)
- **Data Engineering:** NumPy, Pandas

---

## 📂 Project Repository Structure

```text
Plant_Disease_Detector/
│
├── data/                   # Dataset directory containing 'train' and 'val' split folders
│
├── src/                    # System Core Source Code
│   ├── train.py            # Advanced PyTorch model training and validation pipeline
│   └── api.py              # FastAPI REST API backend inference engine
│
├── app.py                  # Streamlit interactive user dashboard interface
├── plant_model.pth         # Saved state dictionary checkpoints (Generated after training)
└── requirements.txt        # Comprehensive package dependency manifest