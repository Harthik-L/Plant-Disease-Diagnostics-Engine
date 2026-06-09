# 🌿 AI Plant Disease Diagnostics Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://plant-disease-diagnostics-engine-g3nxepxsv8bmaocot3z3el.streamlit.app/)

An industry-grade, production-ready Computer Vision application that uses Deep Learning and Transfer Learning to classify plant leaf diseases with stellar accuracy.

> 🚀 **Live Application:** [Access the Live Diagnostic Dashboard Here](https://plant-disease-diagnostics-engine-g3nxepxsv8bmaocot3z3el.streamlit.app/)

---

## 🚀 About the Project

This engine is capable of instantly detecting and classifying leaf conditions (such as **Healthy**, **Early Blight**, and **Late Blight**) from high-resolution image uploads. 

Unlike traditional monolithic applications that wrap heavy machine learning frameworks directly inside the user interface, this project is engineered using a modern **Decoupled Microservice Architecture** to optimize computing efficiency, resource footprints, and deployment flexibility.

### 🏗️ Architectural Breakdown

1. **Deep Learning Pipeline (`src/train.py`):** Built using **PyTorch**, leveraging a **ResNet18 transfer learning backbone** fine-tuned with customized data augmentation matrices. The optimized training routine reached a stellar **94.89% validation accuracy** over 5 epochs.
2. **Inference Backend Microservice (`src/api.py`):** Powered by **FastAPI** and packaged within a custom **Docker container** hosted on **Hugging Face Spaces**. It exposes a secure asynchronous `/predict` REST endpoint, loading model tensors into memory to process image file data stream uploads entirely in the cloud.
3. **Frontend Presentation Dashboard (`app.py`):** An intuitive, interactive user interface engineered with **Streamlit** and deployed on **Streamlit Community Cloud**. It safely handles image payload streams over secure HTTP pipelines without requiring local machine learning dependencies or heavy library packages.

---

## 🛠️ Tech Stack & Architecture Highlights

* **Core Frameworks:** PyTorch, FastAPI, Streamlit, Uvicorn
* **Data Processing & CV:** Torchvision, Pillow (PIL), NumPy, Pandas
* **Infrastructure & DevOps:** Docker Containers, Git/GitHub Version Control, Hugging Face Spaces, Streamlit Cloud Pipelines
* **Architecture Pattern:** Split Frontend/Backend Microservices (REST API over HTTP POST)

---

## 📂 Project Repository Structure

```text
Plant_Disease_Detector/
│
├── data/                  # Dataset directory containing 'train' and 'val' split folders (Git ignored)
│
├── src/                    # System Core Source Code
│   ├── train.py            # Advanced PyTorch model training and validation pipeline
│   └── api.py              # FastAPI REST API backend inference engine
│
├── app.py                  # Streamlit interactive user dashboard interface
├── plant_model.pth         # Saved state dictionary checkpoints (Generated after training)
├── Dockerfile              # Container deployment instructions for the Hugging Face API
└── requirements.txt        # Comprehensive package dependency manifest