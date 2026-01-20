# ⚽ AI Football Tactical Analysis
<img width="1651" height="753" alt="image" src="https://github.com/user-attachments/assets/1e14018c-0801-4109-a065-774af9f7e88a" />

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-green)
![OpenCV](https://img.shields.io/badge/Computer_Vision-OpenCV-red)
![Data Engineering](https://img.shields.io/badge/Role-Data%20Engineer-orange)

Ce projet de **Computer Vision** analyse des matchs de football en temps réel. Il utilise l'intelligence artificielle pour détecter les joueurs, l'arbitre et le ballon, puis projette leurs positions sur une carte tactique 2D (Radar) grâce à une transformation homographique pour générer des **Heatmaps** de pression.

---

## 👨‍💻 Auteur

**EL OUALI Abderrahman** *Data Engineer*

---

## 🚀 Fonctionnalités Clés

- **Détection d'objets (YOLOv8)** : Identification précise des joueurs (par équipe), arbitres et du ballon.
- **Transformation de Perspective (Homographie)** : Conversion des coordonnées vidéo (3D) vers un plan tactique (2D) à l'aide de matrices OpenCV.
- **Analyse Spatiale (Data Engineering)** :
  - Tracking des positions en temps réel.
  - Calcul de la "Pression" exercée par une équipe via des gaussiennes.
- **Visualisation** : Génération automatique de deux flux vidéo (Détection + Radar Tactique).

## 📂 Structure du Projet

```text
├── best.pt             # Modèle IA entraîné (YOLOv8 custom)
├── calibration.py      # Script de calibration de la matrice d'homographie
├── h_matrix.npy        # Matrice de transformation sauvegardée (NumPy)
├── main.py             # Pipeline ETL et Traitement Vidéo principal
├── pressure.py         # Algorithme de calcul des Heatmaps
└── README.md           # Documentation
 
