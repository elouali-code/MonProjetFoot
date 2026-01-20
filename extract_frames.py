import cv2
import numpy as np
from ultralytics import YOLO
from pressure import PressureMap
import torch

# --- CONFIGURATION ---
VIDEO_PATH = "videomarocnig.mp4"
MODEL_PATH = "best.pt"         # TON FICHIER ENTRAINÉ SUR COLAB
MATRIX_PATH = "h_matrix.npy"

# --- PARAMÈTRES ---
CONF_THRESHOLD = 0.25          # Confiance modérée (le modèle est sur-entraîné donc confiant)
IMG_SIZE = 1280                # Doit correspondre à l'entraînement
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Echelle Radar
SCALE = 8 
PITCH_W = 105 * SCALE
PITCH_H = 68 * SCALE

def load_homography():
    try:
        return np.load(MATRIX_PATH)
    except:
        print("ERREUR: h_matrix.npy manquant.")
        exit()

def project_point(H, x, y):
    point_vec = np.array([[[x, y]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(point_vec, H)[0][0]
    return dst[0], dst[1]

def draw_pitch_background():
    img = np.zeros((PITCH_H, PITCH_W, 3), dtype=np.uint8)
    img[:] = (34, 139, 34) # Vert pelouse
    mid_x = int(52.5 * SCALE)
    cv2.line(img, (mid_x, 0), (mid_x, PITCH_H), (255, 255, 255), 2)
    cv2.circle(img, (mid_x, int(34 * SCALE)), int(9.15 * SCALE), (255, 255, 255), 2)
    return img

def main():
    print(f"Démarrage sur {DEVICE}...")
    H = load_homography()
    pm = PressureMap(pitch_width=105, pitch_height=68, resolution=1.0)
    
    # 1. Chargement du modèle custom
    try:
        model = YOLO(MODEL_PATH)
        model.to(DEVICE)
    except Exception as e:
        print(f"ERREUR : Impossible de charger {MODEL_PATH}. Vérifie le chemin.")
        print(e)
        return

    # 2. MAPPING AUTOMATIQUE DES CLASSES
    # On ne devine pas les IDs, on demande au modèle.
    # Roboflow peut avoir nommé les classes "Maroc", "maroc", "players-maroc"... on cherche par mot-clé.
    print(f"Classes trouvées dans le modèle : {model.names}")
    
    id_maroc = None
    id_nigeria = None
    id_ball = None
    id_ref = None

    # Recherche intelligente des IDs
    for id_class, name in model.names.items():
        name_lower = name.lower()
        if "maroc" in name_lower: id_maroc = id_class
        elif "nigeria" in name_lower: id_nigeria = id_class
        elif "ball" in name_lower: id_ball = id_class
        elif "ref" in name_lower: id_ref = id_class

    print(f"Mapping ID : Maroc={id_maroc}, Nigeria={id_nigeria}, Ball={id_ball}, Ref={id_ref}")

    # Si une classe n'est pas trouvée, on alerte
    if id_maroc is None: print("ATTENTION: Classe 'Maroc' non trouvée. Vérifie tes noms sur Roboflow.")

    cap = cv2.VideoCapture(VIDEO_PATH)

    while True:
        ret, frame = cap.read()
        if not ret: break

        radar_img = draw_pitch_background()

        # 3. INFERENCE (Detection)
        # On utilise juste le modèle. Plus besoin de filtre couleur.
        results = model.track(frame, persist=True, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)
        
        current_frame_marocains = [] 

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, cls in zip(boxes, cls_ids):
                x1, y1, x2, y2 = box
                
                # Projection Terrain (On prend le milieu bas de la boîte)
                feet_x, feet_y = (x1 + x2) / 2, y2
                real_x, real_y = project_point(H, feet_x, feet_y)
                rx, ry = int(real_x * SCALE), int(real_y * SCALE)
                
                # Vérification que c'est sur le terrain
                on_pitch = (0 <= rx < PITCH_W and 0 <= ry < PITCH_H)

                # --- LOGIQUE D'AFFICHAGE SELON LA CLASSE ---
                
                # CAS 1 : MAROC (Rouge)
                if cls == id_maroc:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Rouge
                    if on_pitch:
                        cv2.circle(radar_img, (rx, ry), 8, (0, 0, 255), -1)
                        # On l'ajoute à la liste pour la pression (Team ID 0 arbitraire pour la heatmap)
                        current_frame_marocains.append((real_x, real_y, 0))

                # CAS 2 : NIGERIA (Blanc/Vert)
                elif cls == id_nigeria:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2) # Cyan
                    if on_pitch:
                        cv2.circle(radar_img, (rx, ry), 8, (255, 255, 0), -1)

                # CAS 3 : BALLON (Jaune)
                elif cls == id_ball:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2) # Jaune
                    if on_pitch:
                        cv2.circle(radar_img, (rx, ry), 6, (0, 255, 255), -1)

                # CAS 4 : ARBITRE (Gris)
                elif cls == id_ref:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
                    # On ne l'affiche pas sur le radar pour ne pas polluer

        # 4. CALCUL DE PRESSION (Seulement Maroc)
        if len(current_frame_marocains) > 0:
            # On génère la heatmap pour l'ID 0 (qu'on a donné aux Marocains juste au-dessus)
            pressure_grid = pm.generate_heatmap(current_frame_marocains, team_of_interest_id=0)
            heatmap_img = pm.visualize(pressure_grid, PITCH_W, PITCH_H)
            
            # Superposition
            radar_img = cv2.addWeighted(radar_img, 0.6, heatmap_img, 0.4, 0)

        # 5. AFFICHAGE
        frame_resized = cv2.resize(frame, (900, 500))
        cv2.imshow("Detection Custom", frame_resized)
        cv2.imshow("Radar Tactique", radar_img)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()