import cv2
import numpy as np
from ultralytics import YOLO
from pressure import PressureMap
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VIDEO_PATH = "videomarocnig.mp4"
MODEL_PATH = "best.pt"
MATRIX_PATH = "h_matrix.npy"
OUTPUT_DETECTION = "output_detection.mp4"
OUTPUT_RADAR = "output_radar.mp4"

CONF_THRESHOLD = 0.25
IMG_SIZE = 1280

SCALE = 8
PITCH_W = 105 * SCALE
PITCH_H = 68 * SCALE


def load_homography():
    try:
        return np.load(MATRIX_PATH)
    except:
        print("Error: h_matrix.npy not found")
        exit()


def project_point(H, x, y):
    """Project image coordinates to pitch coordinates"""
    point_vec = np.array([[[x, y]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(point_vec, H)[0][0]
    return dst[0], dst[1]


def draw_pitch_background():
    """Draw tactical pitch background"""
    img = np.zeros((PITCH_H, PITCH_W, 3), dtype=np.uint8)
    img[:] = (34, 139, 34)
    mid_x = int(52.5 * SCALE)
    cv2.line(img, (mid_x, 0), (mid_x, PITCH_H), (255, 255, 255), 2)
    cv2.circle(img, (mid_x, int(34 * SCALE)), int(9.15 * SCALE), (255, 255, 255), 2)
    return img


def main():
    print(f"Device: {DEVICE}")
    H = load_homography()
    pm = PressureMap(pitch_width=105, pitch_height=68, resolution=1.0)
    
    try:
        model = YOLO(MODEL_PATH)
        model.to(DEVICE)
    except Exception as e:
        print(f"Error loading {MODEL_PATH}")
        print(e)
        return

    print(f"Classes: {model.names}")
    
    id_maroc = None
    id_nigeria = None
    id_ball = None
    id_ref = None

    for id_class, name in model.names.items():
        name_lower = name.lower()
        if "maroc" in name_lower:
            id_maroc = id_class
        elif "nigeria" in name_lower:
            id_nigeria = id_class
        elif "ball" in name_lower:
            id_ball = id_class
        elif "ref" in name_lower:
            id_ref = id_class

    print(f"Mapping: Maroc={id_maroc}, Nigeria={id_nigeria}, Ball={id_ball}, Ref={id_ref}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_detection = cv2.VideoWriter(OUTPUT_DETECTION, fourcc, fps, (900, 500))
    out_radar = cv2.VideoWriter(OUTPUT_RADAR, fourcc, fps, (PITCH_W, PITCH_H))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        radar_img = draw_pitch_background()
        results = model.track(frame, persist=True, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)
        current_frame_players = []

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, cls in zip(boxes, cls_ids):
                x1, y1, x2, y2 = box
                feet_x, feet_y = (x1 + x2) / 2, y2
                real_x, real_y = project_point(H, feet_x, feet_y)
                rx, ry = int(real_x * SCALE), int(real_y * SCALE)
                on_pitch = (0 <= rx < PITCH_W and 0 <= ry < PITCH_H)

                if cls == id_maroc:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    if on_pitch:
                        cv2.circle(radar_img, (rx, ry), 8, (0, 0, 255), -1)
                        current_frame_players.append((real_x, real_y, 0))

                elif cls == id_nigeria:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    if on_pitch:
                        cv2.circle(radar_img, (rx, ry), 8, (255, 255, 0), -1)

                elif cls == id_ball:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    if on_pitch:
                        cv2.circle(radar_img, (rx, ry), 6, (0, 255, 255), -1)

                elif cls == id_ref:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

        if len(current_frame_players) > 0:
            pressure_grid = pm.generate_heatmap(current_frame_players, team_of_interest_id=0)
            heatmap_img = pm.visualize(pressure_grid, PITCH_W, PITCH_H)
            radar_img = cv2.addWeighted(radar_img, 0.6, heatmap_img, 0.4, 0)

        frame_resized = cv2.resize(frame, (900, 500))
        cv2.imshow("Detection", frame_resized)
        cv2.imshow("Radar", radar_img)
        
        out_detection.write(frame_resized)
        out_radar.write(radar_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_detection.release()
    out_radar.release()
    cv2.destroyAllWindows()
    print(f"Videos saved: {OUTPUT_DETECTION}, {OUTPUT_RADAR}")


if __name__ == "__main__":
    main()