import cv2
import numpy as np

VIDEO_PATH = "videomarocnig.mp4"
OUTPUT_MATRIX = "h_matrix.npy"
CALIBRATION_FRAME = 550

image_points = []


def click_event(event, x, y, flags, params):
    """Record mouse clicks on image"""
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Point captured: ({x}, {y})")
        image_points.append([x, y])
        cv2.circle(params['img'], (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Calibration", params['img'])


def main():
    """Compute homography matrix from 4 point correspondences"""
    
    # Load frame from video
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, CALIBRATION_FRAME)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error reading video")
        return

    print("\n=== CAMERA CALIBRATION ===")
    print("1. Click 4 key points on the pitch (line intersections)")
    print("2. Space them as far apart as possible")
    print("3. Press any key when done\n")
    
    cv2.imshow("Calibration", frame)
    cv2.setMouseCallback("Calibration", click_event, {'img': frame})
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(image_points) < 4:
        print("Error: You must click at least 4 points!")
        return

    # Get real world coordinates
    world_points = []
    print("\n=== REAL-WORLD COORDINATES (meters) ===")
    print("Origin (0,0) = Top-Left of pitch")
    print("Pitch dimensions: ~105m x 68m")
    print("Center line X = 52.5m\n")
    
    for i, pt in enumerate(image_points):
        print(f"Point {i+1} (pixels: {pt}):")
        try:
            wx = float(input("  X (meters): "))
            wy = float(input("  Y (meters): "))
            world_points.append([wx, wy])
        except ValueError:
            print("Invalid input. Using 0,0")
            world_points.append([0, 0])

    # Compute homography
    src_pts = np.float32(image_points)
    dst_pts = np.float32(world_points)
    H, status = cv2.findHomography(src_pts, dst_pts)

    print("\n=== RESULT ===")
    print("Homography matrix:")
    print(H)
    
    np.save(OUTPUT_MATRIX, H)
    print(f"\nMatrix saved to '{OUTPUT_MATRIX}'")
    print("Ready for next step!")


if __name__ == "__main__":
    main()