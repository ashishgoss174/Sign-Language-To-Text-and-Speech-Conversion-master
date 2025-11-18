import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os as oss
import traceback

capture = cv2.VideoCapture(0)
hd = HandDetector(maxHands=1)

# You don't need two hand detectors
# hd2 = HandDetector(maxHands=1)  # Remove this

count = len(oss.listdir("C:\\Users\\hp\\Documents\\projects\\Myprojects\\Sign-Language-To-Text-and-Speech-Conversion-master\\AtoZ_3.1\\A"))
c_dir = 'A'

offset = 15
step = 1
flag = False
suv = 0

# Create white image
white = np.ones((400, 400, 3), np.uint8) * 255  # Fixed: Added 3 channels for color image
cv2.imwrite("C:\\Users\\hp\\Documents\\projects\\Myprojects\\Sign-Language-To-Text-and-Speech-Conversion-master\\white.jpg", white)

while True:
    try:
        _, frame = capture.read()
        if frame is None:
            continue
            
        frame = cv2.flip(frame, 1)
        hands, img = hd.findHands(frame, draw=True, flipType=True)  # Set draw=True to see hand landmarks
        white = cv2.imread("C:\\Users\\hp\\Documents\\projects\\Myprojects\\Sign-Language-To-Text-and-Speech-Conversion-master\\white.jpg")

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            # Reset white image for each frame
            white = np.ones((400, 400, 3), np.uint8) * 255
            
            # Get landmark points
            pts = hand['lmList']
            
            # Calculate offsets to center the hand in the white image
            os = (400 - w) // 2 - 15
            os1 = (400 - h) // 2 - 15
            
            # Draw hand connections (skeleton)
            # Palm connections
            connections = [
                # Thumb
                [0, 1], [1, 2], [2, 3], [3, 4],
                # Index finger
                [0, 5], [5, 6], [6, 7], [7, 8],
                # Middle finger
                [0, 9], [9, 10], [10, 11], [11, 12],
                # Ring finger
                [0, 13], [13, 14], [14, 15], [15, 16],
                # Pinky finger
                [0, 17], [17, 18], [18, 19], [19, 20],
                # Palm connections
                [5, 9], [9, 13], [13, 17]
            ]
            
            # Draw all connections
            for connection in connections:
                start_idx, end_idx = connection
                if 0 <= start_idx < len(pts) and 0 <= end_idx < len(pts):
                    start_point = (pts[start_idx][0] - x + os + offset, pts[start_idx][1] - y + os1 + offset)
                    end_point = (pts[end_idx][0] - x + os + offset, pts[end_idx][1] - y + os1 + offset)
                    
                    # Check if points are within image bounds
                    if (0 <= start_point[0] < 400 and 0 <= start_point[1] < 400 and
                        0 <= end_point[0] < 400 and 0 <= end_point[1] < 400):
                        cv2.line(white, start_point, end_point, (0, 255, 0), 2)
            
            # Draw landmarks
            for i, point in enumerate(pts):
                point_x = point[0] - x + os + offset
                point_y = point[1] - y + os1 + offset
                
                if 0 <= point_x < 400 and 0 <= point_y < 400:
                    cv2.circle(white, (point_x, point_y), 3, (0, 0, 255), -1)
            
            skeleton1 = white.copy()
            cv2.imshow("Hand Skeleton", skeleton1)

        # Display information on frame
        frame = cv2.putText(frame, f"dir={c_dir}  count={count}", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Frame", frame)
        
        interrupt = cv2.waitKey(1)
        if interrupt & 0xFF == 27:  # ESC key
            break
            
        if interrupt & 0xFF == ord('n'):  # Next directory
            c_dir = chr(ord(c_dir) + 1)
            if ord(c_dir) > ord('Z'):
                c_dir = 'A'
            flag = False
            
            # Create directory if it doesn't exist
            dir_path = f"C:\\Users\\hp\\Documents\\projects\\Myprojects\\Sign-Language-To-Text-and-Speech-Conversion-master\\AtoZ_3.1\\{c_dir}"
            if not oss.path.exists(dir_path):
                oss.makedirs(dir_path)
            count = len(oss.listdir(dir_path))
            print(f"Switched to directory: {c_dir}, current count: {count}")

        if interrupt & 0xFF == ord('a'):  # Start/stop capturing
            flag = not flag
            suv = 0
            step = 1
            print(f"Capture {'started' if flag else 'stopped'}")

        # Capture images when flag is True
        if flag and hands:
            if suv >= 180:  # Limit number of captures
                flag = False
                print("Reached capture limit (180 images)")
                
            if step % 3 == 0:  # Capture every 3rd frame
                # Ensure directory exists
                dir_path = f"C:\\Users\\hp\\Documents\\projects\\Myprojects\\Sign-Language-To-Text-and-Speech-Conversion-master\\AtoZ_3.1\\{c_dir}"
                if not oss.path.exists(dir_path):
                    oss.makedirs(dir_path)
                
                # Save the skeleton image
                filename = oss.path.join(dir_path, f"{count}.jpg")
                cv2.imwrite(filename, skeleton1)
                print(f"Saved: {filename}")
                
                count += 1
                suv += 1
            step += 1

    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())

capture.release()
cv2.destroyAllWindows()