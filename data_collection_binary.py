import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import traceback

# Initialize video capture
capture = cv2.VideoCapture(0)

# Initialize hand detector
hd = HandDetector(maxHands=1)

# Use hardcoded path to avoid issues
base_path = "C:/Users/hp/Documents/projects/Myprojects/Sign-Language-To-Text-and-Speech-Conversion-master"

print(f"Base path: {base_path}")

# Initialize directories with proper path handling
test_data_path = os.path.join(base_path, "test_data_2.0")

# Create main test_data directory if it doesn't exist
if not os.path.exists(test_data_path):
    os.makedirs(test_data_path)
    print(f"Created directory: {test_data_path}")

# Initialize count
count = 0
p_dir = "A"
c_dir = "a"

# Create initial directories for letter A
gray_imgs_path = os.path.join(test_data_path, "Gray_imgs", p_dir)
gray_draw_path = os.path.join(test_data_path, "Gray_imgs_with_drawing", p_dir)
binary_imgs_path = os.path.join(test_data_path, "Binary_imgs", p_dir)

# Create directories if they don't exist
os.makedirs(gray_imgs_path, exist_ok=True)
os.makedirs(gray_draw_path, exist_ok=True)
os.makedirs(binary_imgs_path, exist_ok=True)

# Get initial count
if os.path.exists(gray_imgs_path):
    count = len(os.listdir(gray_imgs_path))

offset = 30
step = 1
flag = False
suv = 0

# Create white image for skeleton drawing - FIXED VARIABLE NAME
white_img_path = os.path.join(base_path, "white.jpg")
white = np.ones((400, 400, 3), np.uint8) * 255
cv2.imwrite(white_img_path, white)

print("Data Collection Started!")
print("Press 'a' to start/stop capturing")
print("Press 'n' to switch to next letter")
print("Press 'ESC' to exit")

while True:
    try:
        _, frame = capture.read()
        if frame is None:
            continue
         
        #Preprocessing the frame   
        frame = cv2.flip(frame, 1)
        hands, img = hd.findHands(frame, draw=True, flipType=True)
        img_final = img_final1 = img_final2 = None
        
        #If a hand is detected
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            # Ensure coordinates are within frame bounds
            y_start = max(0, y - offset)
            y_end = min(frame.shape[0], y + h + offset)
            x_start = max(0, x - offset)
            x_end = min(frame.shape[1], x + w + offset)
            
            if y_end > y_start and x_end > x_start:
                image = frame[y_start:y_end, x_start:x_end]
                
                # Process ROI for different image types
                roi = image

                # For simple gray image without draw
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (1, 1), 2)

                # For binary image
                gray2 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blur2 = cv2.GaussianBlur(gray2, (5, 5), 2)
                th3 = cv2.adaptiveThreshold(blur2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                ret, test_image = cv2.threshold(th3, 27, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # Create standardized 400x400 images
                
                # 1. Gray image without drawing (img_final1)
                test_image1 = blur
                img_final1 = np.ones((400, 400), np.uint8) * 148
                if test_image1.size > 0:
                    h1 = test_image1.shape[0]
                    w1 = test_image1.shape[1]
                    if h1 > 0 and w1 > 0:
                        y1_start = (400 - h1) // 2
                        x1_start = (400 - w1) // 2
                        if y1_start + h1 <= 400 and x1_start + w1 <= 400:
                            img_final1[y1_start:y1_start + h1, x1_start:x1_start + w1] = test_image1

                # 2. Binary image (img_final)
                img_final = np.ones((400, 400), np.uint8) * 255
                if test_image.size > 0:
                    h2 = test_image.shape[0]
                    w2 = test_image.shape[1]
                    if h2 > 0 and w2 > 0:
                        y2_start = (400 - h2) // 2
                        x2_start = (400 - w2) // 2
                        if y2_start + h2 <= 400 and x2_start + w2 <= 400:
                            img_final[y2_start:y2_start + h2, x2_start:x2_start + w2] = test_image

        # Process hand for skeleton drawing
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            y_start = max(0, y - offset)
            y_end = min(frame.shape[0], y + h + offset)
            x_start = max(0, x - offset)
            x_end = min(frame.shape[1], x + w + offset)
            
            if y_end > y_start and x_end > x_start:
                image = frame[y_start:y_end, x_start:x_end]
                white = cv2.imread(white_img_path)  # FIXED: Using correct variable name
                
                handz, imgz = hd.findHands(image, draw=False, flipType=True)
                if handz:
                    hand_cropped = handz[0]
                    pts = hand_cropped['lmList']
                    
                    os = ((400 - w) // 2) - 15
                    os1 = ((400 - h) // 2) - 15
                    
                    white = np.ones((400, 400, 3), np.uint8) * 255
                    
                    connections = [
                        (0, 1), (1, 2), (2, 3), (3, 4),
                        (5, 6), (6, 7), (7, 8),
                        (9, 10), (10, 11), (11, 12),
                        (13, 14), (14, 15), (15, 16),
                        (17, 18), (18, 19), (19, 20),
                        (5, 9), (9, 13), (13, 17),
                        (0, 5), (0, 17)
                    ]
                    
                    for start, end in connections:
                        if start < len(pts) and end < len(pts):
                            start_point = (pts[start][0] + os, pts[start][1] + os1)
                            end_point = (pts[end][0] + os, pts[end][1] + os1)
                            
                            if (0 <= start_point[0] < 400 and 0 <= start_point[1] < 400 and
                                0 <= end_point[0] < 400 and 0 <= end_point[1] < 400):
                                cv2.line(white, start_point, end_point, (0, 255, 0), 3)

                    for i in range(21):
                        if i < len(pts):
                            point_x = pts[i][0] + os
                            point_y = pts[i][1] + os1
                            if 0 <= point_x < 400 and 0 <= point_y < 400:
                                cv2.circle(white, (point_x, point_y), 2, (0, 0, 255), 1)

                    cv2.imshow("skeleton", white)

                # 3. Gray image with drawings (img_final2)
                image1 = frame[y_start:y_end, x_start:x_end]
                roi1 = image1

                gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
                blur1 = cv2.GaussianBlur(gray1, (1, 1), 2)

                test_image2 = blur1
                img_final2 = np.ones((400, 400), np.uint8) * 148
                if test_image2.size > 0:
                    h3 = test_image2.shape[0]
                    w3 = test_image2.shape[1]
                    if h3 > 0 and w3 > 0:
                        y3_start = (400 - h3) // 2
                        x3_start = (400 - w3) // 2
                        if y3_start + h3 <= 400 and x3_start + w3 <= 400:
                            img_final2[y3_start:y3_start + h3, x3_start:x3_start + w3] = test_image2

                # Display processed images
                if img_final is not None:
                    cv2.imshow("binary", img_final)
                if img_final1 is not None:
                    cv2.imshow("gray w/o draw", img_final1)
                if img_final2 is not None:
                    cv2.imshow("gray with draw", img_final2)

        # Display main frame
        status_text = f"Directory: {p_dir}  Count: {count}  Capturing: {'ON' if flag else 'OFF'}"
        frame = cv2.putText(frame, status_text, (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Data Collection - Sign Language", frame)
        
        # Handle keyboard input
        interrupt = cv2.waitKey(1)
        if interrupt & 0xFF == 27:
            break
            
        if interrupt & 0xFF == ord('n'):
            p_dir = chr(ord(p_dir) + 1)
            c_dir = chr(ord(c_dir) + 1)
            if ord(p_dir) > ord('Z'):
                p_dir = "A"
                c_dir = "a"
            flag = False
            
            # Create new directory paths
            gray_imgs_path = os.path.join(test_data_path, "Gray_imgs", p_dir)
            os.makedirs(gray_imgs_path, exist_ok=True)
            
            # Reset count for new directory
            count = len(os.listdir(gray_imgs_path))
            print(f"Switched to directory: {p_dir}, current count: {count}")

        if interrupt & 0xFF == ord('a'):
            flag = not flag
            suv = 0
            step = 1
            print(f"Capture {'started' if flag else 'stopped'}")

        # Capture images when flag is True
        if flag and img_final is not None and img_final1 is not None and img_final2 is not None:
            if suv >= 50:
                flag = False
                print("Reached capture limit (50 images)")
                
            if step % 2 == 0:
                # Create directory paths
                gray_img_dir = os.path.join(test_data_path, "Gray_imgs", p_dir)
                gray_draw_dir = os.path.join(test_data_path, "Gray_imgs_with_drawing", p_dir)
                binary_img_dir = os.path.join(test_data_path, "Binary_imgs", p_dir)
                
                # Ensure directories exist
                os.makedirs(gray_img_dir, exist_ok=True)
                os.makedirs(gray_draw_dir, exist_ok=True)
                os.makedirs(binary_img_dir, exist_ok=True)
                
                # Create file paths
                gray_img_path = os.path.join(gray_img_dir, f"{c_dir}{count}.jpg")
                gray_draw_path = os.path.join(gray_draw_dir, f"{c_dir}{count}.jpg")
                binary_img_path = os.path.join(binary_img_dir, f"{c_dir}{count}.jpg")
                
                # Save all three images
                cv2.imwrite(gray_img_path, img_final1)
                cv2.imwrite(gray_draw_path, img_final2)
                cv2.imwrite(binary_img_path, img_final)
                
                print(f"Saved: {gray_img_path}, {gray_draw_path}, and {binary_img_path}")
                count += 1
                suv += 1
            step += 1
            
    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())

# Cleanup
print("Data collection completed!")
capture.release()
cv2.destroyAllWindows()