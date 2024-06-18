import cv2
import os
import shutil
import sys
import time

def main():
    try:
        source_dir = sys.argv[1]
        target_dir = sys.argv[2]
    except IndexError:
        print("Usage: python script.py <source_dir> <target_dir>")
        sys.exit(1)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    subdirectories = [os.path.join(source_dir, d) for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    kept_count = 0

    for subdirectory in subdirectories:
        video_path = os.path.join(subdirectory, 'combined.mp4')
        if os.path.isfile(video_path):
            key_pressed = False
            while not key_pressed:
                cap = cv2.VideoCapture(video_path)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Play at 2x speed
                    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 1)
                    cv2.imshow('Video', frame)
                    # Wait 15ms and check for user input which is approximately a 2x speed
                    key = cv2.waitKey(15)
                    if key == 32:  # Space key
                        key_pressed = True
                        break
                    if key == 13:  # Enter key
                        key_pressed = True
                        shutil.copytree(subdirectory, os.path.join(target_dir, os.path.basename(subdirectory)))
                        kept_count += 1
                        break
                if not key_pressed:
                    time.sleep(2)

            cap.release()
            cv2.destroyAllWindows()

    print(f"Subdirectories kept: {kept_count}")

if __name__ == "__main__":
    main()