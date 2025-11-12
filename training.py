import cv2, os, time

dataset_path = "faces/"
os.makedirs(dataset_path, exist_ok=True)

print("Enter the name of the person to record:")
person_name = input().strip()
if person_name == "":
    person_name = "Person"

person_folder = os.path.join(dataset_path, person_name)
os.makedirs(person_folder, exist_ok=True)

# Try front camera first
cam = cv2.VideoCapture(1)
if not cam.isOpened():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("No camera available.")
        exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("Recording faces. Please move your face slowly in front of the camera...")
frame_count = 0
target_frames = 50  # number of frames to capture per person
capture_interval = 0.2  # seconds between captures
last_capture_time = time.time()

while frame_count < target_frames:
    ret, frame = cam.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    current_time = time.time()
    if current_time - last_capture_time >= capture_interval:
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            img_path = os.path.join(person_folder, f"{frame_count+1}.jpg")
            cv2.imwrite(img_path, face_img)
            frame_count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        last_capture_time = current_time

    cv2.imshow("Recording", frame)
    if cv2.waitKey(1) == ord('q'): break

cam.release()
cv2.destroyAllWindows()
print(f"Saved {frame_count} frames for {person_name}")
