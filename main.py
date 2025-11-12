import cv2, os, numpy as np

# ----------------------------
# 1️⃣ Load Dataset
# ----------------------------
dataset_path = "faces/"
images, labels, label_names, label_id = [], [], {}, 0

# Iterate over each person folder in 'faces/'
for person_name in os.listdir(dataset_path):
    folder = os.path.join(dataset_path, person_name)
    if os.path.isdir(folder):
        # Assign a numeric label to each person
        label_names[label_id] = person_name
        # Load all images for this person
        for img_name in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, img_name), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(label_id)
        label_id += 1

# Exit if no faces found
if len(images) == 0:
    print("No faces found. Please run the training script first.")
    exit()

# Convert labels to numpy array (required by OpenCV LBPH)
labels = np.array(labels, dtype=np.int32)

# ----------------------------
# 2️⃣ Train LBPH Recognizer
# ----------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(images, labels)  # Train once at startup

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ----------------------------
# 3️⃣ Open Camera (front preferred)
# ----------------------------
cam = cv2.VideoCapture(1)  # Try front camera first
if not cam.isOpened():
    cam = cv2.VideoCapture(0)  # Fallback to default camera
    if not cam.isOpened():
        print("No camera available.")
        exit()

# ----------------------------
# 4️⃣ Main Loop: Face Recognition
# ----------------------------
while True:
    ret, frame = cam.read()  # Capture frame from camera
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces

    # Iterate through all detected faces
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]  # Crop the face
        label, confidence = recognizer.predict(face_img)  # Predict label using LBPH
        # If confidence < 70 → recognized, else → Unknown
        name = label_names[label] if confidence < 70 else "Unknown"

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        # Display name above rectangle
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # Display the camera feed with annotations
    cv2.imshow("Face Recognition", frame)
    # Exit loop if user presses 'q'
    if cv2.waitKey(1) == ord('q'): break

# ----------------------------
# 5️⃣ Cleanup
# ----------------------------
cam.release()             # Release camera
cv2.destroyAllWindows()   # Close OpenCV windows
