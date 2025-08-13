
from flask import Flask, render_template, Response, request, redirect
import cv2
import os
import face_recognition
import numpy as np
from datetime import datetime

app = Flask(__name__)

# CAMERA_URL = "rtsp://Admin_PHT:kb~O,]c]N@%(JU)hxKeBK@10.81.14.119:554/stream1"
CAMERA_URL = "http://192.168.1.3:8080/video"
# CAMERA_URL = 0
cap = cv2.VideoCapture(CAMERA_URL)

known_face_encodings = []
known_face_names = []
unknown_encodings = []
unknown_id = 1

def load_existing_unknown_id():
    global unknown_id
    dataset_path = "dataset"
    os.makedirs(dataset_path, exist_ok=True)
    for name in os.listdir(dataset_path):
        if name.startswith("unknown_"):
            try:
                idx = int(name.split("_")[1])
                if idx >= unknown_id:
                    unknown_id = idx + 1
            except:
                pass

def load_dataset():
    base_dir = "dataset"
    for person_name in os.listdir(base_dir):
        person_path = os.path.join(base_dir, person_name)
        if not os.path.isdir(person_path):
            continue
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model="hog")
            encodings = face_recognition.face_encodings(rgb, boxes)
            for encoding in encodings:
                known_face_encodings.append(encoding)
                known_face_names.append(person_name)

def get_latest_frame():
    for _ in range(5):
        cap.grab()
    return cap.read()

def gen_frames():
    global unknown_id
    face_save_path = "captured_faces"
    os.makedirs(face_save_path, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    saved_unknown_encodings = []

    while True:
        success, frame = get_latest_frame()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(distances) > 0:
                min_distance = min(distances)
                best_match_index = distances.tolist().index(min_distance)
                if min_distance < 0.45:
                    name = known_face_names[best_match_index]

            log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("logs/detections.log", "a") as log_file:
                log_file.write(f"[{log_time}] Nhận diện: {name}\n")

            if name == "Unknown":
                is_new = True
                for enc in unknown_encodings + saved_unknown_encodings:
                    if np.linalg.norm(enc - face_encoding) < 0.5:
                        is_new = False
                        break
                if is_new:
                    folder_name = f"dataset/unknown_{unknown_id}"
                    os.makedirs(folder_name, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    face_crop = frame[top:bottom, left:right]
                    save_path = os.path.join(folder_name, f"{timestamp}.jpg")
                    cv2.imwrite(save_path, face_crop)
                    print(f"📸 Đã lưu khuôn mặt Unknown mới: {save_path}")

                    # Lưu thêm vào captured_faces/
                    day_folder = datetime.now().strftime("%Y-%m-%d")
                    captured_dir = os.path.join("captured_faces", day_folder)
                    os.makedirs(captured_dir, exist_ok=True)
                    captured_path = os.path.join(captured_dir, f"{timestamp}_unknown_{unknown_id}.jpg")
                    cv2.imwrite(captured_path, face_crop)
                    print(f"📸 Cũng đã lưu vào: {captured_path}")

                    unknown_encodings.append(face_encoding)
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(f"unknown_{unknown_id}")
                    unknown_id += 1

            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add_face', methods=['POST'])
def add_face():
    name = request.form['name']
    person_dir = os.path.join("dataset", name)
    os.makedirs(person_dir, exist_ok=True)

    count = 0
    while count < 5:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)

        if len(boxes) == 0:
            print("❌ Không thấy khuôn mặt, bỏ qua ảnh.")
            continue

        filepath = os.path.join(person_dir, f"{name}_{count}.jpg")
        cv2.imwrite(filepath, frame)
        print(f"✅ Đã lưu ảnh: {filepath}")
        count += 1

        encodings = face_recognition.face_encodings(rgb, boxes)
        for encoding in encodings:
            known_face_encodings.append(encoding)
            known_face_names.append(name)

    return redirect("/")

if __name__ == '__main__':
    load_existing_unknown_id()
    load_dataset()
    app.run(host='0.0.0.0', port=5000)
