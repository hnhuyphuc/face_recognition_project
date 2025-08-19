import os
import cv2
import face_recognition
import pickle

def train_and_save():
    image_paths = []
    labels = []

    base_dir = "dataset"
    for person_name in os.listdir(base_dir):
        person_path = os.path.join(base_dir, person_name)
        if not os.path.isdir(person_path):
            continue
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            image_paths.append(img_path)
            labels.append(person_name)

    known_encodings = []
    known_names = []

    for (img_path, name) in zip(image_paths, labels):
        image = cv2.imread(img_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="cnn")
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(name)

    data = {"encodings": known_encodings, "names": known_names}
    with open("encodings/encodings.pickle", "wb") as f:
        pickle.dump(data, f)
