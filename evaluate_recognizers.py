import os
import time
import pickle
import face_recognition
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2
import numpy as np

# --- Cấu hình thư mục để train và test---
TRAIN_DIR = "face_evaluation/train"
TEST_DIR = "face_evaluation/test"
# Đường dẫn để lưu mô hình HOG + SVM
HOG_SVM_MODEL_PATH = "hog_svm_model.pkl"

# --- 1. Huấn luyện mô hình HOG + SVM ---

def train_hog_svm():
    print("Bắt đầu huấn luyện mô hình HOG + SVM...")
    # Chứa các đặc trưng HOG và nhãn tương ứng
    hog_features = []
    labels = []

    for person_name in os.listdir(TRAIN_DIR):
        person_dir = os.path.join(TRAIN_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        for image_file in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_file)
            image = cv2.imread(image_path)

             # Kiểm tra xem ảnh có được đọc thành công không
            if image is None:
                print(f"  [Cảnh báo] Không thể đọc được file ảnh, bỏ qua: {image_path}")
                continue # Bỏ qua file này và đi đến file tiếp theo

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Sử dụng face_recognition để tìm khuôn mặt trước
            face_locations = face_recognition.face_locations(image, model='hog')
            
            if len(face_locations) == 1:
                top, right, bottom, left = face_locations[0]
                face = gray_image[top:bottom, left:right]
                face_resized = cv2.resize(face, (64, 128))
                
                # Trích xuất đặc trưng HOG
                features = hog(face_resized, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")
                
                hog_features.append(features)
                labels.append(person_name)

    # Huấn luyện bộ phân loại SVM
    svm_classifier = SVC(kernel='linear', probability=True)
    svm_classifier.fit(hog_features, labels)
    
    # Lưu model
    with open(HOG_SVM_MODEL_PATH, 'wb') as f:
        pickle.dump(svm_classifier, f)
        
    print(f"Đã huấn luyện và lưu mô hình HOG + SVM vào '{HOG_SVM_MODEL_PATH}'")
    return svm_classifier

# --- 2. "Huấn luyện" (Tạo CSDL) cho mô hình CNN ---

def create_cnn_database():
    print("Tạo cơ sở dữ liệu khuôn mặt cho CNN...")
    # Tạo danh sách chứa các mã hóa khuôn mặt và tên tương ứng
    known_encodings = []
    known_names = []

    for person_name in os.listdir(TRAIN_DIR):
        person_dir = os.path.join(TRAIN_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        for image_file in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_file)
            image = face_recognition.load_image_file(image_path)
            
            # Tạo vector 128 chiều cho khuôn mặt
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(person_name)
                
    print("Đã tạo xong cơ sở dữ liệu CNN.")
    return known_encodings, known_names

# --- 3. Đánh giá mô hình trên tập Test ---

def evaluate_models(hog_svm_model, cnn_known_encodings, cnn_known_names):
    print("\nBắt đầu đánh giá trên tập test...")
    y_true = [] #Danh sách nhãn đúng
    y_pred_hog_svm = []
    y_pred_cnn = []
    
    hog_svm_time = 0
    cnn_time = 0
    
    # Đếm số ảnh test để tính thời gian trung bình
    total_test_images = 0

    for person_name in os.listdir(TEST_DIR):
        person_dir = os.path.join(TEST_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        for image_file in os.listdir(person_dir):
            total_test_images += 1 # Đếm tổng số ảnh
            image_path = os.path.join(person_dir, image_file)
            image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            face_locations = face_recognition.face_locations(rgb_image, model='hog')
            
            if len(face_locations) == 1:
                y_true.append(person_name)
                top, right, bottom, left = face_locations[0]

                # Đánh giá HOG + SVM
                start = time.time()
                face = gray_image[top:bottom, left:right]
                face_resized = cv2.resize(face, (64, 128))
                features = hog(face_resized, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")
                prediction = hog_svm_model.predict([features])[0]
                y_pred_hog_svm.append(prediction)
                hog_svm_time += time.time() - start

                # Đánh giá CNN
                start = time.time()
                unknown_encoding = face_recognition.face_encodings(rgb_image, [face_locations[0]])[0]
                matches = face_recognition.compare_faces(cnn_known_encodings, unknown_encoding)
                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = cnn_known_names[first_match_index]
                y_pred_cnn.append(name)
                cnn_time += time.time() - start

    # --- 4. So sánh hai mô hình theo một số chỉ tiêu---
    print("\n" + "="*70)
    print("           BÁO CÁO SO SÁNH HIỆU NĂNG NHẬN DẠNG KHUÔN MẶT")
    print("="*70)
    
    # --- Báo cáo cho HOG + SVM ---
    print("\n--- MÔ HÌNH HOG + SVM ---")
    print(f"Tổng thời gian dự đoán: {hog_svm_time:.4f} giây")
    print(f"Tổng số lượng ảnh dự đoán: {total_test_images: } ảnh")

    if total_test_images > 0:
        print(f"Thời gian trung bình/ảnh: {(hog_svm_time / total_test_images):.4f} giây")
    
    accuracy_hog_svm = accuracy_score(y_true, y_pred_hog_svm)
    print(f"Accuracy (Độ chính xác tổng thể): {accuracy_hog_svm:.4f}")
    print("Báo cáo chi tiết (Precision, Recall, F1-Score):")
    print(classification_report(y_true, y_pred_hog_svm, zero_division=0))
    
    # --- Báo cáo cho CNN ---
    print("\n--- MÔ HÌNH CNN (face_recognition) ---")
    print(f"Tổng thời gian dự đoán: {cnn_time:.4f} giây")
    print(f"Tổng số lượng ảnh dự đoán: {total_test_images } ảnh")

    if total_test_images > 0:
        print(f"Thời gian trung bình/ảnh: {(cnn_time / total_test_images):.4f} giây")
    
    accuracy_cnn = accuracy_score(y_true, y_pred_cnn)
    print(f"Accuracy (Độ chính xác tổng thể): {accuracy_cnn:.4f}")
    print("Báo cáo chi tiết (Precision, Recall, F1-Score):")
    print(classification_report(y_true, y_pred_cnn, zero_division=0))
    print("="*70)

if __name__ == "__main__":
    # Huấn luyện mô hình HOG + SVM
    if os.path.exists(HOG_SVM_MODEL_PATH):
        print(f"Tải mô hình HOG + SVM đã có từ '{HOG_SVM_MODEL_PATH}'")
        with open(HOG_SVM_MODEL_PATH, 'rb') as f:
            hog_svm_model = pickle.load(f)
    else:
        hog_svm_model = train_hog_svm()
        
    # Tạo CSDL cho CNN
    known_encodings, known_names = create_cnn_database()
    
    # Đánh giá
    evaluate_models(hog_svm_model, known_encodings, known_names)