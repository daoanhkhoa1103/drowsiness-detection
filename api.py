from flask import Flask, request, jsonify
from PIL import Image
import cv2
from model import CNN
import torch
import torchvision.transforms as transforms
import numpy as np
import mediapipe as mp


app = Flask(__name__)


# Load model
model = CNN()
model.load_state_dict(torch.load('best_model_2.pth', map_location=torch.device('cpu')))
model.eval()


mp_face_mesh = mp.solutions.face_mesh

def predict(img, model):
    # Tiền xử lý ảnh
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img = transform(img)
    img = torch.unsqueeze(img, 0).to('cpu').float()

    # Dự đoán
    with torch.no_grad():
        output = model(img)

    # Xử lý kết quả
    _, predicted = torch.max(output.data, 1)
    prediction = "Mở mắt" if predicted.item() == 1 else "Nhắm mắt"

    return prediction

def plot_landmark(img_base, facial_area_obj, results):
    all_lm = []
    img = img_base.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    landmarks = results.multi_face_landmarks
    for source_idx, target_idx in facial_area_obj:
        source = landmarks.landmark[source_idx]
        target = landmarks.landmark[target_idx]


        relative_source = (int(img.shape * source.x), int(img.shape * source.y))
        relative_target = (int(img.shape * target.x), int(img.shape * target.y))
        all_lm.append(relative_source)
        all_lm.append(relative_target)
    
    all_lm = sorted(all_lm, key=lambda a: (a))
    x_min, x_max = all_lm, all_lm[-1]
    all_lm = sorted(all_lm, key=lambda a: (a))
    y_min, y_max = all_lm, all_lm[-1]
    
    img_ = img[y_min:y_max + 1, x_min:x_max + 1]
    return img_, [(x_min, y_min), (x_max, y_max)]


# API endpoint
@app.route('/predict', methods=['POST'])
def predict_drowsiness():
    if request.method == 'POST':
        # Lấy ảnh từ request
        file = request.files['image']
        img = Image.open(file.stream)


        # Chuyển đổi ảnh thành OpenCV format
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


        # Tiền xử lý ảnh và phát hiện mắt
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                l_eyebrow, _ = plot_landmark(img, mp_face_mesh.FACEMESH_LEFT_EYE, results)
                r_eyebrow, _ = plot_landmark(img, mp_face_mesh.FACEMESH_RIGHT_EYE, results)


                # Dự đoán
                pred_left = predict(l_eyebrow, model)
                pred_right = predict(r_eyebrow, model)


                # Trả về kết quả
                return jsonify({'left_eye': pred_left, 'right_eye': pred_right})
            else:
                return jsonify({'error': 'Không tìm thấy khuôn mặt'})


if __name__ == '__main__':
    app.run(debug=True)