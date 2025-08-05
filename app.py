import eventlet
eventlet.monkey_patch()  # PHẢI ở trên cùng!

from flask import Flask, render_template, request, redirect, url_for, Response
from flask_socketio import SocketIO
import sqlite3
import cv2
import torch
import torch.nn.functional as F
from model import CNN
from PIL import Image
import mediapipe as mp
import numpy as np
import threading
import time
import os
import torchvision.transforms as transforms

print("1. app.py started")
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
DATABASE = "database.db"

# ---------- DATABASE INIT ----------
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    with open('schema.sql') as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()

# ----------- AI MODEL SETUP ----------
print("2. mediapipe loaded")
mp_face_mesh = mp.solutions.face_mesh

print("3. Load Model...")
model = CNN()
model.load_state_dict(torch.load('best_model_2.pth', map_location=torch.device('cpu')))
model.eval()
print("4. Model loaded!")

transform_val = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
label2id = {0: 'Close', 1: 'Open'}

latest_frame = None
latest_status = {"drowsy": False, "count": 0, "eye_status": "Unknown"}

def plot_landmark(img_base, facial_area_obj, results):
    all_lm = []
    img = img_base.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    landmarks = results.multi_face_landmarks[0]
    for source_idx, target_idx in facial_area_obj:
        source = landmarks.landmark[source_idx]
        target = landmarks.landmark[target_idx]
        relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
        relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))
        all_lm.append(relative_source)
        all_lm.append(relative_target)
    all_lm = sorted(all_lm, key=lambda a: (a[0]))
    x_min, x_max = all_lm[0][0], all_lm[-1][0]
    all_lm = sorted(all_lm, key=lambda a: (a[1]))
    y_min, y_max = all_lm[0][1], all_lm[-1][1]
    img_ = img[y_min:y_max + 1, x_min:x_max + 1]
    return img_, [(x_min, y_min), (x_max, y_max)]

def predict(img, model):
    img = transform_val(img)
    img = torch.unsqueeze(img, 0).to('cpu').float()
    with torch.no_grad():
        output = model(img)
    output = F.softmax(output, dim=-1)
    predicted = torch.argmax(output)
    p = label2id[predicted.item()]
    prob = torch.max(output).item()
    return p, round(prob, 2)

def drowsiness_detection_loop():
    global latest_frame, latest_status
    cap = cv2.VideoCapture(0)
    count = 0
    count_all = 0
    while True:
        ret, image = cap.read()
        if not ret:
            continue
        drowsy = False
        eye_status = "Unknown"
        try:
            with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                check_l = 0
                check_r = 0
                for face_landmarks in results.multi_face_landmarks:
                    l_eyebrow, coor1 = plot_landmark(image, mp_face_mesh.FACEMESH_LEFT_EYE, results)
                    imgL = Image.fromarray(l_eyebrow)
                    pred_l, prob_l = predict(imgL, model)

                    r_eyebrow, coor2 = plot_landmark(image, mp_face_mesh.FACEMESH_RIGHT_EYE, results)
                    imgR = Image.fromarray(r_eyebrow)
                    pred_r, prob_r = predict(imgR, model)

                    # Draw status
                    cv2.putText(image, f"L: {pred_l} {prob_l}", (coor1[0][0], coor1[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                    cv2.putText(image, f"R: {pred_r} {prob_r}", (coor2[0][0], coor2[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

                    if str(pred_l) == 'Close' and prob_l > 0.75:
                        check_l = 1
                    if str(pred_r) == 'Close' and prob_r > 0.75:
                        check_r = 1

                if check_l == 1 and check_r == 1:
                    count += 1
                    eye_status = "Closed"
                else:
                    count = 0
                    eye_status = "Open"
                if count > 15:
                    drowsy = True
                    if count == 16:
                        count_all += 1
                else:
                    drowsy = False
            else:
                count = 0
                drowsy = False
                eye_status = "No face detected"
        except Exception as e:
            print(f"Error during processing: {e}")
            drowsy = False
            eye_status = "Error"

        latest_status = {
            "drowsy": drowsy,
            "count": count_all,
            "eye_status": eye_status
        }
        latest_frame = image.copy()
        socketio.emit('drowsiness_status', latest_status)
        time.sleep(0.13)
    cap.release()

def gen_video():
    global latest_frame
    while True:
        if latest_frame is not None:
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.04)

# --------------- ROUTES ---------------
@app.route('/')
def index():
    conn = get_db_connection()
    drivers = conn.execute('SELECT * FROM drivers').fetchall()
    vehicles = conn.execute('SELECT * FROM vehicles').fetchall()
    routes = conn.execute(
        '''SELECT r.*, d.name AS driver_name, v.license_plate AS vehicle_license_plate
           FROM routes r
           JOIN drivers d ON r.driver_id = d.id
           JOIN vehicles v ON r.vehicle_id = v.id'''
    ).fetchall()
    conn.close()
    return render_template('index.html',
                           drivers=drivers,
                           vehicles=vehicles,
                           routes=routes)

# ---- DRIVERS ----
@app.route('/drivers')
def drivers():
    conn = get_db_connection()
    drivers = conn.execute('SELECT * FROM drivers').fetchall()
    conn.close()
    return render_template('drivers.html', drivers=drivers)

@app.route('/add_driver', methods=['GET', 'POST'])
def add_driver():
    if request.method == 'POST':
        name = request.form['name']
        birthdate = request.form['birthdate']
        phone = request.form['phone']
        address = request.form['address']
        license_number = request.form['license_number']
        license_issued_date = request.form['license_issued_date']
        license_expiry_date = request.form['license_expiry_date']

        conn = get_db_connection()
        conn.execute('''
            INSERT INTO drivers
            (name, birthdate, phone, address, license_number, license_issued_date, license_expiry_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (name, birthdate, phone, address, license_number, license_issued_date, license_expiry_date))
        conn.commit()
        conn.close()
        return redirect(url_for('drivers'))
    return render_template('add_driver.html')

# ---- VEHICLES ----
@app.route('/vehicles')
def vehicles():
    conn = get_db_connection()
    vehicles = conn.execute('SELECT * FROM vehicles').fetchall()
    conn.close()
    return render_template('vehicles.html', vehicles=vehicles)

@app.route('/add_vehicle', methods=['GET', 'POST'])
def add_vehicle():
    conn = get_db_connection()
    drivers = conn.execute('SELECT * FROM drivers').fetchall()
    if request.method == 'POST':
        license_plate = request.form['license_plate']
        vehicle_type = request.form['vehicle_type']
        brand = request.form['brand']
        chassis_number = request.form['chassis_number']
        engine_number = request.form['engine_number']
        driver_id = request.form.get('driver_id')
        if driver_id == "None" or driver_id == "":
            driver_id = None

        conn.execute('''
            INSERT INTO vehicles
            (license_plate, vehicle_type, brand, chassis_number, engine_number, driver_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (license_plate, vehicle_type, brand, chassis_number, engine_number, driver_id))
        conn.commit()
        conn.close()
        return redirect(url_for('vehicles'))
    conn.close()
    return render_template('add_vehicle.html', drivers=drivers)

# ---- ROUTES ----
@app.route('/routes')
def routes():
    conn = get_db_connection()
    routes = conn.execute(
        '''SELECT r.*, d.name AS driver_name, v.license_plate AS vehicle_license_plate
           FROM routes r
           JOIN drivers d ON r.driver_id = d.id
           JOIN vehicles v ON r.vehicle_id = v.id'''
    ).fetchall()
    conn.close()
    return render_template('routes.html', routes=routes)

@app.route('/add_route', methods=['GET', 'POST'])
def add_route():
    conn = get_db_connection()
    drivers = conn.execute('SELECT * FROM drivers').fetchall()
    vehicles = conn.execute('SELECT * FROM vehicles').fetchall()
    if request.method == 'POST':
        driver_id = request.form['driver_id']
        vehicle_id = request.form['vehicle_id']
        start_point = request.form['start_point']
        end_point = request.form['end_point']
        start_time = request.form['start_time']
        end_time = request.form['end_time']
        distance = request.form['distance']
        fuel_consumption = request.form['fuel_consumption']

        conn.execute('''
            INSERT INTO routes
            (driver_id, vehicle_id, start_point, end_point, start_time, end_time, distance, fuel_consumption)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (driver_id, vehicle_id, start_point, end_point, start_time, end_time, distance, fuel_consumption))
        conn.commit()
        conn.close()
        return redirect(url_for('routes'))
    conn.close()
    return render_template('add_route.html', drivers=drivers, vehicles=vehicles)

# ---- DROWSINESS DETECTION ----
@app.route('/detect_drowsiness')
def detect_drowsiness_page():
    return render_template('detect_drowsiness.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def connect():
    print('Client connected!')
    if not hasattr(app, 'thread') or not app.thread.is_alive():
        app.thread = threading.Thread(target=drowsiness_detection_loop)
        app.thread.daemon = True
        app.thread.start()

if __name__ == '__main__':
    print("5. Prepare to run server")
    if not os.path.exists(DATABASE):
        print("6. Init DB (not found)")
        init_db()
    else:
        print("6. Found existing DB")
    print("7. Running Flask-SocketIO...")
    socketio.run(app, host="0.0.0.0", port=8000)
