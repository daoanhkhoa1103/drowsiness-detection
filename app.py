from flask import Flask, render_template, request, redirect, url_for
import sqlite3
from run import detect_drowsiness  # Import hàm detect_drowsiness từ run.py

app = Flask(__name__)

# Kết nối đến cơ sở dữ liệu
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

# Trang chủ
@app.route('/')
def index():
    conn = get_db_connection()
    drivers = conn.execute('SELECT * FROM drivers').fetchall()
    vehicles = conn.execute('SELECT * FROM vehicles').fetchall()
    routes = conn.execute('SELECT r.*, d.name AS driver_name, v.license_plate AS vehicle_license_plate '
                          'FROM routes r '
                          'JOIN drivers d ON r.driver_id = d.id '
                          'JOIN vehicles v ON r.vehicle_id = v.id').fetchall()

    # Tạo danh sách phương tiện với thông tin tài xế
    vehicles_with_drivers = []
    for vehicle in vehicles:
        driver = conn.execute('SELECT name FROM drivers WHERE id =?', (vehicle['driver_id'],)).fetchone() if vehicle['driver_id'] else None
        vehicle_with_driver = {
            'license_plate': vehicle['license_plate'],
            'vehicle_type': vehicle['vehicle_type'],
            'brand': vehicle['brand'],
            'driver_name': driver['name'] if driver else None
        }
        vehicles_with_drivers.append(vehicle_with_driver)

    conn.close()
    return render_template('index.html',
                           drivers=drivers,
                           vehicles=vehicles_with_drivers,
                           routes=routes)

# Trang chi tiết lái xe
@app.route('/driver/<int:driver_id>')
def driver_detail(driver_id):
    conn = get_db_connection()
    driver = conn.execute('SELECT * FROM drivers WHERE id =?', (driver_id,)).fetchone()
    routes = conn.execute('SELECT * FROM routes WHERE driver_id =?', (driver_id,)).fetchall()
    conn.close()
    return render_template('driver_detail.html', driver=driver, routes=routes)

# Trang chi tiết phương tiện
@app.route('/vehicle/<int:vehicle_id>')
def vehicle_detail(vehicle_id):
    conn = get_db_connection()
    vehicle = conn.execute('SELECT * FROM vehicles WHERE id =?', (vehicle_id,)).fetchone()
    routes = conn.execute('SELECT * FROM routes WHERE vehicle_id =?', (vehicle_id,)).fetchall()
    conn.close()
    return render_template('vehicle_detail.html', vehicle=vehicle, routes=routes)

# Trang thêm lái xe
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
        conn.execute('INSERT INTO drivers (name, birthdate, phone, address, license_number, license_issued_date, license_expiry_date) VALUES (?,?,?,?,?,?,?)',
                     (name, birthdate, phone, address, license_number, license_issued_date, license_expiry_date))
        conn.commit()
        conn.close()
        return redirect(url_for('index'))
    return render_template('add_driver.html')

# Trang thêm phương tiện
@app.route('/add_vehicle', methods=['GET', 'POST'])
def add_vehicle():
    if request.method == 'POST':
        license_plate = request.form['license_plate']
        vehicle_type = request.form['vehicle_type']
        brand = request.form['brand']
        chassis_number = request.form['chassis_number']
        engine_number = request.form['engine_number']
        driver_id = request.form.get('driver_id')  # Lấy driver_id từ form, cho phép null

        conn = get_db_connection()
        conn.execute('INSERT INTO vehicles (license_plate, vehicle_type, brand, chassis_number, engine_number, driver_id) VALUES (?,?,?,?,?,?)',
                     (license_plate, vehicle_type, brand, chassis_number, engine_number, driver_id))
        conn.commit()
        conn.close()
        return redirect(url_for('index'))

    conn = get_db_connection()
    drivers = conn.execute('SELECT * FROM drivers').fetchall()  # Lấy danh sách tài xế để hiển thị trong dropdown
    conn.close()
    return render_template('add_vehicle.html', drivers=drivers)

# Trang thêm lộ trình
@app.route('/add_route', methods=['GET', 'POST'])
def add_route():
    conn = get_db_connection()
    drivers = conn.execute('SELECT * FROM drivers').fetchall()
    vehicles = conn.execute('SELECT * FROM vehicles').fetchall()
    conn.close()

    if request.method == 'POST':
        driver_id = request.form['driver_id']
        vehicle_id = request.form['vehicle_id']
        start_point = request.form['start_point']
        end_point = request.form['end_point']
        start_time = request.form['start_time']
        end_time = request.form['end_time']
        distance = request.form['distance']
        fuel_consumption = request.form['fuel_consumption']

        conn = get_db_connection()
        conn.execute('INSERT INTO routes (driver_id, vehicle_id, start_point, end_point, start_time, end_time, distance, fuel_consumption) VALUES (?,?,?,?,?,?,?,?)',
                     (driver_id, vehicle_id, start_point, end_point, start_time, end_time, distance, fuel_consumption))
        conn.commit()
        conn.close()
        return redirect(url_for('index'))
    return render_template('add_route.html', drivers=drivers, vehicles=vehicles)

# Trang báo cáo
@app.route('/reports')
def reports():
    conn = get_db_connection()
    total_drivers = conn.execute('SELECT COUNT(*) FROM drivers').fetchone()
    total_vehicles = conn.execute('SELECT COUNT(*) FROM vehicles').fetchone()
    total_routes = conn.execute('SELECT COUNT(*) FROM routes').fetchone()

    driver_reports = conn.execute('''
        SELECT d.name, COUNT(r.id) AS route_count
        FROM drivers d
        LEFT JOIN routes r ON d.id = r.driver_id
        GROUP BY d.id
    ''').fetchall()

    vehicle_reports = conn.execute('''
        SELECT v.license_plate, COUNT(r.id) AS route_count
        FROM vehicles v
        LEFT JOIN routes r ON v.id = r.vehicle_id
        GROUP BY v.id
    ''').fetchall()

    conn.close()
    return render_template('reports.html',
                           total_drivers=total_drivers,
                           total_vehicles=total_vehicles,
                           total_routes=total_routes,
                           driver_reports=driver_reports,
                           vehicle_reports=vehicle_reports)

# Route phát hiện buồn ngủ
@app.route('/detect_drowsiness')
def detect_drowsiness_route():
    # Gọi hàm detect_drowsiness từ run.py
    drowsiness_status = detect_drowsiness()
    return render_template('detect_drowsiness.html', drowsiness_status=drowsiness_status)

if __name__ == '__main__':
    app.run(debug=True, port=8000)