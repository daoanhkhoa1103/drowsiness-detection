import sqlite3

# Kết nối đến cơ sở dữ liệu (tạo mới nếu chưa tồn tại)
conn = sqlite3.connect('database.db')

# Tạo bảng drivers
conn.execute('''
CREATE TABLE IF NOT EXISTS drivers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    birthdate DATE,
    phone TEXT,
    address TEXT,
    license_number TEXT,
    license_issued_date DATE,
    license_expiry_date DATE
)
''')

# Tạo bảng vehicles
conn.execute('''
CREATE TABLE IF NOT EXISTS vehicles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    license_plate TEXT NOT NULL,
    vehicle_type TEXT,
    brand TEXT,
    chassis_number TEXT,
    engine_number TEXT,
    driver_id INTEGER,
    FOREIGN KEY (driver_id) REFERENCES drivers (id)
)
''')

# Tạo bảng routes
conn.execute('''
CREATE TABLE IF NOT EXISTS routes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    driver_id INTEGER NOT NULL,
    vehicle_id INTEGER NOT NULL,
    start_point TEXT NOT NULL,
    end_point TEXT NOT NULL,
    start_time DATETIME NOT NULL,
    end_time DATETIME,
    distance REAL,
    fuel_consumption REAL,
    FOREIGN KEY (driver_id) REFERENCES drivers (id),
    FOREIGN KEY (vehicle_id) REFERENCES vehicles (id)
)
''')

# Thêm dữ liệu cho bảng drivers
drivers_data = [
    ('Nguyễn Văn A', '1990-05-10', '0987654321', '123 Đường ABC, Quận 1', 'A1234567', '2020-01-15', '2025-01-15'),
    ('Trần Thị B', '1985-12-20', '0912345678', '456 Đường DEF, Quận 2', 'B7654321', '2018-05-10', '2023-05-10'),
    ('Lê Văn C', '1995-08-15', '0901234567', '789 Đường GHI, Quận 3', 'C1122334', '2021-10-20', '2026-10-20')
]
conn.executemany("INSERT INTO drivers (name, birthdate, phone, address, license_number, license_issued_date, license_expiry_date) VALUES (?,?,?,?,?,?,?)", drivers_data)

# Thêm dữ liệu cho bảng vehicles
vehicles_data = [
    ('51A-123.45', 'Xe tải', 'Hino', 'ABC123456789', 'DEF987654321', 1),
    ('51B-543.21', 'Xe khách', 'Thaco', 'GHI987654321', 'JKL123456789', 2),
    ('51C-987.65', 'Xe con', 'Toyota', 'MNO123456789', 'PQR987654321', 3)
]
conn.executemany("INSERT INTO vehicles (license_plate, vehicle_type, brand, chassis_number, engine_number, driver_id) VALUES (?,?,?,?,?,?)", vehicles_data)

# Thêm dữ liệu cho bảng routes
routes_data = [
    (1, 1, 'TP. Hồ Chí Minh', 'Hà Nội', '2023-02-20 08:00:00', '2023-02-21 17:00:00', 1700, 200),
    (2, 2, 'Đà Nẵng', 'Nha Trang', '2023-02-21 10:00:00', '2023-02-22 14:00:00', 600, 80),
    (3, 3, 'Biên Hòa, Đồng Nai', 'Vũng Tàu', '2023-02-22 12:00:00', '2023-02-22 18:00:00', 100, 15)
]
conn.executemany("INSERT INTO routes (driver_id, vehicle_id, start_point, end_point, start_time, end_time, distance, fuel_consumption) VALUES (?,?,?,?,?,?,?,?)", routes_data)

# Commit các thay đổi và đóng kết nối
conn.commit()
conn.close()