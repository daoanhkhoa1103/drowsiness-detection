DROP TABLE IF EXISTS drivers;
DROP TABLE IF EXISTS vehicles;
DROP TABLE IF EXISTS routes;

CREATE TABLE drivers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    birthdate TEXT NOT NULL,
    phone TEXT,
    address TEXT,
    license_number TEXT,
    license_issued_date TEXT,
    license_expiry_date TEXT
);

CREATE TABLE vehicles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    license_plate TEXT NOT NULL,
    vehicle_type TEXT,
    brand TEXT,
    chassis_number TEXT,
    engine_number TEXT,
    driver_id INTEGER,
    FOREIGN KEY (driver_id) REFERENCES drivers (id)
);

CREATE TABLE routes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    driver_id INTEGER,
    vehicle_id INTEGER,
    start_point TEXT,
    end_point TEXT,
    start_time TEXT,
    end_time TEXT,
    distance REAL,
    fuel_consumption REAL,
    FOREIGN KEY (driver_id) REFERENCES drivers (id),
    FOREIGN KEY (vehicle_id) REFERENCES vehicles (id)
);

-- Dữ liệu mẫu
INSERT INTO drivers (name, birthdate, phone, address, license_number, license_issued_date, license_expiry_date)
VALUES
('Nguyen Van A', '1980-05-15', '0909123456', 'Hanoi', '123456', '2015-01-01', '2030-01-01'),
('Tran Thi B', '1985-09-20', '0909888777', 'TP Ho Chi Minh', '654321', '2016-03-01', '2031-03-01');

INSERT INTO vehicles (license_plate, vehicle_type, brand, chassis_number, engine_number, driver_id)
VALUES
('51A-12345', 'Truck', 'Hyundai', 'CH1234', 'EN5678', 1),
('29B-88888', 'Bus', 'Ford', 'CH9999', 'EN8888', 2);

INSERT INTO routes (driver_id, vehicle_id, start_point, end_point, start_time, end_time, distance, fuel_consumption)
VALUES
(1, 1, 'Hanoi', 'Hai Phong', '2025-06-12T08:00', '2025-06-12T11:00', 120.0, 30.5),
(2, 2, 'TP Ho Chi Minh', 'Can Tho', '2025-06-12T09:00', '2025-06-12T13:00', 170.0, 40.2);
