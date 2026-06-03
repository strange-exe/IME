-- Tonnage Records Table
CREATE TABLE IF NOT EXISTS tonnage_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vessel_name TEXT NOT NULL,
    account_name TEXT,
    open_port TEXT,
    open_date TEXT,
    vessel_type TEXT,
    vessel_size TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Voyage Charter (VC) Records Table
CREATE TABLE IF NOT EXISTS vc_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_name TEXT,
    cargo_name TEXT NOT NULL,
    loading_port TEXT,
    discharge_port TEXT,
    laycan TEXT,
    cargo_type TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Time Charter (TC) Records Table
CREATE TABLE IF NOT EXISTS tc_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_name TEXT,
    cargo_name TEXT NOT NULL,
    delivery_port TEXT,
    redelivery_port TEXT,
    duration TEXT,
    laycan TEXT,
    cargo_type TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
