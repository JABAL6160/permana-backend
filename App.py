from flask import Flask, jsonify, request
from flask_cors import CORS
import bcrypt
import psycopg2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from werkzeug.security import generate_password_hash, check_password_hash
import os, requests
import datetime

# ===========================
# PATH RELATIVE (WAJIB)
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "model_cuaca.keras")
DATASET_PATH = os.path.join(BASE_DIR, "data_cuaca.csv")


# ===========================================================
# ⚙️ KONFIGURASI DASAR
# ===========================================================

app = Flask(__name__)
CORS(app, resources={
       r"/*": {
           "origins": [
               "https://permanagis.vercel.app",
               "https://*.vercel.app",
               "https://your-flask-api.railway.app",  # Tambah ini
               "http://localhost:5173",
               "http://localhost:3000"
           ]
       }
   })


def get_conn():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL belum diset di environment")
    return psycopg2.connect(database_url)


# ===========================
#  HELPER: LOG ACTIVITY
# ===========================
def log_activity(user_id, username, role, action, description, ip_address=None):
    """Helper function to log user activities"""
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO activity_logs (user_id, username, role, action, description, ip_address)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (user_id, username, role, action, description, ip_address))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[ERROR] Log activity failed: {e}")

# ===========================
#  KONFIGURASI API OPENWEATHER
# ===========================
OPENWEATHER_API_KEY = os.getenv ("c91b0fa374f991fb0da760d7c98d8f7f")

if not OPENWEATHER_API_KEY:
    print("⚠️ OPENWEATHER_API_KEY belum diset")

# ===========================
#  LAZY LOADING untuk ML Libraries
# ===========================
_ml_modules = {}

def get_ml_modules():
    """Lazy load ML modules hanya saat dibutuhkan"""
    if not _ml_modules:
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.models import Sequential, load_model
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        
        _ml_modules['pd'] = pd
        _ml_modules['MinMaxScaler'] = MinMaxScaler
        _ml_modules['Sequential'] = Sequential
        _ml_modules['load_model'] = load_model
        _ml_modules['LSTM'] = LSTM
        _ml_modules['Dense'] = Dense
        _ml_modules['Dropout'] = Dropout
        _ml_modules['EarlyStopping'] = EarlyStopping
        _ml_modules['ModelCheckpoint'] = ModelCheckpoint
        
    return _ml_modules

# Cache untuk model ML
_weather_model = None
_weather_scaler = None

def get_weather_model():
    """Load model hanya sekali dan cache"""
    global _weather_model, _weather_scaler
    
    if _weather_model is None:
        ml = get_ml_modules()
        model_path = MODEL_PATH
        
        if os.path.exists(model_path):
            _weather_model = ml['load_model'](model_path, compile=False)
            
            # Load dataset untuk scaler
            df = load_dataset()
            ml_pd = ml['pd']
            df["temperature"] = ml_pd.to_numeric(df["temperature"], errors="coerce")
            df = df.dropna(subset=["temperature"])
            data = df["temperature"].values.reshape(-1, 1)
            _weather_scaler = ml['MinMaxScaler'](feature_range=(0, 1))
            _weather_scaler.fit(data)
            
    return _weather_model, _weather_scaler

    
# ===========================================================
# LOGIN STAFF & ADMIN (WITH LOGGING)
# ===========================================================
    
@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    ip_address = request.remote_addr

    conn = get_conn()
    cur = conn.cursor()

    # Cek Admin
    cur.execute("SELECT id, username, email, password_hash, 'admin' as role FROM admins WHERE username=%s", (username,))
    user = cur.fetchone()

    if user:
        password_hash = user[3]
        
        # ✅ Admin menggunakan bcrypt
        if password_hash.startswith("$2a$") or password_hash.startswith("$2b$"):
            if bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8")):
                # 📝 LOG ACTIVITY
                log_activity(user[0], user[1], 'admin', 'login', f'Admin {user[1]} logged in', ip_address)
                
                cur.close()
                conn.close()
                return jsonify({
                    "success": True,
                    "user": {
                        "id": user[0],
                        "username": user[1],
                        "email": user[2],
                        "role": user[4]
                    }
                }), 200

    # Cek Staff
    cur.execute("SELECT id, username, email, password_hash, role FROM staff WHERE username=%s", (username,))
    user = cur.fetchone()

    if user:
        password_hash = user[3]
        
        # ✅ Staff menggunakan werkzeug (dari register)
        if check_password_hash(password_hash, password):
            # 📝 LOG ACTIVITY
            log_activity(user[0], user[1], 'staff', 'login', f'Staff {user[1]} logged in', ip_address)
            
            cur.close()
            conn.close()
            return jsonify({
                "success": True,
                "user": {
                    "id": user[0],
                    "username": user[1],
                    "email": user[2],
                    "role": user[4]
                }
            }), 200

    cur.close()
    conn.close()
    return jsonify({"error": "Username atau password salah"}), 401


# ===========================================================
# 🗝️ REGISTER STAFF
# ===========================================================

@app.route("/api/staff/register", methods=["POST"])
def register_staff():
    try:
        data = request.get_json()
        fullname = data.get("fullname")
        nip = data.get("nip")
        email = data.get("email")
        username = data.get("username")
        password = data.get("password")
        reason = data.get("reason")

        conn = get_conn()
        cur = conn.cursor()

        # Cek apakah sudah ada di staff (sudah disetujui)
        cur.execute("SELECT id FROM staff WHERE email=%s OR username=%s", (email, username))
        if cur.fetchone():
            cur.close()
            conn.close()
            return jsonify({"success": False, "error": "Akun sudah aktif, silakan login"}), 400
        
        # Cek apakah sudah pending / rejected
        cur.execute("SELECT status FROM pending_user WHERE email=%s OR username=%s", (email, username))
        status = cur.fetchone()

        if status:
            status = status[0]

            if status == "pending":
                cur.close()
                conn.close()
                return jsonify({"success": False, "error": "Permohonan kamu masih menunggu review admin"}), 400
            
            if status == "approved":
                cur.close()
                conn.close()
                return jsonify({"success": False, "error": "Akun kamu sudah disetujui, silakan login"}), 400

            if status == "rejected":
                # Overwrite request baru
                password_hash = generate_password_hash(password)
                cur.execute("""
                    UPDATE pending_user
                    SET fullname=%s, nip=%s, email=%s, username=%s, password_hash=%s,
                        reason=%s, status='pending', created_at=NOW()
                    WHERE email=%s OR username=%s
                """, (fullname, nip, email, username, password_hash, reason, email, username))
                conn.commit()
                cur.close()
                conn.close()
                return jsonify({"success": True, "message": "Permohonan baru dikirim ulang"}), 200

        # Input baru
        password_hash = generate_password_hash(password)
        cur.execute("""
            INSERT INTO pending_user (fullname, nip, email, username, password_hash, reason, status)
            VALUES (%s, %s, %s, %s, %s, %s, 'pending')
        """, (fullname, nip, email, username, password_hash, reason))

        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"success": True, "message": "Permohonan telah dikirim"}), 201

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ===========================================================
# GET PENDING STAFF
# ===========================================================

@app.route("/api/staff/pending", methods=["GET"])
def get_pending_staff():
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, fullname, nip, email, username, reason, status, created_at
            FROM pending_user
            WHERE status='pending'
            ORDER BY created_at DESC
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        result = [{
            "id": r[0],
            "fullname": r[1],
            "nip": r[2],
            "email": r[3],
            "username": r[4],
            "reason": r[5],
            "status": r[6],
            "created_at": str(r[7])
        } for r in rows]

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================================================
# APPROVE STAFF (WITH LOGGING)
# ===========================================================

@app.route("/api/staff/<int:id>/approve", methods=["POST"])
def approve_staff(id):
    try:
        # Get admin info from request headers (sent from frontend)
        admin_username = request.headers.get('X-User-Username', 'Unknown Admin')
        admin_id = request.headers.get('X-User-Id', 0)
        
        conn = get_conn()
        cur = conn.cursor()

        cur.execute("""
            SELECT fullname, nip, email, username, password_hash
            FROM pending_user WHERE id=%s
        """, (id,))
        user = cur.fetchone()

        if not user:
            cur.close()
            conn.close()
            return jsonify({"success": False, "error": "User tidak ditemukan"}), 404

        fullname, nip, email, username, password_hash = user

        # Insert ke tabel staff otomatis
        cur.execute("""
            INSERT INTO staff (fullname, nip, email, username, password_hash, role, created_at)
            VALUES (%s, %s, %s, %s, %s, 'staff', NOW())
        """, (fullname, nip, email, username, password_hash))

        # Update pending_user → approved
        cur.execute("UPDATE pending_user SET status='approved' WHERE id=%s", (id,))
        conn.commit()
        
        # 📝 LOG ACTIVITY
        log_activity(
            admin_id, 
            admin_username, 
            'admin', 
            'approve_staff', 
            f'Admin {admin_username} approved staff registration for {username} ({fullname})',
            request.remote_addr
        )
        
        cur.close()
        conn.close()

        return jsonify({"success": True, "message": "Akun Staff telah disetujui dan aktif"}), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    
# ===========================================================
# REJECT STAFF (WITH LOGGING)
# ===========================================================

@app.route("/api/staff/<int:id>/reject", methods=["POST"])
def reject_staff(id):
    try:
        # Get admin info from request headers
        admin_username = request.headers.get('X-User-Username', 'Unknown Admin')
        admin_id = request.headers.get('X-User-Id', 0)
        
        conn = get_conn()
        cur = conn.cursor()
        
        # Get pending user info before rejecting
        cur.execute("SELECT username, fullname FROM pending_user WHERE id=%s", (id,))
        pending_user = cur.fetchone()

        cur.execute("UPDATE pending_user SET status='rejected' WHERE id=%s", (id,))
        conn.commit()
        
        # 📝 LOG ACTIVITY
        if pending_user:
            log_activity(
                admin_id,
                admin_username,
                'admin',
                'reject_staff',
                f'Admin {admin_username} rejected staff registration for {pending_user[0]} ({pending_user[1]})',
                request.remote_addr
            )
        
        cur.close()
        conn.close()

        return jsonify({"success": True, "message": "Permohonan ditolak"}), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    
#===========================================================
# GET ALL USERS (ADMIN + STAFF) - untuk Manage Staff Tab
# ===========================================================

@app.route("/api/admin/users", methods=["GET"])
def get_all_users():
    """Get semua users (admins + staff) untuk ditampilkan di Manage Staff"""
    try:
        conn = get_conn()
        cur = conn.cursor()
        
        # Query untuk mengambil dari kedua tabel (admins dan staff)
        cur.execute("""
            SELECT id, username, email, 'admin' as role, NULL as fullname, NULL as nip
            FROM admins
            UNION ALL
            SELECT id, username, email, role, fullname, nip
            FROM staff
            ORDER BY id;
        """)
        
        rows = cur.fetchall()
        cur.close()
        conn.close()

        result = [{
            "id": r[0],
            "username": r[1],
            "email": r[2],
            "role": r[3],
            "fullname": r[4],
            "nip": r[5]
        } for r in rows]

        return jsonify(result), 200

    except Exception as e:
        print("[ERROR] get_all_users:", e)
        return jsonify({"error": str(e)}), 500


# ===========================================================
# DELETE USER (WITH LOGGING)
# ===========================================================

@app.route("/api/admin/users/<int:user_id>", methods=["DELETE"])
def delete_user(user_id):
    """Delete user dari tabel admins atau staff"""
    try:
        # Get admin info from request headers
        admin_username = request.headers.get('X-User-Username', 'Unknown Admin')
        admin_id = request.headers.get('X-User-Id', 0)
        
        conn = get_conn()
        cur = conn.cursor()
        
        # Cek apakah user ada di tabel admins
        cur.execute("SELECT id, username FROM admins WHERE id = %s", (user_id,))
        admin = cur.fetchone()
        
        if admin:
            cur.execute("DELETE FROM admins WHERE id = %s RETURNING id", (user_id,))
            deleted = cur.fetchone()
            conn.commit()
            
            # 📝 LOG ACTIVITY
            log_activity(
                admin_id,
                admin_username,
                'admin',
                'delete_user',
                f'Admin {admin_username} deleted admin user {admin[1]} (ID: {user_id})',
                request.remote_addr
            )
            
            cur.close()
            conn.close()
            
            if deleted:
                return jsonify({"success": True, "message": "Admin berhasil dihapus"}), 200
        
        # Jika tidak ada di admins, cek di staff
        cur.execute("SELECT id, username, fullname FROM staff WHERE id = %s", (user_id,))
        staff = cur.fetchone()
        
        if staff:
            cur.execute("DELETE FROM staff WHERE id = %s RETURNING id", (user_id,))
            deleted = cur.fetchone()
            conn.commit()
            
            # 📝 LOG ACTIVITY
            log_activity(
                admin_id,
                admin_username,
                'admin',
                'delete_user',
                f'Admin {admin_username} deleted staff user {staff[1]} ({staff[2]}, ID: {user_id})',
                request.remote_addr
            )
            
            cur.close()
            conn.close()
            
            if deleted:
                return jsonify({"success": True, "message": "Staff berhasil dihapus"}), 200
        
        # Jika tidak ditemukan di kedua tabel
        cur.close()
        conn.close()
        return jsonify({"success": False, "error": "User tidak ditemukan"}), 404

    except Exception as e:
        print("[ERROR] delete_user:", e)
        return jsonify({"success": False, "error": str(e)}), 500


# ===========================================================
# 🗺️ DATA POLES & KNN
# ===========================================================

@app.route("/api/poles/<poles>", methods=["GET"])
def get_poles(poles):
    try:
        alias_map = {"odp": "odp_poles", "odc": "odc_poles", "pop": "pop_poles"}
        poles = alias_map.get(poles, poles)
        if poles not in ["odp_poles", "odc_poles", "pop_poles"]:
            return jsonify({"error": "Invalid table name"}), 400

        conn = get_conn()
        cur = conn.cursor()
        cur.execute(f"""
            SELECT id, poles_code, marking_date, location, attached_cables,
                   ST_Y(geom::geometry) AS lat, ST_X(geom::geometry) AS lon
            FROM {poles};
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        results = [{
            "id": r[0],
            "poles_code": r[1],
            "marking_date": str(r[2]) if r[2] else None,
            "location": r[3],
            "attached_cables": r[4],
            "lat": r[5],
            "lon": r[6]
        } for r in rows]
        return jsonify(results)
    except Exception as e:
        print("[ERROR] get_poles:", e)
        return jsonify([])

@app.route("/api/nearest_poles_ml", methods=["GET"])
def get_nearest_poles_ml():
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))

        conn = get_conn()
        cur = conn.cursor()

        def fetch_coords(poles):
            cur.execute(f"""
                SELECT id, poles_code, ST_Y(geom::geometry) AS lat, ST_X(geom::geometry) AS lon
                FROM {poles};
            """)
            rows = cur.fetchall()
            return np.array([[r[2], r[3]] for r in rows]), rows

        odp_points, odp_rows = fetch_coords("odp_poles")
        odc_points, odc_rows = fetch_coords("odc_poles")
        pop_points, pop_rows = fetch_coords("pop_poles")

        def knn_search(points, rows, k):
            if len(points) == 0:
                return []
            model = NearestNeighbors(n_neighbors=min(k, len(points)), metric='haversine')
            model.fit(np.radians(points))
            distances, indices = model.kneighbors(np.radians([[lat, lon]]))
            earth_radius = 6371000
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                r = rows[idx]
                results.append({
                    "id": r[0],
                    "poles_code": r[1],
                    "lat": r[2],
                    "lon": r[3],
                    "distance": round(dist * earth_radius, 2)
                })
            return results

        odp_result = knn_search(odp_points, odp_rows, 3)
        odc_result = knn_search(odc_points, odc_rows, 1)
        pop_result = knn_search(pop_points, pop_rows, 1)

        cur.close()
        conn.close()

        return jsonify({
            "odp_nearest": odp_result,
            "odc_nearest": odc_result,
            "pop_nearest": pop_result
        })
    except Exception as e:
        print("[ERROR] get_nearest_poles_ml:", e)
        return jsonify({"error": str(e)}), 500

# ===========================================================
# POLES CRUD (WITH LOGGING)
# ===========================================================
POLES_ALIAS = {
    "odp": "odp_poles",
    "odc": "odc_poles",
    "pop": "pop_poles"
}

@app.route("/api/poles/<poles>/<int:id>", methods=["GET"])
def get_pole_by_id(poles, id):
    table = POLES_ALIAS.get(poles)
    if not table:
        return jsonify({"error": "Invalid poles type"}), 400

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"""
        SELECT id, poles_code, marking_date, location, attached_cables,
               ST_Y(geom::geometry) AS lat, ST_X(geom::geometry) AS lon
        FROM {table} WHERE id = %s;
    """, (id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        return jsonify({"error": "Not found"}), 404

    return jsonify({
        "id": row[0],
        "poles_code": row[1],
        "marking_date": str(row[2]) if row[2] else None,
        "location": row[3],
        "attached_cables": row[4],
        "lat": row[5],
        "lon": row[6]
    })

@app.route("/api/poles/<poles>", methods=["POST"])
def create_pole(poles):
    table = POLES_ALIAS.get(poles)
    if not table:
        return jsonify({"error": "Invalid poles type"}), 400

    data = request.get_json()

    # Get user info from request headers
    username = request.headers.get('X-User-Username', 'Unknown User')
    user_id = request.headers.get('X-User-Id', 0)
    user_role = request.headers.get('X-User-Role', 'staff')

    required = ["poles_code", "lat", "lon"]
    for r in required:
        if r not in data:
            return jsonify({"error": f"Missing field: {r}"}), 400

    poles_code = data["poles_code"]
    marking_date = data.get("marking_date")
    location = data.get("location")
    attached_cables = data.get("attached_cables")
    lat = data["lat"]
    lon = data["lon"]

    pop_id = data.get("pop_id") if poles == "odp" else None

    conn = get_conn()
    cur = conn.cursor()

    if poles == "odp":
        cur.execute(f"""
            INSERT INTO {table} 
                (poles_code, marking_date, location, attached_cables, geom, pop_id)
            VALUES (%s, %s, %s, %s, ST_SetSRID(ST_Point(%s, %s), 4326), %s)
            RETURNING id;
        """, (poles_code, marking_date, location, attached_cables, lon, lat, pop_id))
    else:
        cur.execute(f"""
            INSERT INTO {table} 
                (poles_code, marking_date, location, attached_cables, geom)
            VALUES (%s, %s, %s, %s, ST_SetSRID(ST_Point(%s, %s), 4326))
            RETURNING id;
        """, (poles_code, marking_date, location, attached_cables, lon, lat))

    new_id = cur.fetchone()[0]
    conn.commit()
    
    # 📝 LOG ACTIVITY
    log_activity(
        user_id,
        username,
        user_role,
        'create_pole',
        f'{username} created new {poles.upper()} pole: {poles_code} at {location}',
        request.remote_addr
    )
    
    cur.close()
    conn.close()

    return jsonify({"success": True, "id": new_id}), 201

@app.route("/api/poles/<poles>/<int:id>", methods=["PUT"])
def update_pole(poles, id):
    table = POLES_ALIAS.get(poles)
    if not table:
        return jsonify({"error": "Invalid poles type"}), 400

    data = request.get_json()

    # Get user info from request headers
    username = request.headers.get('X-User-Username', 'Unknown User')
    user_id = request.headers.get('X-User-Id', 0)
    user_role = request.headers.get('X-User-Role', 'staff')

    poles_code = data.get("poles_code")
    marking_date = data.get("marking_date")
    location = data.get("location")
    attached_cables = data.get("attached_cables")
    lat = data.get("lat")
    lon = data.get("lon")
    pop_id = data.get("pop_id") if poles == "odp" else None

    conn = get_conn()
    cur = conn.cursor()

    if poles == "odp":
        cur.execute(f"""
            UPDATE {table}
            SET poles_code=%s, marking_date=%s, location=%s,
                attached_cables=%s, geom=ST_SetSRID(ST_Point(%s, %s), 4326),
                pop_id=%s
            WHERE id=%s RETURNING id;
        """, (poles_code, marking_date, location, attached_cables, lon, lat, pop_id, id))
    else:
        cur.execute(f"""
            UPDATE {table}
            SET poles_code=%s, marking_date=%s, location=%s,
                attached_cables=%s, geom=ST_SetSRID(ST_Point(%s, %s), 4326)
            WHERE id=%s RETURNING id;
        """, (poles_code, marking_date, location, attached_cables, lon, lat, id))

    updated = cur.fetchone()

    conn.commit()
    
    # 📝 LOG ACTIVITY
    if updated:
        log_activity(
            user_id,
            username,
            user_role,
            'update_pole',
            f'{username} updated {poles.upper()} pole ID {id}: {poles_code} at {location}',
            request.remote_addr
        )
    
    cur.close()
    conn.close()

    if not updated:
        return jsonify({"error": "Not found"}), 404

    return jsonify({"success": True})

@app.route("/api/poles/<poles>/<int:id>", methods=["DELETE"])
def delete_pole(poles, id):
    table = POLES_ALIAS.get(poles)
    if not table:
        return jsonify({"error": "Invalid poles type"}), 400

    # Get user info from request headers
    username = request.headers.get('X-User-Username', 'Unknown User')
    user_id = request.headers.get('X-User-Id', 0)
    user_role = request.headers.get('X-User-Role', 'admin')

    conn = get_conn()
    cur = conn.cursor()
    
    # Get pole info before deleting
    cur.execute(f"SELECT poles_code, location FROM {table} WHERE id = %s", (id,))
    pole_info = cur.fetchone()
    
    cur.execute(f"DELETE FROM {table} WHERE id = %s RETURNING id;", (id,))
    deleted = cur.fetchone()

    conn.commit()
    
    # 📝 LOG ACTIVITY
    if deleted and pole_info:
        log_activity(
            user_id,
            username,
            user_role,
            'delete_pole',
            f'{username} deleted {poles.upper()} pole ID {id}: {pole_info[0]} at {pole_info[1]}',
            request.remote_addr
        )
    
    cur.close()
    conn.close()

    if not deleted:
        return jsonify({"error": "Not found"}), 404

    return jsonify({"success": True})

    
# ===========================================================
# ODP PORTS (WITH LOGGING)
# ===========================================================

@app.route("/api/odp/<int:odp_id>/ports", methods=["GET"])
def get_odp_ports(odp_id):
    """Ambil 16 port untuk satu ODP"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, port_number, ip_address, customer_name, owner, status, notes, updated_at
        FROM odp_ports
        WHERE odp_id = %s
        ORDER BY port_number;
    """, (odp_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    ports = [{
        "id": r[0],
        "port_number": r[1],
        "ip_address": r[2],
        "customer_name": r[3],
        "owner": r[4],
        "status": r[5],
        "notes": r[6],
        "updated_at": str(r[7]) if r[7] else None
    } for r in rows]

    return jsonify(ports)


@app.route("/api/odp/<int:odp_id>/ports", methods=["POST"])
def create_or_update_odp_port(odp_id):
    """Insert / Update satu port"""
    data = request.get_json()
    
    # Get user info from request headers
    username = request.headers.get('X-User-Username', 'Unknown User')
    user_id = request.headers.get('X-User-Id', 0)
    user_role = request.headers.get('X-User-Role', 'staff')
    
    port_number = data.get("port_number")
    ip_address = data.get("ip_address")
    customer_name = data.get("customer_name")
    owner = data.get("owner")
    status = data.get("status", "used")
    notes = data.get("notes")

    if not (1 <= int(port_number) <= 16):
        return jsonify({"error": "Invalid port number"}), 400

    conn = get_conn()
    cur = conn.cursor()
    
    # Check if port exists to determine action type
    cur.execute("SELECT id FROM odp_ports WHERE odp_id = %s AND port_number = %s", (odp_id, port_number))
    existing_port = cur.fetchone()
    action_type = 'update_port' if existing_port else 'create_port'
    
    cur.execute("""
        INSERT INTO odp_ports (odp_id, port_number, ip_address, customer_name, owner, status, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (odp_id, port_number)
        DO UPDATE SET
            ip_address = EXCLUDED.ip_address,
            customer_name = EXCLUDED.customer_name,
            owner = EXCLUDED.owner,
            status = EXCLUDED.status,
            notes = EXCLUDED.notes,
            updated_at = now()
        RETURNING id;
    """, (odp_id, port_number, ip_address, customer_name, owner, status, notes))
    new_id = cur.fetchone()[0]
    conn.commit()
    
    # 📝 LOG ACTIVITY
    log_activity(
        user_id,
        username,
        user_role,
        action_type,
        f'{username} {"updated" if action_type == "update_port" else "created"} ODP port {port_number} for ODP ID {odp_id}: {customer_name}',
        request.remote_addr
    )
    
    cur.close()
    conn.close()
    return jsonify({"success": True, "id": new_id})


@app.route("/api/odp_ports/<int:port_id>", methods=["DELETE"])
def delete_odp_port(port_id):
    """Hapus port tertentu"""
    # Get user info from request headers
    username = request.headers.get('X-User-Username', 'Unknown User')
    user_id = request.headers.get('X-User-Id', 0)
    user_role = request.headers.get('X-User-Role', 'admin')
    
    conn = get_conn()
    cur = conn.cursor()
    
    # Get port info before deleting
    cur.execute("SELECT odp_id, port_number, customer_name FROM odp_ports WHERE id = %s", (port_id,))
    port_info = cur.fetchone()
    
    cur.execute("DELETE FROM odp_ports WHERE id = %s RETURNING id;", (port_id,))
    deleted = cur.fetchone()
    conn.commit()
    
    # 📝 LOG ACTIVITY
    if deleted and port_info:
        log_activity(
            user_id,
            username,
            user_role,
            'delete_port',
            f'{username} deleted port {port_info[1]} from ODP ID {port_info[0]}: {port_info[2]}',
            request.remote_addr
        )
    
    cur.close()
    conn.close()

    if not deleted:
        return jsonify({"error": "Port not found"}), 404
    return jsonify({"success": True})


# ===========================================================
# 📊 GET ACTIVITY LOGS (NEW ENDPOINT)
# ===========================================================

@app.route("/api/admin/activity-logs", methods=["GET"])
def get_activity_logs():
    """Get all activity logs for admin dashboard"""
    try:
        # Optional: pagination
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        conn = get_conn()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT id, user_id, username, role, action, description, ip_address, created_at
            FROM activity_logs
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """, (limit, offset))
        
        rows = cur.fetchall()
        
        # Get total count
        cur.execute("SELECT COUNT(*) FROM activity_logs")
        total_count = cur.fetchone()[0]
        
        cur.close()
        conn.close()

        result = [{
            "id": r[0],
            "user_id": r[1],
            "username": r[2],
            "role": r[3],
            "action": r[4],
            "description": r[5],
            "ip_address": r[6],
            "created_at": str(r[7])
        } for r in rows]

        return jsonify({
            "logs": result,
            "total": total_count,
            "limit": limit,
            "offset": offset
        }), 200

    except Exception as e:
        print("[ERROR] get_activity_logs:", e)
        return jsonify({"error": str(e)}), 500


# ===========================================================
# 🌦️ PREDIKSI CUACA
# ===========================================================

@app.route("/api/weather/current")
def weather_current():
    lat = request.args.get("lat", "-6.200000")
    lon = request.args.get("lon", "106.816666")
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather?"
            f"lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric&lang=id"
        )
        res = requests.get(url)
        res.raise_for_status()
        return jsonify(res.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/weather/forecast")
def weather_forecast():
    lat = request.args.get("lat", "-6.200000")
    lon = request.args.get("lon", "106.816666")
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/forecast?"
            f"lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric&lang=id"
        )
        res = requests.get(url)
        res.raise_for_status()
        return jsonify(res.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===========================
#  MODEL ML LSTM UNTUK PREDIKSI CUACA
# ===========================
def load_dataset():
    ml = get_ml_modules()
    pd = ml['pd']
    
    df = pd.read_csv(DATASET_PATH, delimiter=';')

    if "temp_rata-rata" in df.columns:
        df.rename(columns={"temp_rata-rata": "temperature"}, inplace=True)
    else:
        raise ValueError(f"Kolom suhu tidak ditemukan. Kolom tersedia: {list(df.columns)}")

    return df


@app.route("/api/weather/train", methods=["POST"])
def train_model():
    ml = get_ml_modules()
    pd = ml['pd']
    MinMaxScaler = ml['MinMaxScaler']
    Sequential = ml['Sequential']
    LSTM = ml['LSTM']
    Dense = ml['Dense']
    
    df = load_dataset()
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df = df.dropna(subset=["temperature"])

    data = df["temperature"].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    look_back = 24
    for i in range(len(data_scaled) - look_back):
        X.append(data_scaled[i:i+look_back])
        y.append(data_scaled[i+look_back])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=20, batch_size=16, verbose=1)

    model.save(r"C:\Users\gl65\Desktop\Project1\backend\models\model_cuaca.keras")
    
    # Reset cache agar model baru dimuat ulang
    global _weather_model, _weather_scaler
    _weather_model = None
    _weather_scaler = None
    
    return jsonify({"message": "Model berhasil dilatih ulang."})


@app.route("/api/weather/predict", methods=["GET"])
def predict_weather():
    steps = int(request.args.get("steps", 1))
    look_back = int(request.args.get("look_back", 24))

    try:
        model, scaler = get_weather_model()
        
        if model is None:
            return jsonify({"error": "Model belum dilatih. Jalankan /api/weather/train dulu."}), 400

        df = load_dataset()
        ml = get_ml_modules()
        pd = ml['pd']
        
        df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
        df = df.dropna(subset=["temperature"])
        data = df["temperature"].values.reshape(-1, 1)

        data_scaled = scaler.transform(data)
        last_seq = data_scaled[-look_back:]
        preds = []
        current_seq = last_seq.copy()

        for _ in range(steps):
            pred = model.predict(current_seq.reshape(1, look_back, 1), verbose=0)
            preds.append(pred[0][0])
            current_seq = np.append(current_seq[1:], pred)[-look_back:]

        preds_rescaled = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
        preds_rescaled = np.nan_to_num(preds_rescaled, nan=0.0)

        return jsonify({"predictions": preds_rescaled.flatten().tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===========================================================
# 🚀 RUN SERVER
# ===========================================================

if __name__ == "__main__":
       port = int(os.environ.get("PORT", 5000))
       app.run(host="0.0.0.0", port=port, debug=False)