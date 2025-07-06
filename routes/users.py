from flask import Blueprint, request, jsonify, current_app
import os
from db import get_connection
import json
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from flask_cors import CORS
import requests
import random

users_bp = Blueprint('users', __name__)
CORS(users_bp)

@users_bp.route("/", methods=["GET"])
@jwt_required()
def get_users():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    # Removed role from SELECT
    cursor.execute("SELECT id, username, email, created_at FROM users")
    users = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(users)

@users_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')
    phone_number = data.get('phone_number')

    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400

    password_hash = generate_password_hash(password)
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        # Removed role from INSERT
        cursor.execute(
            "INSERT INTO users (username, password_hash, email, phone_number) VALUES (%s, %s, %s, %s)",
            (username, password_hash, email, phone_number)
        )
        conn.commit()
        # Fetch the new user
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        if user:
            access_token = create_access_token(identity=str(user['id']))
            return jsonify({'message': 'User registered successfully', 'access_token': access_token}), 201
        else:
            return jsonify({'error': 'User registration failed'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    finally:
        cursor.close()
        conn.close()
        conn.close()

@users_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    # Allow login with username OR email
    cursor.execute("SELECT * FROM users WHERE username = %s OR email = %s", (username, username))
    user = cursor.fetchone()
    cursor.close()
    conn.close()

    if user and check_password_hash(user['password_hash'], password):
        access_token = create_access_token(identity=str(user['id']))
        return jsonify({'access_token': access_token}), 200
    else:
        return jsonify({'error': 'Invalid username or password'}), 401

@users_bp.route("/preferences", methods=["PATCH"])
@jwt_required()
def update_preferences():
    user_id = get_jwt_identity()
    data = request.get_json()
    preferred_working_hours = data.get("preferred_working_hours")
    working_hours_constraint = data.get("working_hours_constraint")
    buffer_hours = data.get("buffer_hours", 4)  # Default to 4 if not provided

    # General working hours
    general_start_hour = data.get("general_start_hour", 8)
    general_end_hour = data.get("general_end_hour", 22)

    # Questionnaire fields
    age = data.get("age")
    gender = data.get("gender")
    major = data.get("major")
    preferred_study_hours = data.get("preferred_study_hours")
    default_break_start = data.get("break_start")
    default_break_end = data.get("break_end")

    # Validation: general hours must be greater than or equal to all preferred hours
    gen_end = general_end_hour if general_end_hour != 0 else 24

    if preferred_working_hours is not None:
        for entry in preferred_working_hours:
            day, start, end = entry
            pref_end = end if end != 0 else 24
            if general_start_hour > start or gen_end < pref_end:
                return jsonify({
                    "error": "General working hours must be greater than or equal to all preferred working hours for all days."
                }), 400

    # Calculate days off (not stored, just for logic)
    days_off = []
    if preferred_working_hours is not None:
        all_days = set(range(7))
        working_days = set([entry[0] for entry in preferred_working_hours])
        days_off = list(all_days - working_days)
    else:
        preferred_working_hours = [[d, 9, 17] for d in range(7)]
        days_off = []

    if working_hours_constraint is None:
        working_hours_constraint = False

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
    user_row = cursor.fetchone()
    if user_row is None:
        cursor.close()
        conn.close()
        return jsonify({"error": "User not found"}), 404

    cursor.close()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE users SET
            preferred_working_hours=%s,
            working_hours_constraint=%s,
            buffer_hours=%s,
            general_start_hour=%s,
            general_end_hour=%s,
            age=%s,
            gender=%s,
            major=%s,
            preferred_study_hours=%s,
            default_break_start=%s,
            default_break_end=%s
        WHERE id=%s
        """,
        (
            json.dumps(preferred_working_hours),
            working_hours_constraint,
            buffer_hours,
            general_start_hour,
            general_end_hour,
            age,
            gender,
            major,
            json.dumps(preferred_study_hours) if preferred_study_hours else None,
            default_break_start,
            default_break_end,
            user_id,
        ),
    )

    # --- Create or update breaks if break_start and break_end are provided ---
    def parse_time_string_to_hour(time_str):
        if not time_str:
            return None
        time_part, am_pm = time_str.split(' ')
        hour, minute = map(int, time_part.split(':'))
        if am_pm == 'PM' and hour != 12:
            hour += 12
        if am_pm == 'AM' and hour == 12:
            hour = 0
        return hour

    if default_break_start and default_break_end:
        start_hour = parse_time_string_to_hour(default_break_start)
        end_hour = parse_time_string_to_hour(default_break_end)
        for day in range(7):  # 0=Sunday, 6=Saturday
            cursor.execute(
                "SELECT id FROM breaks WHERE user_id=%s AND day_of_week=%s",
                (user_id, day)
            )
            existing = cursor.fetchone()
            if existing:
                cursor.execute(
                    "UPDATE breaks SET start_hour=%s, end_hour=%s WHERE id=%s",
                    (start_hour, end_hour, existing[0])
                )
            else:
                cursor.execute(
                    "INSERT INTO breaks (user_id, day_of_week, start_hour, end_hour) VALUES (%s, %s, %s, %s)",
                    (user_id, day, start_hour, end_hour)
                )
    # --- END BREAK LOGIC ---

    conn.commit()
    cursor.close()
    conn.close()

    from routes.planner import recommend_reschedule_for_user
    recommend_reschedule_for_user(user_id)

    return jsonify({"message": "Preferences updated"})

@users_bp.route("/login/google", methods=["POST"])
def google_login():
    data = request.get_json()
    id_token = data.get("id_token")
    if not id_token:
        return jsonify({"error": "Missing id_token"}), 400

    # Verify token with Google
    google_resp = requests.get(
        f"https://oauth2.googleapis.com/tokeninfo?id_token={id_token}"
    )
    if google_resp.status_code != 200:
        return jsonify({"error": "Invalid Google token"}), 401

    google_data = google_resp.json()
    email = google_data.get("email")
    if not email:
        return jsonify({"error": "No email in Google token"}), 400

    # Check if user exists, else create
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    if not user:
        # Register new user with Google email
        cursor = conn.cursor()
        # Removed role from INSERT
        cursor.execute(
            "INSERT INTO users (username, email) VALUES (%s, %s)",
            (email, email)
        )
        conn.commit()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
    cursor.close()
    conn.close()

    access_token = create_access_token(identity=str(user['id']))
    return jsonify({'access_token': access_token}), 200

@users_bp.route("/login/facebook", methods=["POST"])
def facebook_login():
    data = request.get_json()
    access_token = data.get("access_token")
    if not access_token:
        return jsonify({"error": "Missing access_token"}), 400

    # Verify token with Facebook
    fb_resp = requests.get(
        f"https://graph.facebook.com/me?fields=id,name,email&access_token={access_token}"
    )
    if fb_resp.status_code != 200:
        return jsonify({"error": "Invalid Facebook token"}), 401

    fb_data = fb_resp.json()
    email = fb_data.get("email")
    name = fb_data.get("name")
    if not email:
        return jsonify({"error": "No email in Facebook profile"}), 400

    # Check if user exists, else create
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    if not user:
        cursor = conn.cursor()
        # Removed role from INSERT
        cursor.execute(
            "INSERT INTO users (username, email) VALUES (%s, %s)",
            (name or email, email)
        )
        conn.commit()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
    cursor.close()
    conn.close()

    access_token_jwt = create_access_token(identity=str(user['id']))
    return jsonify({'access_token': access_token_jwt}), 200

confirmation_codes = {}  # For demo only; use DB/Redis in production

@users_bp.route("/send_confirmation_code", methods=["POST"])
def send_confirmation_code():
    data = request.get_json()
    email = data.get("email")
    if not email:
        return jsonify({"error": "Email required"}), 400

    code = str(random.randint(1000, 9999))
    confirmation_codes[email] = code

    # TODO: Send code via email/SMS here (for demo, just print)
    print(f"Confirmation code for {email}: {code}")

    return jsonify({"message": "Code sent"}), 200

@users_bp.route("/verify_confirmation_code", methods=["POST"])
def verify_confirmation_code():
    data = request.get_json()
    email = data.get("email")
    code = data.get("code")
    if not email or not code:
        return jsonify({"error": "Email and code required"}), 400

    if confirmation_codes.get(email) == code:
        del confirmation_codes[email]
        return jsonify({"message": "Code verified"}), 200
    else:
        return jsonify({"error": "Invalid code"}), 400

@users_bp.route("/me", methods=["GET"])
@jwt_required()
def get_me():
    user_id = get_jwt_identity()
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    # Removed role from SELECT
    cursor.execute(
        "SELECT id, username, email, phone_number, profile_picture, preferred_working_hours, working_hours_constraint, general_start_hour, general_end_hour FROM users WHERE id = %s",
        (user_id,)
    )
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    if user:
        return jsonify(user), 200
    else:
        return jsonify({"error": "User not found"}), 404

@users_bp.route("/reset_password", methods=["POST"])
def reset_password():
    data = request.get_json()
    email = data.get("email")
    code = data.get("code")
    new_password = data.get("new_password")
    if not email or not code or not new_password:
        return jsonify({"error": "Email, code, and new password required"}), 400

    # Check code
    if confirmation_codes.get(email) != code:
        return jsonify({"error": "Invalid code"}), 400

    # Update password
    conn = get_connection()
    cursor = conn.cursor()
    password_hash = generate_password_hash(new_password)
    cursor.execute("UPDATE users SET password_hash=%s WHERE email=%s", (password_hash, email))
    conn.commit()
    cursor.close()
    conn.close()
    del confirmation_codes[email]
    return jsonify({"message": "Password reset successful"}), 200

@users_bp.route("/upload_profile_picture", methods=["POST"])
@jwt_required()
def upload_profile_picture():
    user_id = get_jwt_identity()
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    import time
    filename = f"user_{user_id}_{int(time.time())}_{file.filename}"
    save_path = os.path.join(current_app.root_path, "static", "profile_pics")
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, filename)
    file.save(file_path)

    # Save filename in DB
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET profile_picture=%s WHERE id=%s", (filename, user_id))
    conn.commit()
    cursor.close()
    conn.close()

    url = f"/static/profile_pics/{filename}"
    return jsonify({"profile_picture_url": url}), 200

def parse_time_string_to_hour(time_str):
    if not time_str:
        return None
    time_part, am_pm = time_str.split(' ')
    hour, minute = map(int, time_part.split(':'))
    if am_pm == 'PM' and hour != 12:
        hour += 12
    if am_pm == 'AM' and hour == 12:
        hour = 0
    return hour

@users_bp.route("/update_name", methods=["PUT"])
@jwt_required()
def update_name():
    user_id = get_jwt_identity()
    data = request.get_json()
    name = data.get("name")
    if not name:
        return jsonify({"error": "Name is required"}), 400

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET username=%s WHERE id=%s", (name, user_id))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"message": "Name updated successfully"}), 200