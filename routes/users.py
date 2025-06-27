from flask import Blueprint, request, jsonify
from db import get_connection
import json
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity

users_bp = Blueprint('users', __name__)

@users_bp.route("/", methods=["GET"])
@jwt_required()
def get_users():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, username, email, role, created_at FROM users")
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
    role = data.get('role', 'user')

    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400

    password_hash = generate_password_hash(password)
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash, email, role) VALUES (%s, %s, %s, %s)",
            (username, password_hash, email, role)
        )
        conn.commit()
        return jsonify({'message': 'User registered successfully'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    finally:
        cursor.close()
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
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()

    if user and check_password_hash(user['password_hash'], password):
        access_token = create_access_token(identity=str(user['id']),additional_claims={"role": user['role']})
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

    if preferred_working_hours is None:
        preferred_working_hours = [[d, 9, 17] for d in range(7)]
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
        "UPDATE users SET preferred_working_hours=%s, working_hours_constraint=%s, buffer_hours=%s WHERE id=%s",
        (json.dumps(preferred_working_hours), working_hours_constraint, buffer_hours, user_id)
    )
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"message": "Preferences updated"})