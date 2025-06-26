from flask import Blueprint, jsonify, request
from db import get_connection
from flask_jwt_extended import jwt_required, get_jwt_identity

settings_bp = Blueprint('settings', __name__)

@settings_bp.route("/", methods=["GET"])
@jwt_required()
def get_settings():
    user = get_jwt_identity()
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, key_name, value FROM settings WHERE user_id = %s", (user['id'],))
    settings = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(settings)

@settings_bp.route("/<key_name>", methods=["GET"])
@jwt_required()
def get_setting(key_name):
    user = get_jwt_identity()
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, key_name, value FROM settings WHERE user_id = %s AND key_name = %s", (user['id'], key_name))
    setting = cursor.fetchone()
    cursor.close()
    conn.close()
    if not setting:
        return jsonify({"error": "Setting not found"}), 404
    return jsonify(setting)

@settings_bp.route("/", methods=["POST"])
@jwt_required()
def set_setting():
    user = get_jwt_identity()
    data = request.get_json()
    key_name = data.get("key_name")
    value = data.get("value")
    if not key_name:
        return jsonify({"error": "Missing key_name"}), 400

    conn = get_connection()
    cursor = conn.cursor()
    # Upsert: update if exists, else insert
    cursor.execute(
        "SELECT id FROM settings WHERE user_id = %s AND key_name = %s",
        (user['id'], key_name)
    )
    existing = cursor.fetchone()
    if existing:
        cursor.execute(
            "UPDATE settings SET value = %s WHERE id = %s",
            (value, existing[0])
        )
    else:
        cursor.execute(
            "INSERT INTO settings (user_id, key_name, value) VALUES (%s, %s, %s)",
            (user['id'], key_name, value)
        )
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"message": "Setting saved"})

@settings_bp.route("/<key_name>", methods=["DELETE"])
@jwt_required()
def delete_setting(key_name):
    user = get_jwt_identity()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM settings WHERE user_id = %s AND key_name = %s", (user['id'], key_name))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"message": "Setting deleted"})