from flask import Blueprint, jsonify, request
from db import get_connection
from flask_jwt_extended import jwt_required, get_jwt_identity

notifications_bp = Blueprint('notifications', __name__)

@notifications_bp.route("/", methods=["GET"])
@jwt_required()
def get_notifications():
    user = get_jwt_identity()
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, message, is_read, created_at FROM notifications WHERE user_id = %s ORDER BY created_at DESC", (user['id'],))
    notifications = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(notifications)

@notifications_bp.route("/<int:notification_id>/read", methods=["PATCH"])
@jwt_required()
def mark_notification_read(notification_id):
    user = get_jwt_identity()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE notifications SET is_read = TRUE WHERE id = %s AND user_id = %s",
        (notification_id, user['id'])
    )
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"message": "Notification marked as read"})

@notifications_bp.route("/", methods=["POST"])
@jwt_required()
def create_notification():
    user = get_jwt_identity()
    data = request.get_json()
    message = data.get("message")
    if not message:
        return jsonify({"error": "Missing message"}), 400

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO notifications (user_id, message) VALUES (%s, %s)",
        (user['id'], message)
    )
    conn.commit()
    notification_id = cursor.lastrowid
    cursor.close()
    conn.close()
    return jsonify({"message": "Notification created", "notification_id": notification_id}), 201

@notifications_bp.route("/<int:notification_id>", methods=["DELETE"])
@jwt_required()
def delete_notification(notification_id):
    user = get_jwt_identity()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM notifications WHERE id = %s AND user_id = %s", (notification_id, user['id']))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"message": "Notification deleted"})