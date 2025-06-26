from flask import Blueprint, jsonify, request
from db import get_connection
from flask_jwt_extended import jwt_required, get_jwt_identity

breaks_bp = Blueprint('breaks', __name__)

@breaks_bp.route("/", methods=["GET"])
@jwt_required()
def get_breaks():
    user = get_jwt_identity()
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, day_of_week, start_hour, end_hour FROM breaks WHERE user_id = %s", (user['id'],))
    breaks = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(breaks)

@breaks_bp.route("/", methods=["POST"])
@jwt_required()
def create_break():
    user = get_jwt_identity()
    data = request.get_json()
    day_of_week = data.get("day_of_week")
    start_hour = data.get("start_hour")
    end_hour = data.get("end_hour")
    if day_of_week is None or start_hour is None or end_hour is None:
        return jsonify({"error": "Missing required fields"}), 400

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO breaks (user_id, day_of_week, start_hour, end_hour) VALUES (%s, %s, %s, %s)",
        (user['id'], day_of_week, start_hour, end_hour)
    )
    conn.commit()
    break_id = cursor.lastrowid
    cursor.close()
    conn.close()
    return jsonify({"message": "Break created", "break_id": break_id}), 201

@breaks_bp.route("/<int:break_id>", methods=["PATCH"])
@jwt_required()
def update_break(break_id):
    user = get_jwt_identity()
    data = request.get_json()
    fields = []
    values = []
    for field in ["day_of_week", "start_hour", "end_hour"]:
        if field in data:
            fields.append(f"{field} = %s")
            values.append(data[field])
    if not fields:
        return jsonify({"error": "No fields to update"}), 400
    values.append(break_id)
    values.append(user['id'])
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        f"UPDATE breaks SET {', '.join(fields)} WHERE id = %s AND user_id = %s",
        tuple(values)
    )
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"message": "Break updated"})

@breaks_bp.route("/<int:break_id>", methods=["DELETE"])
@jwt_required()
def delete_break(break_id):
    user = get_jwt_identity()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM breaks WHERE id = %s AND user_id = %s", (break_id, user['id']))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"message": "Break deleted"})