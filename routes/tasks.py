from flask import Blueprint, jsonify, request
from db import get_connection
from flask_jwt_extended import jwt_required, get_jwt_identity

tasks_bp = Blueprint('tasks', __name__)

def is_overlapping_with_breaks(user_id, day_of_week, start_hour, end_hour):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM breaks WHERE user_id = %s AND day_of_week = %s AND NOT (end_hour <= %s OR start_hour >= %s)",
        (user_id, day_of_week, start_hour, end_hour)
    )
    overlap = cursor.fetchone()
    cursor.close()
    conn.close()
    return overlap is not None

@tasks_bp.route("/", methods=["GET"])
@jwt_required()
def get_tasks():
    user_id = get_jwt_identity()
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM tasks WHERE user_id = %s", (user_id,))
    tasks = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(tasks)

@tasks_bp.route("/<int:task_id>", methods=["PATCH"])
@jwt_required()
def update_task(task_id):
    user_id = get_jwt_identity()
    data = request.get_json()
    fields = []
    values = []
    time_fields = ["day_of_week", "start_hour", "end_hour"]
    time_updates = {field: data.get(field) for field in time_fields if field in data}

    # If any time fields are being updated, fetch the current values for missing ones
    if time_updates:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT day_of_week, start_hour, end_hour FROM tasks WHERE id = %s AND user_id = %s", (task_id, user_id))
        current = cursor.fetchone()
        cursor.close()
        conn.close()
        if not current:
            return jsonify({"error": "Task not found"}), 404
        # Use updated values if provided, else current
        day_of_week = time_updates.get("day_of_week", current["day_of_week"])
        start_hour = time_updates.get("start_hour", current["start_hour"])
        end_hour = time_updates.get("end_hour", current["end_hour"])
        # Check for overlap
        if is_overlapping_with_breaks(user_id, day_of_week, start_hour, end_hour):
            return jsonify({"error": "Task time overlaps with a break"}), 400

    for field in ["name", "deadline", "importance", "difficulty", "status", "is_checked", "day_of_week", "start_hour", "end_hour"]:
        if field in data:
            fields.append(f"{field} = %s")
            values.append(data[field])
    if not fields:
        return jsonify({"error": "No fields to update"}), 400
    values.append(task_id)
    values.append(user_id)
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        f"UPDATE tasks SET {', '.join(fields)} WHERE id = %s AND user_id = %s",
        tuple(values)
    )
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"message": "Task updated"})

@tasks_bp.route("/<int:task_id>", methods=["DELETE"])
@jwt_required()
def delete_task(task_id):
    user_id = get_jwt_identity()
    conn = get_connection()
    cursor = conn.cursor()
    # First, delete from schedules to avoid foreign key constraint errors
    cursor.execute("DELETE FROM schedules WHERE task_id = %s AND user_id = %s", (task_id, user_id))
    # Then, delete the task itself
    cursor.execute("DELETE FROM tasks WHERE id = %s AND user_id = %s", (task_id, user_id))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"message": "Task deleted"})

@tasks_bp.route("/", methods=["POST"])
@jwt_required()
def create_task():
    user_id = get_jwt_identity()
    data = request.get_json()
    name = data.get("name")
    deadline = data.get("deadline")
    importance = data.get("importance")
    difficulty = data.get("difficulty")
    status = data.get("status", "pending")
    is_checked = data.get("is_checked", False)
    day_of_week = data.get("day_of_week")
    start_hour = data.get("start_hour")
    end_hour = data.get("end_hour")

    if not all([name, deadline, importance, difficulty, day_of_week, start_hour, end_hour]):
        return jsonify({"error": "Missing required fields"}), 400
    
    if is_overlapping_with_breaks(user_id, day_of_week, start_hour, end_hour):
        return jsonify({"error": "Task time overlaps with a break"}), 400

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO tasks (user_id, name, deadline, importance, difficulty, status, is_checked, day_of_week, start_hour, end_hour) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (user_id, name, deadline, importance, difficulty, status, is_checked, day_of_week, start_hour, end_hour)
    )
    conn.commit()
    task_id = cursor.lastrowid
    cursor.close()
    conn.close()

    # --- Reschedule all tasks for this user ---
    from routes.planner import recommend_reschedule_for_user
    recommend_reschedule_for_user(user_id)

    return jsonify({"message": "Task created and schedule updated", "task_id": task_id}), 201