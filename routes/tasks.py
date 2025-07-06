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
    from datetime import datetime

    def iso_to_mysql_datetime(iso_str):
        if not isinstance(iso_str, str):
            return iso_str
        if iso_str.endswith('Z'):
            iso_str = iso_str[:-1]
        iso_str = iso_str.replace('T', ' ')
        if '.' in iso_str:
            iso_str = iso_str.split('.')[0]
        return iso_str

    user_id = get_jwt_identity()
    data = request.get_json()
    fields = []
    values = []
    time_fields = ["day_of_week", "start_hour", "end_hour"]

    # Only check for overlap with breaks if ALL time fields are being updated
    if all(field in data for field in time_fields):
        day_of_week = data["day_of_week"]
        start_hour = data["start_hour"]
        end_hour = data["end_hour"]
        if is_overlapping_with_breaks(user_id, day_of_week, start_hour, end_hour):
            return jsonify({"error": "Task time overlaps with a break"}), 400

    for field in ["name", "deadline", "importance", "difficulty", "status", "is_checked", "day_of_week", "start_hour", "end_hour"]:
        if field in data:
            value = data[field]
            if field == "deadline" and value is not None:
                value = iso_to_mysql_datetime(value)
            fields.append(f"{field} = %s")
            values.append(value)
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

    # --- Reschedule all tasks for this user ---
    from routes.planner import recommend_reschedule_for_user
    recommend_reschedule_for_user(user_id)

    return jsonify({"message": "Task updated and schedule refreshed"})

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

    # --- Reschedule all tasks for this user ---
    from routes.planner import recommend_reschedule_for_user
    recommend_reschedule_for_user(user_id)

    return jsonify({"message": "Task deleted and schedule refreshed"})

@tasks_bp.route("/", methods=["POST"])
@jwt_required()
def create_task():
    from datetime import datetime

    def iso_to_mysql_datetime(iso_str):
        # Handle 'Z' (UTC) and milliseconds
        if iso_str.endswith('Z'):
            iso_str = iso_str[:-1]
        iso_str = iso_str.replace('T', ' ')
        if '.' in iso_str:
            iso_str = iso_str.split('.')[0]
        return iso_str

    user_id = get_jwt_identity()
    data = request.get_json()
    name = data.get("name")
    deadline = data.get("deadline")
    importance = data.get("importance")
    difficulty = data.get("difficulty")
    status = data.get("status", "pending")
    is_checked = data.get("is_checked", False)

    # If you ever allow direct scheduling (with day_of_week), check for break overlap
    day_of_week = data.get("day_of_week")
    start_hour = data.get("start_hour")
    end_hour = data.get("end_hour")
    if day_of_week is not None and start_hour is not None and end_hour is not None:
        if is_overlapping_with_breaks(user_id, day_of_week, start_hour, end_hour):
            return jsonify({"error": "Task time overlaps with a break"}), 400

    # Do NOT require day_of_week, start_hour, end_hour
    if not all([name, deadline, importance, difficulty]):
        return jsonify({"error": "Missing required fields"}), 400

    # Convert deadline to MySQL DATETIME format
    deadline_mysql = iso_to_mysql_datetime(deadline)

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO tasks (user_id, name, deadline, importance, difficulty, status, is_checked) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (user_id, name, deadline_mysql, importance, difficulty, status, is_checked)
        )
        task_id = cursor.lastrowid

        # Try to generate schedule
        from routes.planner import generate_schedule_for_user
        schedule = generate_schedule_for_user(user_id)

        conn.commit()
        return jsonify(schedule), 201
    except Exception as e:
        conn.rollback()
        return jsonify({"error": "Task not saved due to scheduling error", "details": str(e)}), 500
    finally:
        cursor.close()
        conn.close()