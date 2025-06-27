from flask import Blueprint, jsonify, request
from db import get_connection
from flask_jwt_extended import jwt_required, get_jwt_identity

tasks_bp = Blueprint('tasks', __name__)

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
    for field in ["name", "deadline", "importance", "difficulty", "status", "is_checked"]:
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

    if not all([name, deadline, importance, difficulty]):
        return jsonify({"error": "Missing required fields"}), 400

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO tasks (user_id, name, deadline, importance, difficulty, status, is_checked) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (user_id, name, deadline, importance, difficulty, status, is_checked)
    )
    conn.commit()
    task_id = cursor.lastrowid
    cursor.close()
    conn.close()

    # --- Reschedule all tasks for this user ---
    from routes.planner import recommend_reschedule_for_user
    recommend_reschedule_for_user(user_id)

    return jsonify({"message": "Task created and schedule updated", "task_id": task_id}), 201