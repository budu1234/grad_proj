from flask import Blueprint, jsonify, request
from db import get_connection
from routes.planner import recommend_reschedule_for_user  
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime, timezone

schedule_bp = Blueprint('schedule', __name__)

@schedule_bp.route("/", methods=["GET"])
@jwt_required()
def get_schedule():
    user_id = get_jwt_identity()
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    # Get scheduled tasks
    cursor.execute("""
        SELECT 
            s.id as schedule_id,
            s.slot_start,
            s.slot_end,
            t.id as task_id,
            t.name as task_name,
            t.importance,
            t.difficulty,
            t.deadline,
            t.status,
            t.is_checked
        FROM schedules s
        JOIN tasks t ON s.task_id = t.id
        WHERE s.user_id = %s
            AND t.status != 'completed'
        ORDER BY s.slot_start
    """, (user_id,))
    scheduled = cursor.fetchall()

    # Mark overdue scheduled tasks
    now = datetime.now(timezone.utc)
    for task in scheduled:
        deadline = task.get("deadline")
        deadline_dt = None
        if isinstance(deadline, str):
            try:
                deadline_dt = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
            except Exception:
                deadline_dt = None
        elif isinstance(deadline, datetime):
            deadline_dt = deadline

        # Make deadline_dt timezone-aware if it's naive
        if deadline_dt is not None and deadline_dt.tzinfo is None:
            deadline_dt = deadline_dt.replace(tzinfo=timezone.utc)

        is_overdue = (
            task["status"] != "completed"
            and deadline_dt is not None
            and deadline_dt < now
        )
        task["overdue"] = bool(is_overdue)

    # Get unscheduled tasks (tasks not in schedules)
    cursor.execute("""
        SELECT 
            t.id as task_id,
            t.name as task_name,
            t.importance,
            t.difficulty,
            t.deadline,
            t.status,
            t.is_checked
        FROM tasks t
        WHERE t.user_id = %s
        AND t.status = 'pending'
        AND t.id NOT IN (SELECT task_id FROM schedules WHERE user_id = %s)
    """, (user_id, user_id))
    unscheduled = cursor.fetchall()
    cursor.close()
    conn.close()

    # Mark overdue unscheduled tasks
    for task in unscheduled:
        deadline = task.get("deadline")
        deadline_dt = None
        if isinstance(deadline, str):
            try:
                deadline_dt = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
            except Exception:
                deadline_dt = None
        elif isinstance(deadline, datetime):
            deadline_dt = deadline

        if deadline_dt is not None and deadline_dt.tzinfo is None:
            deadline_dt = deadline_dt.replace(tzinfo=timezone.utc)

        is_overdue = (
            task["status"] != "completed"
            and deadline_dt is not None
            and deadline_dt < now
        )
        task["overdue"] = bool(is_overdue)

        # Only set reason if NOT overdue
        if not is_overdue:
            task["reason"] = "No available slot before deadline or due to constraints"
        else:
            task["reason"] = ""  # Or remove this line to omit the field

    return jsonify({
        "scheduled": scheduled,
        "unscheduled": unscheduled
    }), 200

@schedule_bp.route("/", methods=["POST"])
@jwt_required()
def create_schedule():
    user_id = get_jwt_identity()
    data = request.get_json()
    task_id = data.get("task_id")
    slot_start = data.get("slot_start")
    slot_end = data.get("slot_end")

    if not all([task_id, slot_start, slot_end]):
        return jsonify({"error": "Missing required fields"}), 400

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO schedules (user_id, task_id, slot_start, slot_end) VALUES (%s, %s, %s, %s)",
        (user_id, task_id, slot_start, slot_end)
    )
    conn.commit()
    schedule_id = cursor.lastrowid
    cursor.close()
    conn.close()
    return jsonify({"message": "Schedule created", "schedule_id": schedule_id}), 201

@schedule_bp.route("/<int:schedule_id>", methods=["PATCH"])
@jwt_required()
def update_schedule(schedule_id):
    user_id = get_jwt_identity()
    data = request.get_json()
    fields = []
    values = []
    for field in ["task_id", "slot_start", "slot_end"]:
        if field in data:
            fields.append(f"{field} = %s")
            values.append(data[field])
    if not fields:
        return jsonify({"error": "No fields to update"}), 400
    values.append(schedule_id)
    values.append(user_id)
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        f"UPDATE schedules SET {', '.join(fields)} WHERE id = %s AND user_id = %s",
        tuple(values)
    )
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"message": "Schedule updated"})

@schedule_bp.route("/<int:schedule_id>", methods=["DELETE"])
@jwt_required()
def delete_schedule(schedule_id):   
    user_id = get_jwt_identity()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM schedules WHERE id = %s AND user_id = %s", (schedule_id, user_id))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"message": "Schedule deleted"})

@schedule_bp.route("/reschedule", methods=["POST"])
@jwt_required()
def reschedule_all():
    user_id = get_jwt_identity()
    try:
        recommend_reschedule_for_user(user_id)
        return jsonify({"message": "Rescheduling triggered"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@schedule_bp.route("/completed", methods=["GET"])
@jwt_required()
def get_completed_tasks():
    user_id = get_jwt_identity()
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT 
            s.id as schedule_id,
            s.slot_start,
            s.slot_end,
            t.id as task_id,
            t.name as task_name,
            t.importance,
            t.difficulty,
            t.deadline,
            t.status,
            t.is_checked
        FROM schedules s
        JOIN tasks t ON s.task_id = t.id
        WHERE s.user_id = %s
          AND t.status = 'completed'
        ORDER BY s.slot_start
    """, (user_id,))
    completed = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify({"completed": completed}), 200