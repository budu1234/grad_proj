from db import get_connection
from flask_jwt_extended import jwt_required, get_jwt_identity
from flask import Blueprint, request, jsonify
from models.AI_Personalization_Schedule_Planner_Core_Model import User, Task, generate_free_time_blocks, schedule_tasks
from datetime import datetime, timedelta
import json

planner_bp = Blueprint('planner', __name__)

@planner_bp.route("/recommend", methods=["POST"])
@jwt_required()
def recommend():
    try:
        # Get user_id from JWT
        user_id = get_jwt_identity()

        # Optionally get user preferences from request (or fetch from DB if you store them)
        user_data = request.get_json() or {}
        preferred_working_hours = user_data.get("preferred_working_hours", [(0,9,17), (1,9,17)])
        breaks = user_data.get("breaks", [(0,12,13), (1,12,13)])
        working_hours_constraint = user_data.get("working_hours_constraint", False)

        user = User(user_id, preferred_working_hours, breaks, working_hours_constraint)

        # Fetch tasks from DB for this user
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, name, deadline, importance, difficulty FROM tasks WHERE user_id = %s", (user_id,))
        task_rows = cursor.fetchall()
        cursor.close()
        conn.close()

        # Convert DB rows to Task objects
        tasks = []
        for t in task_rows:
            deadline = t["deadline"]
            if isinstance(deadline, str):
                deadline = datetime.fromisoformat(deadline)
            tasks.append(Task(
                task_id=t["id"],
                name=t["name"],
                deadline=deadline,
                importance=t["importance"],
                difficulty=t["difficulty"]
            ))

        now = datetime.now()
        available_blocks = generate_free_time_blocks(user, now, now + timedelta(days=7))
        scheduled, unscheduled = schedule_tasks(user, tasks, available_blocks)

        # --- Save the generated schedule to the schedules table ---
        conn = get_connection()
        cursor = conn.cursor()
        for slot, task_list in scheduled.items():
            slot_start = slot[0]
            slot_end = slot[1]
            for task in task_list:
                cursor.execute(
                    "INSERT INTO schedules (user_id, task_id, slot_start, slot_end) VALUES (%s, %s, %s, %s)",
                    (user_id, task.task_id, slot_start, slot_end)
                )
        conn.commit()
        cursor.close()
        conn.close()
        # ---------------------------------------------------------

        def serialize_task(task):
            return {
                "task_id": task.task_id,
                "name": task.name,
                "deadline": task.deadline.isoformat(),
                "importance": task.importance,
                "difficulty": task.difficulty,
                "estimated_duration_hours": getattr(task, "estimated_duration_hours", None)
            }

        schedule_json = [
            {
                "slot_start": str(slot[0]),
                "slot_end": str(slot[1]),
                "tasks": [serialize_task(task) for task in task_list]
            }
            for slot, task_list in scheduled.items()
        ]
        unscheduled_json = [serialize_task(task) for task in unscheduled]

        return jsonify({
            "scheduled": schedule_json,
            "unscheduled": unscheduled_json
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------ Helper function for rescheduling ------------------

def recommend_reschedule_for_user(user_id):
    from models.AI_Personalization_Schedule_Planner_Core_Model import User, Task, generate_free_time_blocks, schedule_tasks
    from datetime import datetime, timedelta
    import json

    # Fetch user preferences
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT preferred_working_hours, working_hours_constraint FROM users WHERE id = %s", (user_id,))
    user_row = cursor.fetchone()
    cursor.close()
    conn.close()

    preferred_working_hours = json.loads(user_row["preferred_working_hours"])
    working_hours_constraint = user_row["working_hours_constraint"]

    # Fetch breaks from breaks table
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT day_of_week, start_hour, end_hour FROM breaks WHERE user_id = %s", (user_id,))
    breaks_rows = cursor.fetchall()
    cursor.close()
    conn.close()
    breaks = [(row["day_of_week"], row["start_hour"], row["end_hour"]) for row in breaks_rows]

    user = User(user_id, preferred_working_hours, breaks, working_hours_constraint)

    # Fetch all tasks for this user
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, name, deadline, importance, difficulty FROM tasks WHERE user_id = %s", (user_id,))
    task_rows = cursor.fetchall()
    cursor.close()
    conn.close()

    tasks = []
    for t in task_rows:
        deadline = t["deadline"]
        if isinstance(deadline, str):
            deadline = datetime.fromisoformat(deadline)
        tasks.append(Task(
            task_id=t["id"],
            name=t["name"],
            deadline=deadline,
            importance=t["importance"],
            difficulty=t["difficulty"]
        ))

    now = datetime.now()
    available_blocks = generate_free_time_blocks(user, now, now + timedelta(days=7))
    scheduled, _ = schedule_tasks(user, tasks, available_blocks)

    # Clear old schedule
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM schedules WHERE user_id = %s", (user_id,))
    # Save new schedule
    for slot, task_list in scheduled.items():
        slot_start = slot[0]
        slot_end = slot[1]
        for task in task_list:
            cursor.execute(
                "INSERT INTO schedules (user_id, task_id, slot_start, slot_end) VALUES (%s, %s, %s, %s)",
                (user_id, task.task_id, slot_start, slot_end)
            )
    conn.commit()
    cursor.close()
    conn.close()