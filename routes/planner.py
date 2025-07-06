from db import get_connection
from flask_jwt_extended import jwt_required, get_jwt_identity
from flask import Blueprint, jsonify
from models.AI_Personalization_Schedule_Planner_Core_Model import User, Task, generate_free_time_blocks, schedule_tasks
from datetime import datetime, timedelta, timezone
import json

planner_bp = Blueprint('planner', __name__)

def assign_times_within_slot(slot_start, slot_end, tasks, user_breaks):
    assigned = []
    current_time = slot_start
    breaks_today = [
        (datetime(slot_start.year, slot_start.month, slot_start.day, b_start, 0, tzinfo=timezone.utc),
         datetime(slot_start.year, slot_start.month, slot_start.day, b_end, 0, tzinfo=timezone.utc))
        for day, b_start, b_end in user_breaks if day == slot_start.weekday()
    ]
    breaks_today.sort()
    for task in tasks:
        duration = timedelta(hours=task.estimated_duration_hours)
        while True:
            next_break = next((b for b in breaks_today if b[0] < current_time + duration and b[1] > current_time), None)
            if next_break:
                if current_time < next_break[0]:
                    if next_break[0] - current_time >= duration:
                        assigned_start = current_time
                        assigned_end = current_time + duration
                        assigned.append((task, assigned_start, assigned_end))
                        current_time = assigned_end
                        break
                    else:
                        current_time = next_break[1]
                else:
                    current_time = next_break[1]
            else:
                if current_time + duration <= slot_end:
                    assigned_start = current_time
                    assigned_end = current_time + duration
                    assigned.append((task, assigned_start, assigned_end))
                    current_time = assigned_end
                    break
                else:
                    break
    return assigned

def generate_schedule_for_user(user_id):
    # Fetch user preferences
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT preferred_working_hours, working_hours_constraint, buffer_hours FROM users WHERE id = %s", (user_id,))
    user_row = cursor.fetchone()
    cursor.close()
    conn.close()

    if user_row is None:
        raise Exception("User not found")

    preferred_working_hours_raw = user_row["preferred_working_hours"]
    try:
        preferred_working_hours = json.loads(preferred_working_hours_raw)
        if not preferred_working_hours:
            preferred_working_hours = [[d, 9, 17] for d in range(7)]
    except Exception:
        preferred_working_hours = [[d, 9, 17] for d in range(7)]

    working_hours_constraint = user_row["working_hours_constraint"]
    if working_hours_constraint is None:
        working_hours_constraint = False

    buffer_hours = int(user_row.get("buffer_hours", 4) or 4)
    if buffer_hours is None:
        buffer_hours = 4

    # Fetch breaks
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT day_of_week, start_hour, end_hour FROM breaks WHERE user_id = %s", (user_id,))
    breaks_rows = cursor.fetchall()
    cursor.close()
    conn.close()
    breaks = [(row["day_of_week"], row["start_hour"], row["end_hour"]) for row in breaks_rows]

    user = User(user_id, preferred_working_hours, breaks, working_hours_constraint)

    # Fetch tasks
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, name, deadline, importance, difficulty, day_of_week, start_hour, end_hour FROM tasks WHERE user_id = %s", (user_id,))
    task_rows = cursor.fetchall()
    cursor.close()
    conn.close()

    tasks = []
    for t in task_rows:
        deadline = t["deadline"]
        if isinstance(deadline, str):
            deadline = datetime.strptime(deadline, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
        effective_deadline = deadline - timedelta(hours=buffer_hours)
        difficulty = t.get("difficulty") or "Medium"
        if difficulty not in ("Low", "Medium", "High"):
            difficulty = "Medium"
        importance = t.get("importance") or "Medium"
        tasks.append(Task(
            task_id=t["id"],
            name=t["name"],
            deadline=effective_deadline,
            importance=importance,
            difficulty=difficulty,
            day_of_week=t.get("day_of_week"),
            start_hour=t.get("start_hour"),
            end_hour=t.get("end_hour")
        ))

    now = datetime.now(timezone.utc)
    available_blocks = generate_free_time_blocks(
        user, now, now + timedelta(days=7),
        use_preferred_only=user.check_working_constraints()
    )

    for i in range(7):
        print(f"Available blocks for {['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][i]}:")
        for block in available_blocks:
            if block[0].weekday() == i:
                print(f"  Block: {block[0]} to {block[1]}", flush=True)

    # --- PRINT TASKS TO SCHEDULE ---
    print("Tasks to schedule:")
    for t in tasks:
        print(f"  Task: {t.name}, Deadline: {t.deadline}, Duration: {getattr(t, 'estimated_duration_hours', None)}, DayOfWeek: {t.day_of_week}, Start: {t.start_hour}, End: {t.end_hour}", flush=True)

    scheduled, unscheduled = schedule_tasks(user, tasks, available_blocks)

    # --- PRINT UNSCHEDULED TASKS ---
    print("Unscheduled tasks:")
    for t in unscheduled:
        print(f"  Unscheduled: {t.name}, Deadline: {t.deadline}, Duration: {getattr(t, 'estimated_duration_hours', None)}, Reason: No available slot", flush=True)

    # Save new schedule to DB
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM schedules WHERE user_id = %s", (user_id,))
    for slot, task_list in scheduled.items():
        slot_start = slot[0]
        slot_end = slot[1]
        assigned_times = assign_times_within_slot(slot_start, slot_end, task_list, breaks)
        for task, assigned_start, assigned_end in assigned_times:
            cursor.execute(
                "INSERT INTO schedules (user_id, task_id, slot_start, slot_end) VALUES (%s, %s, %s, %s)",
                (user_id, task.task_id, assigned_start, assigned_end)
            )
    conn.commit()
    cursor.close()
    conn.close()

    def serialize_task(task):
        return {
            "task_id": task.task_id,
            "name": task.name,
            "deadline": task.deadline.isoformat(),
            "importance": task.importance,
            "difficulty": task.difficulty,
            "estimated_duration_hours": getattr(task, "estimated_duration_hours", None)
        }

    schedule_json = []
    for slot, task_list in scheduled.items():
        slot_start = slot[0]
        slot_end = slot[1]
        assigned_times = assign_times_within_slot(slot_start, slot_end, task_list, breaks)
        for task, assigned_start, assigned_end in assigned_times:
            schedule_json.append({
                "slot_start": str(assigned_start),
                "slot_end": str(assigned_end),
                "tasks": [serialize_task(task)]
            })
    unscheduled_json = [serialize_task(task) for task in unscheduled]

    return {
        "scheduled": schedule_json,
        "unscheduled": unscheduled_json
    }

@planner_bp.route("/recommend", methods=["POST"])
@jwt_required()
def recommend():
    try:
        user_id = get_jwt_identity()
        schedule = generate_schedule_for_user(user_id)
        return jsonify(schedule)
    except Exception as e:
        import traceback
        print("Exception in /recommend:", str(e), flush=True)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def recommend_reschedule_for_user(user_id):
    # This just updates the DB, doesn't return JSON
    generate_schedule_for_user(user_id)