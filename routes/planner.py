from flask import Blueprint, request, jsonify
from models.AI_Personalization_Schedule_Planner_Core_Model import User, Task, generate_free_time_blocks, schedule_tasks
from datetime import datetime, timedelta

planner_bp = Blueprint('planner', __name__)

@planner_bp.route("/recommend", methods=["POST"])
def recommend():
    user_data = request.get_json()
    try:
        # Parse user info
        user_id = user_data["user_id"]
        preferred_working_hours = user_data.get("preferred_working_hours", [(0,9,17), (1,9,17)])  # Example default
        breaks = user_data.get("breaks", [(0,12,13), (1,12,13)])  # Example default
        working_hours_constraint = user_data.get("working_hours_constraint", False)

        user = User(user_id, preferred_working_hours, breaks, working_hours_constraint)

        # Parse tasks
        tasks = []
        for t in user_data.get("tasks", []):
            deadline = datetime.fromisoformat(t["deadline"])
            tasks.append(Task(
                task_id=t["task_id"],
                name=t["name"],
                deadline=deadline,
                importance=t["importance"],
                difficulty=t["difficulty"]
            ))

        now = datetime.now()
        available_blocks = generate_free_time_blocks(user, now, now + timedelta(days=7))
        scheduled, unscheduled = schedule_tasks(user, tasks, available_blocks)

        # Format output for JSON
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