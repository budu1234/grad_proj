from flask import Blueprint, jsonify, request
from db import get_connection
from models.AI_Personalization_Schedule_Planner_Core_Model import Task

tasks_bp = Blueprint('tasks', __name__)

@tasks_bp.route("/", methods=["GET"])
def get_tasks():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM Tasks")
    tasks = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(tasks)

def find_task_by_id(task_id):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM Tasks WHERE task_id = %s", (task_id,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    if not row:
        return None
    # If you store checklist as JSON in the DB, load it here
    checklist = row.get("checklist", [])
    if isinstance(checklist, str):
        import json
        checklist = json.loads(checklist)
    return Task(
        task_id=row["task_id"],
        name=row["name"],
        deadline=row["deadline"],
        importance=row["importance"],
        difficulty=row["difficulty"],
        checklist=checklist
    )

@tasks_bp.route('/<task_id>/checklist', methods=['POST'])
def add_checklist_item(task_id):
    data = request.get_json()
    item_text = data.get('item')
    # Find the task (implement your own task lookup)
    task = find_task_by_id(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    task.checklist.append({"item": item_text, "done": False})
    return jsonify({"message": "Checklist item added", "checklist": task.checklist})

# Example: Mark checklist item as done
@tasks_bp.route('/<task_id>/checklist/<int:item_index>', methods=['PATCH'])
def mark_checklist_item(task_id, item_index):
    data = request.get_json()
    done = data.get('done', True)
    task = find_task_by_id(task_id)
    if not task or item_index >= len(task.checklist):
        return jsonify({"error": "Task or checklist item not found"}), 404
    task.checklist[item_index]['done'] = done
    return jsonify({"message": "Checklist item updated", "checklist": task.checklist})
