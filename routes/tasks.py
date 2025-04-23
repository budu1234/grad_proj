from flask import Blueprint, jsonify, request
from db import get_connection

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
