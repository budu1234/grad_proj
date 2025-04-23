from flask import Blueprint, jsonify
from db import get_connection

schedule_bp = Blueprint('schedule', __name__)

@schedule_bp.route("/", methods=["GET"])
def get_schedule():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM Schedule")
    schedule = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(schedule)
