from flask import Blueprint, jsonify
from db import get_connection
from flask import Flask

users_bp = Blueprint('users', __name__)

@users_bp.route("/", methods=["GET"])
def get_users():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM Users")
    users = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(users)

