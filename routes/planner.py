from flask import Blueprint, request, jsonify
from models.rl_agent import recommend_schedule  # Your RL model's function

planner_bp = Blueprint('planner', __name__)

@planner_bp.route("/recommend", methods=["POST"])
def recommend():
    user_data = request.get_json()

    try:
        # Example expected input: {"user_id": 123}
        user_id = user_data["user_id"]
        result = recommend_schedule(user_id)
        return jsonify({"recommended_schedule": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
