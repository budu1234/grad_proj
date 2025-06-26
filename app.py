from flask import Flask
from routes.users import users_bp
from routes.tasks import tasks_bp
from routes.schedule import schedule_bp
from routes.planner import planner_bp
import joblib

app = Flask(__name__)

app.suitability_model = joblib.load('suitability_model_checkpoint.pkl')

# Register blueprints
app.register_blueprint(users_bp, url_prefix='/users')
app.register_blueprint(tasks_bp, url_prefix='/tasks')
app.register_blueprint(schedule_bp, url_prefix='/schedule')
app.register_blueprint(planner_bp, url_prefix='/planner')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
