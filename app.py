from flask import Flask
from routes.users import users_bp
from routes.tasks import tasks_bp
from routes.schedule import schedule_bp
from routes.settings import settings_bp
from routes.planner import planner_bp
from routes.breaks import breaks_bp
from routes.notifications import notifications_bp
import joblib
from flask_jwt_extended import JWTManager

app = Flask(__name__)

app.suitability_model = joblib.load('suitability_model_checkpoint.pkl')
app.config['JWT_SECRET_KEY'] = 'your-secret-key'
jwt = JWTManager(app)

# Register blueprints
app.register_blueprint(users_bp, url_prefix='/users')
app.register_blueprint(tasks_bp, url_prefix='/tasks')
app.register_blueprint(notifications_bp, url_prefix='/notifications')
app.register_blueprint(breaks_bp, url_prefix='/breaks')
app.register_blueprint(settings_bp, url_prefix='/settings')
app.register_blueprint(schedule_bp, url_prefix='/schedule')
app.register_blueprint(planner_bp, url_prefix='/planner')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
