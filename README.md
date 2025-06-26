# AI Schedule Planner

## Overview
AI Schedule Planner is a project designed to help users manage their schedules using artificial intelligence. This application connects to a MySQL database to store and retrieve scheduling information.

## Project Structure
```
ai_schedule_planner/
├── src/
│   └── config.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup Instructions

### Prerequisites
- Python 3.x
- Docker
- MySQL Database

### Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd ai_schedule_planner
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Configure the database settings in `src/config.py`:
   - Update the `DB_CONFIG` dictionary with your database credentials.

### Running the Application
To run the application using Docker, build the Docker image and run the container:
1. Build the Docker image:
   ```
   docker build -t ai_schedule_planner .
   ```

2. Run the Docker container:
   ```
   docker run -d -p 5000:5000 ai_schedule_planner
   ```

### Usage
Once the application is running, you can access it at `http://localhost:5000`. Follow the on-screen instructions to manage your schedule.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License.