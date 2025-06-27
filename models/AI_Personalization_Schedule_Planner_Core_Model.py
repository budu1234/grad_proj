import pandas as pd
from datetime import datetime, timedelta
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pulp import * # PuLP for linear programming optimization

# --- 1. Data Structures (No Change) ---

class User:
    """Represents a user with their scheduling preferences."""
    def __init__(self, user_id, preferred_working_hours, breaks,working_hours=False):
        self.user_id = user_id
        # preferred_working_hours: List of tuples (day_of_week, start_hour, end_hour)
        # e.g., [(0, 9, 17), (1, 10, 18)] for Monday 9-5, Tuesday 10-6
        self.preferred_working_hours = preferred_working_hours
        self.working_hours_constraint=working_hours
        # breaks: List of tuples (day_of_week, start_hour, end_hour)
        self.breaks = breaks
    def check_working_constraints(self):
        return self.working_hours_constraint

class Task:
    """Represents a single task with its attributes."""
    def __init__(self, task_id, name, deadline, importance, difficulty, checklist=None):
        self.task_id = task_id
        self.name = name
        self.deadline = deadline # datetime object
        self.importance = importance # 'High', 'Medium', 'Low'
        self.difficulty = difficulty # 'High', 'Medium', 'Low'
        # Estimated duration based on difficulty (can be refined with ML in a real app)
        self.estimated_duration_hours = self._get_estimated_duration()
        self.checklist = checklist if checklist is not None else []  # List of dicts: [{"item": "Do research", "done": False}, ...]
    
    def check_working_constraints(self):
        return self.working_hours_constraint


    def _get_estimated_duration(self):
        """Maps difficulty to a default estimated duration in hours."""
        if self.difficulty == 'Low':
            return 1.0 # 1 hour
        elif self.difficulty == 'Medium':
            return 3.0 # 3 hours
        elif self.difficulty == 'High':
            return 6.0 # 6 hours
        return 0.0 # Default

# --- 2. Feature Engineering (No Change) ---

def generate_features(task, user, current_slot_start, current_slot_end):
    """
    Generates features for a (task, time slot) pair to feed into the ML model.
    """
    features = {
        'task_id': task.task_id,
        'user_id': user.user_id,
        'task_name': task.name,
        'current_slot_start_hour': current_slot_start.hour,
        'current_slot_day_of_week': current_slot_start.weekday(), # Monday is 0, Sunday is 6
        'slot_duration_hours': (current_slot_end - current_slot_start).total_seconds() / 3600,
        'time_until_deadline_hours': (task.deadline - current_slot_start).total_seconds() / 3600,
        'task_importance': task.importance,
        'task_difficulty': task.difficulty,
        'estimated_task_duration_hours': task.estimated_duration_hours,
    }

    # Check if the slot is within user's preferred working hours
    is_preferred = False
    for day_of_week, start_hour, end_hour in user.preferred_working_hours:
        if (current_slot_start.weekday() == day_of_week and
            current_slot_start.hour >= start_hour and
            current_slot_end.hour <= end_hour):
            is_preferred = True
            break
    features['is_preferred_working_hour'] = int(is_preferred)

    # Check if the slot overlaps with breaks
    overlaps_break = False
    for day_of_week, break_start_hour, break_end_hour in user.breaks:
        if current_slot_start.weekday() == day_of_week:
            break_start = datetime(current_slot_start.year, current_slot_start.month,
                                   current_slot_start.day, break_start_hour, 0)
            break_end = datetime(current_slot_start.year, current_slot_start.month,
                                 current_slot_start.day, break_end_hour, 0)
            if not (current_slot_end <= break_start or current_slot_start >= break_end):
                overlaps_break = True
                break
    features['overlaps_break'] = int(overlaps_break)

    return features

def train_suitability_model_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    numerical_features = ['time_until_deadline_hours', 'slot_duration_hours',
                          'estimated_task_duration_hours', 'is_preferred_working_hour',
                          'overlaps_break']
    categorical_features = ['task_importance', 'task_difficulty']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', LinearRegression())])

    X = df[numerical_features + categorical_features]
    y = df['suitability_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_pipeline.fit(X_train, y_train)
    print("Suitability model trained on CSV data.")
    print("Train R^2:", model_pipeline.score(X_train, y_train))
    print("Test R^2:", model_pipeline.score(X_test, y_test))
    joblib.dump(model_pipeline, 'suitability_model_checkpoint.pkl')

    return model_pipeline

# --- 3. Suitability Scoring Model (Placeholder - No Change) ---

def train_dummy_suitability_model():
    """
    Trains a simple Linear Regression model for suitability scoring.
    In a real application, this would be trained on real user data.
    The 'suitability_score' would be a target variable representing how good a slot is.
    (e.g., 1 for accepted, 0 for rejected, or based on task completion metrics).
    """
    importance_map = {'Low': 1, 'Medium': 2, 'High': 3}
    difficulty_map = {'Low': 1, 'Medium': 2, 'High': 3}

    data = [
        # Good examples (high score)
        {'time_until_deadline_hours': 72, 'slot_duration_hours': 3, 'estimated_task_duration_hours': 3, 'is_preferred_working_hour': 1, 'overlaps_break': 0, 'task_importance': 3, 'task_difficulty': 2, 'suitability_score': 0.9},
        {'time_until_deadline_hours': 24, 'slot_duration_hours': 2, 'estimated_task_duration_hours': 1, 'is_preferred_working_hour': 1, 'overlaps_break': 0, 'task_importance': 3, 'task_difficulty': 1, 'suitability_score': 0.95},
        {'time_until_deadline_hours': 168, 'slot_duration_hours': 6, 'estimated_task_duration_hours': 6, 'is_preferred_working_hour': 1, 'overlaps_break': 0, 'task_importance': 2, 'task_difficulty': 3, 'suitability_score': 0.8},
        {'time_until_deadline_hours': 48, 'slot_duration_hours': 1.5, 'estimated_task_duration_hours': 1, 'is_preferred_working_hour': 1, 'overlaps_break': 0, 'task_importance': 2, 'task_difficulty': 1, 'suitability_score': 0.85},
        # Bad examples (low score)
        {'time_until_deadline_hours': 10, 'slot_duration_hours': 0.5, 'estimated_task_duration_hours': 3, 'is_preferred_working_hour': 0, 'overlaps_break': 1, 'task_importance': 3, 'task_difficulty': 2, 'suitability_score': 0.1},
        {'time_until_deadline_hours': 5, 'slot_duration_hours': 6, 'estimated_task_duration_hours': 1, 'is_preferred_working_hour': 0, 'overlaps_break': 0, 'task_importance': 1, 'task_difficulty': 1, 'suitability_score': 0.3},
        {'time_until_deadline_hours': 200, 'slot_duration_hours': 1, 'estimated_task_duration_hours': 6, 'is_preferred_working_hour': 1, 'overlaps_break': 0, 'task_importance': 3, 'task_difficulty': 3, 'suitability_score': 0.4}, # Too long for slot
        {'time_until_deadline_hours': 72, 'slot_duration_hours': 3, 'estimated_task_duration_hours': 3, 'is_preferred_working_hour': 0, 'overlaps_break': 0, 'task_importance': 1, 'task_difficulty': 1, 'suitability_score': 0.5},
    ]

    df = pd.DataFrame(data)

    numerical_features = ['time_until_deadline_hours', 'slot_duration_hours',
                          'estimated_task_duration_hours', 'is_preferred_working_hour',
                          'overlaps_break']
    categorical_features = ['task_importance', 'task_difficulty']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', LinearRegression())])

    X = df[numerical_features + categorical_features]
    y = df['suitability_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_pipeline.fit(X_train, y_train)
    print("Dummy suitability model trained.")
    return model_pipeline

suitability_model = joblib.load('suitability_model_checkpoint.pkl')

def get_suitability_score(features_df):
    """
    Predicts the suitability score for a given set of features using the trained model.
    """
    expected_cols = ['time_until_deadline_hours', 'slot_duration_hours',
                     'estimated_task_duration_hours', 'is_preferred_working_hour',
                     'overlaps_break', 'task_importance', 'task_difficulty']

    full_df = pd.DataFrame(columns=expected_cols)
    new_row = features_df.iloc[0].to_dict()

    importance_rev_map = {1: 'Low', 2: 'Medium', 3: 'High'}
    difficulty_rev_map = {1: 'Low', 2: 'Medium', 3: 'High'}

    if 'task_importance' in new_row and isinstance(new_row['task_importance'], int):
        new_row['task_importance'] = importance_rev_map.get(new_row['task_importance'], new_row['task_importance'])
    if 'task_difficulty' in new_row and isinstance(new_row['task_difficulty'], int):
        new_row['task_difficulty'] = difficulty_rev_map.get(new_row['task_difficulty'], new_row['task_difficulty'])

    temp_df = pd.DataFrame([new_row])

    for col in expected_cols:
        if col not in temp_df.columns:
            temp_df[col] = None

    temp_df = temp_df[expected_cols]

    score = suitability_model.predict(temp_df)[0]
    return max(0.0, min(1.0, score)) # Ensure score is between 0 and 1

# --- 4. Optimization Layer (MODIFIED FOR MULTI-TASKING WITH CAPACITY) ---

def schedule_tasks(user, pending_tasks, available_time_slots):
    """
    Schedules tasks using linear programming to maximize suitability scores.
    Allows multiple tasks in the same slot as long as their combined duration fits.
    Schedules as many tasks as possible and identifies unscheduled ones.

    Args:
        user (User): The user object.
        pending_tasks (list[Task]): List of tasks to be scheduled.
        available_time_slots (list[tuple[datetime, datetime]]): List of
            (start_time, end_time) tuples for available time blocks.

    Returns:
        tuple: A tuple containing two elements:
            - dict: A dictionary mapping (slot_start, slot_end) tuple to a list of Task objects assigned to it.
            - list: A list of Task objects that could not be scheduled.
    """
    prob = LpProblem("Task_Scheduling_Multi_Tasking_Capacity_Problem", LpMaximize)

    # Decision variables: x[(task_id, slot_index)] = 1 if task is assigned to this slot, 0 otherwise
    task_slot_vars = LpVariable.dicts("Assign",
                                      ((t.task_id, i) for t in pending_tasks for i in range(len(available_time_slots))),
                                      0, 1, LpBinary)

    # Store suitability scores for easy access
    scores = {}
    for task in pending_tasks:
        for i, (slot_start, slot_end) in enumerate(available_time_slots):
            # Tasks cannot be assigned if the slot ends after their deadline.
            if slot_end > task.deadline:
                scores[(task.task_id, i)] = -1000000 # Strong penalty for assigning past deadline
            else:
                features = generate_features(task, user, slot_start, slot_end)
                features_df = pd.DataFrame([features])
                score = get_suitability_score(features_df)
                scores[(task.task_id, i)] = score
            

    # Objective: Maximize the total suitability score of assigned tasks
    # Re-checked and corrected: Ensure the generator expression is properly enclosed for lpSum.
    prob += lpSum([scores[(task.task_id, i)] * task_slot_vars[(task.task_id, i)]
                   for task in pending_tasks for i in range(len(available_time_slots))]), "Total Suitability Score"

    # Constraints:

    # 1. EACH TASK IS ASSIGNED TO AT MOST ONE TIME SLOT (Allows tasks to be unscheduled)
    for task in pending_tasks:
        prob += lpSum(task_slot_vars[(task.task_id, i)] for i in range(len(available_time_slots))) <= 1, \
                f"Task_{task.task_id}_One_Slot_Max"

    # 2. SLOT CAPACITY CONSTRAINT (Crucial for multi-tasking)
    # The sum of estimated durations of all tasks assigned to a slot must not exceed the slot's duration.
    for i, (slot_start, slot_end) in enumerate(available_time_slots):
        slot_duration = (slot_end - slot_start).total_seconds() / 3600
        prob += lpSum(task.estimated_duration_hours * task_slot_vars[(task.task_id, i)] for task in pending_tasks) <= slot_duration, \
                f"Slot_{i}_Capacity"

    # 3. DO NOT SCHEDULE TASKS IN BREAK TIMES (No Change)
    for task in pending_tasks:
        for i, (slot_start, slot_end) in enumerate(available_time_slots):
            slot_features = generate_features(task, user, slot_start, slot_end)
            if slot_features['overlaps_break'] == 1:
                prob += task_slot_vars[(task.task_id, i)] == 0, \
                        f"Task_{task.task_id}_Slot_{i}_NoBreakOverlap"
    if user.check_working_constraints():           
        for task in pending_tasks:
            for i, (slot_start, slot_end) in enumerate(available_time_slots):
                slot_features = generate_features(task, user, slot_start, slot_end)
                
                if slot_features['is_preferred_working_hour'] == 0:
                    # HARD CONSTRAINT: Do not allow task to be assigned to non-preferred hours
                    prob += task_slot_vars[(task.task_id, i)] == 0, \
                            f"Task_{task.task_id}_Slot_{i}_NoNonPreferred"

    # Solve the problem
    prob.solve(PULP_CBC_CMD(msg=0))

    # Extract results
    scheduled_tasks_by_slot = {i: [] for i in range(len(available_time_slots))}
    assigned_task_ids = set()

    if LpStatus[prob.status] == "Optimal":
        for task in pending_tasks:
            for i, (slot_start, slot_end) in enumerate(available_time_slots):
                if task_slot_vars[(task.task_id, i)].varValue > 0.9: # If scheduled
                    scheduled_tasks_by_slot[i].append(task)
                    assigned_task_ids.add(task.task_id)
                    # Note: No 'break' here, as a task is assigned to only one slot (Task_One_Slot_Max ensures this)
                    # but multiple tasks can go into the same slot.

        # Prepare the final schedule output, mapping (start, end) tuples to lists of tasks
        final_schedule_output = {}
        for i, (slot_start, slot_end) in enumerate(available_time_slots):
            if scheduled_tasks_by_slot[i]: # Only include slots that actually have tasks
                final_schedule_output[(slot_start, slot_end)] = scheduled_tasks_by_slot[i]

        unscheduled_tasks = [task for task in pending_tasks if task.task_id not in assigned_task_ids]

        return final_schedule_output, unscheduled_tasks
    else:
        # If solver status is not 'Optimal', it means it couldn't find ANY solution.
        # This is unlikely with the flexible 'Task_One_Slot_Max' constraint, but for robustness.
        print(f"Solver did not find an optimal solution. Status: {LpStatus[prob.status]}. All tasks considered unscheduled.")
        return {}, pending_tasks

# --- Helper Function to Generate Contiguous Free Time Blocks (No Change) ---

def generate_free_time_blocks(user, start_datetime, end_datetime, min_slot_duration_minutes=30, use_preferred_only=True):
    """
    Generates contiguous blocks of free time.
    If use_preferred_only is True, only preferred working hours are considered.
    If False, all hours are considered (except breaks).
    """
    free_blocks = []
    current_day = start_datetime.date()

    while current_day <= end_datetime.date():
        day_of_week = current_day.weekday()

        if use_preferred_only:
            # Only preferred working hours
            daily_working_intervals = []
            for pref_day, start_h, end_h in user.preferred_working_hours:
                if pref_day == day_of_week:
                    daily_working_intervals.append((
                        datetime(current_day.year, current_day.month, current_day.day, start_h, 0),
                        datetime(current_day.year, current_day.month, current_day.day, end_h, 0)
                    ))
        else:
            # All hours in the day (midnight to midnight)
            daily_working_intervals = [(
                datetime(current_day.year, current_day.month, current_day.day, 0, 0),
                datetime(current_day.year, current_day.month, current_day.day, 23, 59)
            )]

        daily_break_intervals = []
        for break_day, start_h, end_h in user.breaks:
            if break_day == day_of_week:
                daily_break_intervals.append((
                    datetime(current_day.year, current_day.month, current_day.day, start_h, 0),
                    datetime(current_day.year, current_day.month, current_day.day, end_h, 0)
                ))

        all_intervals = sorted(daily_working_intervals + daily_break_intervals)

        for work_start, work_end in daily_working_intervals:
            current_free_start = max(work_start, start_datetime)

            for break_start, break_end in daily_break_intervals:
                if not (break_end <= current_free_start or break_start >= work_end):
                    if current_free_start < break_start:
                        block_end = min(work_end, break_start)
                        if (block_end - current_free_start).total_seconds() / 60 >= min_slot_duration_minutes:
                            free_blocks.append((current_free_start, block_end))
                    current_free_start = max(current_free_start, break_end)

            if (work_end - current_free_start).total_seconds() / 60 >= min_slot_duration_minutes:
                block_end = min(work_end, end_datetime)
                if (block_end - current_free_start).total_seconds() / 60 >= min_slot_duration_minutes:
                    free_blocks.append((current_free_start, block_end))

        current_day += timedelta(days=1)

    if not free_blocks:
        return []

    free_blocks.sort(key=lambda x: x[0])
    merged_blocks = []
    current_merge_start, current_merge_end = free_blocks[0]

    for next_start, next_end in free_blocks[1:]:
        if next_start <= current_merge_end:
            current_merge_end = max(current_merge_end, next_end)
        else:
            merged_blocks.append((current_merge_start, current_merge_end))
            current_merge_start, current_merge_end = next_start, next_end
    merged_blocks.append((current_merge_start, current_merge_end))

    final_blocks = [(s, e) for s, e in merged_blocks if e > start_datetime and (e - s).total_seconds()/60 >= min_slot_duration_minutes]

    return final_blocks


# --- Example Usage ---

if __name__ == "__main__":
    current_user = User(
        user_id="user_123",
        preferred_working_hours=[
            (0, 9, 19), # Monday 9 AM - 7 PM
            (1, 9, 19), # Tuesday 9 AM - 7 PM
            (2, 10, 17), # Wednesday 10 AM - 5 PM
            (3, 8, 17), # Thursday 8 AM - 5 PM
            (4, 9, 17)  # Friday 9 AM - 5 PM
        ],
        breaks=[
            (0, 19, 21), # Monday 12 PM - 1 PM lunch
            (1, 12, 13),
            (2, 12, 13),
            (3, 12, 13),
            (4, 12, 13)
        ]
    )

    now = datetime(2025, 6, 16, 8, 12, 31) # Monday, June 16, 2025, 8:12 AM

    tasks_to_schedule = [
        Task(task_id="T001", name="Complete Project Proposal",
             deadline=now + timedelta(days=2, hours=10), # Due Wed 6/18 ~6 PM, 6 hrs estimated
             importance="High", difficulty="High"),
        Task(task_id="T002", name="Review Team Reports",
             deadline=now + timedelta(days=1, hours=5), # Due Tue 6/17 ~1 PM, 3 hrs estimated
             importance="High", difficulty="Medium"),
        Task(task_id="T003", name="Email Clients",
             deadline=now + timedelta(days=5, hours=8), # Due Mon 6/21 ~4 PM, 1 hr estimated
             importance="Medium", difficulty="Low"),
        Task(task_id="T004", name="Learn new Python feature",
             deadline=now + timedelta(days=4), # Due Fri 6/20 ~8 AM, 3 hrs estimated
             importance="Low", difficulty="Medium"),
        Task(task_id="T005", name="Prepare marketing materials",
             deadline=now + timedelta(days=2), # Due Wed 6/18 ~8 AM, 6 hrs estimated
             importance="High", difficulty="High"),
         Task(task_id="T006", name="Organize Files",
             deadline=now + timedelta(days=1, hours=10), # Due Tue 6/17 ~6 PM, 1 hr estimated
             importance="Low", difficulty="Low"),
         Task(task_id="T007", name="Client Call Prep",
             deadline=now + timedelta(days=3, hours=6), # Due Mon 6/19 ~2 PM, 1 hr estimated
             importance="High", difficulty="Low"),
         Task(task_id="T008", name="Quick Research", # Small task
             deadline=now + timedelta(days=1), # Due Tue 6/17 ~8 AM, 0.5 hrs estimated
             importance="Low", difficulty="Low"),
    ]
    # Adjust estimated duration for T008 to be shorter (0.5 hours) to encourage multi-tasking in 1-hour slots
    for task in tasks_to_schedule:
        if task.task_id == "T008":
            task.estimated_duration_hours = 0.5


    future_days_to_consider = 7
    end_consideration_datetime = now + timedelta(days=future_days_to_consider)

    # Use a smaller min_slot_duration_minutes to allow for more granular multi-tasking
    available_time_blocks = generate_free_time_blocks(current_user, now, end_consideration_datetime, min_slot_duration_minutes=30)

    print(f"Total available free time blocks considered: {len(available_time_blocks)}")
    print("Available Blocks (Start, End, Duration):")
    for s, e in available_time_blocks:
        duration = (e - s).total_seconds() / 3600
        print(f"  {s.strftime('%Y-%m-%d %H:%M')} to {e.strftime('%Y-%m-%d %H:%M')} ({duration:.1f} hours)")

    print("\nAttempting to schedule tasks (multi-tasking with capacity limits)...")

    scheduled_slots_data, unscheduled_tasks = schedule_tasks(current_user, tasks_to_schedule, available_time_blocks)

    if scheduled_slots_data:
        print("\n--- Here's your recommended multi-tasking schedule ---")

        # Sort slots by start time for clear display
        sorted_slots = sorted(scheduled_slots_data.keys(), key=lambda x: x[0])

        for (slot_start, slot_end) in sorted_slots:
            tasks_in_slot = scheduled_slots_data[(slot_start, slot_end)]
            slot_duration = (slot_end - slot_start).total_seconds() / 3600

            # Calculate total estimated duration for tasks assigned to this slot
            total_task_duration_in_slot = sum(t.estimated_duration_hours for t in tasks_in_slot)

            print(f"\nTime Slot: {slot_start.strftime('%Y-%m-%d %H:%M')} to {slot_end.strftime('%Y-%m-%d %H:%M')} (Duration: {slot_duration:.1f} hours)")
            print(f"  Total Task Effort in Slot: {total_task_duration_in_slot:.1f} hours (Capacity Used: {(total_task_duration_in_slot / slot_duration * 100):.1f}%)")
            print("  Assigned Tasks:")
            for task in tasks_in_slot:
                print(f"    - {task.name} (ID: {task.task_id}) - {task.estimated_duration_hours:.1f} hrs, Imp: {task.importance}, Diff: {task.difficulty}")
            print("-" * 50)
    else:
        print("\nNo tasks could be scheduled at this time based on your constraints and availability.")

    if unscheduled_tasks:
        print("\n--- The following tasks could not be scheduled ---")
        for task in unscheduled_tasks:
            print(f"Task: {task.name} (ID: {task.task_id})")
            print(f"  Deadline: {task.deadline.strftime('%Y-%m-%d %H:%M')}, Importance: {task.importance}, Difficulty: {task.difficulty}")
            print(f"  Estimated Duration: {task.estimated_duration_hours} hours")
            print("  Possible reasons: No suitable slot before deadline, or total task effort exceeds remaining slot capacity.")
            print("-" * 30)


    # Example of generating features for a single task-slot combination (no change)
    print("\n--- Example Feature Generation & Scoring for T001 in an arbitrary slot ---")
    example_task = tasks_to_schedule[0] # T001
    example_slot_start = now + timedelta(hours=2) # 2 hours from now
    example_slot_end = example_slot_start + timedelta(hours=example_task.estimated_duration_hours)

    if example_slot_end <= example_task.deadline:
        example_features = generate_features(example_task, current_user, example_slot_start, example_slot_end)
        print("Generated Features for T001 in a hypothetical slot:")
        for k, v in example_features.items():
            print(f"  {k}: {v}")

        features_for_scoring = pd.DataFrame([example_features])
        importance_map_rev = {'High': 3, 'Medium': 2, 'Low': 1}
        difficulty_map_rev = {'High': 3, 'Medium': 2, 'Low': 1}

        features_for_scoring['task_importance'] = features_for_scoring['task_importance'].map(importance_map_rev)
        features_for_scoring['task_difficulty'] = features_for_scoring['task_difficulty'].map(difficulty_map_rev)

        example_score = get_suitability_score(features_for_scoring)
        print(f"\nSuitability Score for T001 in this slot: {example_score:.2f}")
    else:
        print("Example slot chosen is after the deadline for T001, or not long enough for the task.")