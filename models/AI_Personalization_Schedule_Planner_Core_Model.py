import pandas as pd
from datetime import datetime, timedelta, timezone  
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
    def __init__(self, user_id, preferred_working_hours, breaks, working_hours=False,
                general_start_hour=8, general_end_hour=22, scheduling_preference_mode=None):
        self.user_id = user_id
        # preferred_working_hours: List of tuples (day_of_week, start_hour, end_hour)
        self.preferred_working_hours = preferred_working_hours
        self.working_hours_constraint = working_hours
        # breaks: List of tuples (day_of_week, start_hour, end_hour)
        self.breaks = breaks
        self.general_start_hour = general_start_hour
        self.general_end_hour = general_end_hour
        # New: scheduling_preference_mode: 'preferred_only' or 'flexible_hours'
        # If not provided, fallback to working_hours_constraint for backward compatibility
        self.scheduling_preference_mode = scheduling_preference_mode or (
            'preferred_only' if working_hours else 'flexible_hours'
        )
    def check_working_constraints(self):
        # For backward compatibility
        return self.working_hours_constraint

class Task:
    """Represents a single task with its attributes."""
    def __init__(self, task_id, name, deadline, importance, difficulty, checklist=None,
                 day_of_week=None, start_hour=None, end_hour=None):
        self.task_id = task_id
        self.name = name
        # --- Timezone fix: always make deadline UTC-aware ---
        if isinstance(deadline, str):
            try:
                self.deadline = datetime.strptime(deadline, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
            except Exception:
                self.deadline = datetime.fromisoformat(deadline).astimezone(timezone.utc)
        elif isinstance(deadline, datetime):
            if deadline.tzinfo is None:
                self.deadline = deadline.replace(tzinfo=timezone.utc)
            else:
                self.deadline = deadline.astimezone(timezone.utc)
        else:
            self.deadline = deadline
        # ----------------------------------------------------
        self.importance = importance # 'High', 'Medium', 'Low'
        self.difficulty = difficulty # 'High', 'Medium', 'Low'
        self.estimated_duration_hours = self._get_estimated_duration()
        self.checklist = checklist if checklist is not None else []
        self.day_of_week = day_of_week
        self.start_hour = start_hour
        self.end_hour = end_hour

    def _get_estimated_duration(self):
        if self.difficulty == 'Low':
            return 1.0
        elif self.difficulty == 'Medium':
            return 3.0
        elif self.difficulty == 'High':
            return 6.0
        return 0.0

# --- 2. Feature Engineering (No Change) ---

def generate_features(task, user, current_slot_start, current_slot_end):
    # --- Timezone fix: ensure UTC-aware ---
    if current_slot_start.tzinfo is None:
        current_slot_start = current_slot_start.replace(tzinfo=timezone.utc)
    if current_slot_end.tzinfo is None:
        current_slot_end = current_slot_end.replace(tzinfo=timezone.utc)
    # ------------------------------------------------
    features = {
        'task_id': task.task_id,
        'user_id': user.user_id,
        'task_name': task.name,
        'current_slot_start_hour': current_slot_start.hour,
        'current_slot_day_of_week': current_slot_start.weekday(),
        'slot_duration_hours': (current_slot_end - current_slot_start).total_seconds() / 3600,
        'time_until_deadline_hours': (task.deadline - current_slot_start).total_seconds() / 3600,
        'task_importance': task.importance,
        'task_difficulty': task.difficulty,
        'estimated_task_duration_hours': task.estimated_duration_hours,
    }
    is_preferred = False
    for day_of_week, start_hour, end_hour in user.preferred_working_hours:
        if (current_slot_start.weekday() == day_of_week and
            current_slot_start.hour >= start_hour and
            current_slot_end.hour <= end_hour):
            is_preferred = True
            break
    features['is_preferred_working_hour'] = int(is_preferred)
    overlaps_break = False
    for day_of_week, break_start_hour, break_end_hour in user.breaks:
        if current_slot_start.weekday() == day_of_week:
            break_start = datetime(current_slot_start.year, current_slot_start.month,
                                   current_slot_start.day, break_start_hour, 0, tzinfo=timezone.utc)
            break_end = datetime(current_slot_start.year, current_slot_start.month,
                                 current_slot_start.day, break_end_hour, 0, tzinfo=timezone.utc)
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

# --- 3. Suitability Scoring Model (No Change) ---

def train_dummy_suitability_model():
    importance_map = {'Low': 1, 'Medium': 2, 'High': 3}
    difficulty_map = {'Low': 1, 'Medium': 2, 'High': 3}
    data = [
        {'time_until_deadline_hours': 72, 'slot_duration_hours': 3, 'estimated_task_duration_hours': 3, 'is_preferred_working_hour': 1, 'overlaps_break': 0, 'task_importance': 3, 'task_difficulty': 2, 'suitability_score': 0.9},
        {'time_until_deadline_hours': 24, 'slot_duration_hours': 2, 'estimated_task_duration_hours': 1, 'is_preferred_working_hour': 1, 'overlaps_break': 0, 'task_importance': 3, 'task_difficulty': 1, 'suitability_score': 0.95},
        {'time_until_deadline_hours': 168, 'slot_duration_hours': 6, 'estimated_task_duration_hours': 6, 'is_preferred_working_hour': 1, 'overlaps_break': 0, 'task_importance': 2, 'task_difficulty': 3, 'suitability_score': 0.8},
        {'time_until_deadline_hours': 48, 'slot_duration_hours': 1.5, 'estimated_task_duration_hours': 1, 'is_preferred_working_hour': 1, 'overlaps_break': 0, 'task_importance': 2, 'task_difficulty': 1, 'suitability_score': 0.85},
        {'time_until_deadline_hours': 10, 'slot_duration_hours': 0.5, 'estimated_task_duration_hours': 3, 'is_preferred_working_hour': 0, 'overlaps_break': 1, 'task_importance': 3, 'task_difficulty': 2, 'suitability_score': 0.1},
        {'time_until_deadline_hours': 5, 'slot_duration_hours': 6, 'estimated_task_duration_hours': 1, 'is_preferred_working_hour': 0, 'overlaps_break': 0, 'task_importance': 1, 'task_difficulty': 1, 'suitability_score': 0.3},
        {'time_until_deadline_hours': 200, 'slot_duration_hours': 1, 'estimated_task_duration_hours': 6, 'is_preferred_working_hour': 1, 'overlaps_break': 0, 'task_importance': 3, 'task_difficulty': 3, 'suitability_score': 0.4},
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

suitability_model = train_suitability_model_from_csv('suitability_training_data.csv')

def get_suitability_score(features_df):
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
    return max(0.0, min(1.0, score))

# --- 4. Optimization Layer (Now with soft/hard constraint logic) ---

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
    # Determine scheduling mode
    scheduling_mode = getattr(user, "scheduling_preference_mode", None)
    if scheduling_mode is None:
        scheduling_mode = 'preferred_only' if user.check_working_constraints() else 'flexible_hours'

    for task in pending_tasks:
        for i, (slot_start, slot_end) in enumerate(available_time_slots):
            # Tasks cannot be assigned if the slot ends after their deadline.
            if slot_end > task.deadline:
                scores[(task.task_id, i)] = -1000000
            else:
                features = generate_features(task, user, slot_start, slot_end)
                features_df = pd.DataFrame([features])
                score = get_suitability_score(features_df)
                # --- Soft/Hard constraint logic ---
                if scheduling_mode == 'preferred_only':
                    if features['is_preferred_working_hour'] == 0:
                        # Hard exclusion
                        score = -1000000
                elif scheduling_mode == 'flexible_hours':
                    if features['is_preferred_working_hour'] == 0:
                        # Soft penalty (downscale score)
                        score *= 0.5
                scores[(task.task_id, i)] = score

    prob += lpSum([scores[(task.task_id, i)] * task_slot_vars[(task.task_id, i)]
                   for task in pending_tasks for i in range(len(available_time_slots))]), "Total Suitability Score"

    # Constraints:
    for task in pending_tasks:
        prob += lpSum(task_slot_vars[(task.task_id, i)] for i in range(len(available_time_slots))) <= 1, \
                f"Task_{task.task_id}_One_Slot_Max"

    for i, (slot_start, slot_end) in enumerate(available_time_slots):
        slot_duration = (slot_end - slot_start).total_seconds() / 3600
        prob += lpSum(task.estimated_duration_hours * task_slot_vars[(task.task_id, i)] for task in pending_tasks) <= slot_duration, \
                f"Slot_{i}_Capacity"

    for task in pending_tasks:
        for i, (slot_start, slot_end) in enumerate(available_time_slots):
            slot_features = generate_features(task, user, slot_start, slot_end)
            if slot_features['overlaps_break'] == 1:
                prob += task_slot_vars[(task.task_id, i)] == 0, \
                        f"Task_{task.task_id}_Slot_{i}_NoBreakOverlap"
    for task in pending_tasks:
        for i, (slot_start, slot_end) in enumerate(available_time_slots):
            if task.day_of_week is not None and slot_start.weekday() != task.day_of_week:
                prob += task_slot_vars[(task.task_id, i)] == 0, \
                        f"Task_{task.task_id}_Slot_{i}_WrongDay"
            if task.start_hour is not None and (slot_start.hour < task.start_hour or slot_end.hour > task.end_hour):
                prob += task_slot_vars[(task.task_id, i)] == 0, \
                        f"Task_{task.task_id}_Slot_{i}_WrongHour"

    # (No need for old hard constraint block, now handled above)

    prob.solve(PULP_CBC_CMD(msg=0))

    scheduled_tasks_by_slot = {i: [] for i in range(len(available_time_slots))}
    assigned_task_ids = set()

    if LpStatus[prob.status] == "Optimal":
        for task in pending_tasks:
            for i, (slot_start, slot_end) in enumerate(available_time_slots):
                if task_slot_vars[(task.task_id, i)].varValue > 0.9:
                    scheduled_tasks_by_slot[i].append(task)
                    assigned_task_ids.add(task.task_id)
        final_schedule_output = {}
        for i, (slot_start, slot_end) in enumerate(available_time_slots):
            if scheduled_tasks_by_slot[i]:
                final_schedule_output[(slot_start, slot_end)] = scheduled_tasks_by_slot[i]
        unscheduled_tasks = [task for task in pending_tasks if task.task_id not in assigned_task_ids]
        return final_schedule_output, unscheduled_tasks
    else:
        print(f"Solver did not find an optimal solution. Status: {LpStatus[prob.status]}. All tasks considered unscheduled.")
        return {}, pending_tasks

# --- Helper Function to Generate Contiguous Free Time Blocks (No Change) ---

def generate_free_time_blocks(user, start_datetime, end_datetime, min_slot_duration_minutes=30, use_preferred_only=True):
    free_blocks = []
    # --- Timezone fix: ensure UTC-aware ---
    if start_datetime.tzinfo is None:
        start_datetime = start_datetime.replace(tzinfo=timezone.utc)
    if end_datetime.tzinfo is None:
        end_datetime = end_datetime.replace(tzinfo=timezone.utc)
    # ------------------------------------------------
    current_day = start_datetime.date()
    while current_day <= end_datetime.date():
        day_of_week = current_day.weekday()
        # --- Always skip days not in preferred_working_hours ---
        preferred_days = [d for d, _, _ in user.preferred_working_hours]
        if day_of_week not in preferred_days:
            current_day += timedelta(days=1)
            continue
        # -------------------------------------------------------
        general_start = datetime(current_day.year, current_day.month, current_day.day, user.general_start_hour, 0, tzinfo=timezone.utc)
        general_end = datetime(current_day.year, current_day.month, current_day.day, user.general_end_hour, 0, tzinfo=timezone.utc)
        if use_preferred_only:
            daily_working_intervals = []
            for pref_day, start_h, end_h in user.preferred_working_hours:
                if pref_day == day_of_week:
                    interval_start = max(datetime(current_day.year, current_day.month, current_day.day, start_h, 0, tzinfo=timezone.utc), general_start)
                    interval_end = min(datetime(current_day.year, current_day.month, current_day.day, end_h, 0, tzinfo=timezone.utc), general_end)
                    if interval_start < interval_end:
                        daily_working_intervals.append((interval_start, interval_end))
        else:
            if general_start < general_end:
                daily_working_intervals = [(general_start, general_end)]
            else:
                daily_working_intervals = []
        daily_break_intervals = []
        for break_day, start_h, end_h in user.breaks:
            if break_day == day_of_week:
                daily_break_intervals.append((
                    datetime(current_day.year, current_day.month, current_day.day, start_h, 0, tzinfo=timezone.utc),
                    datetime(current_day.year, current_day.month, current_day.day, end_h, 0, tzinfo=timezone.utc)
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

    now = datetime(2025, 6, 16, 8, 12, 31, tzinfo=timezone.utc) # Monday, June 16, 2025, 8:12 AM

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