#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import tensorflow as tf
from collections import deque
import random

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# In[4]:


class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


# In[5]:


class TaskSchedulerEnv:
    def __init__(self):
        self.max_deadline = 30  # Max days for a task deadline
        self.action_space = np.linspace(0.1, 1.0, 10)  # Deadline fractions (0.1 to 1.0)
        self.state_size = 2  # [urgency, student_type]
        self.action_size = len(self.action_space)
        self.reset()

    def reset(self):
        # Randomly initialize task urgency (time remaining) and student type
        self.time_remaining = np.random.randint(5, self.max_deadline + 1)
        self.student_type = np.random.choice([0, 1])  # 0: active, 1: procrastinator
        self.urgency = 1 - (self.time_remaining / self.max_deadline)  # Normalize urgency
        self.state = np.array([[self.urgency, self.student_type]])
        return self.state

    def step(self, action):
        # Action is an index selecting a deadline fraction
        deadline_fraction = self.action_space[action]
        assigned_deadline = max(1, int(self.time_remaining * deadline_fraction))

        # Simulate task completion
        if self.student_type == 1:  # Procrastinator
            # Procrastinators tend to delay, so tighter deadlines are better
            completion_time = np.random.randint(assigned_deadline, self.time_remaining + 1)
            if completion_time <= assigned_deadline:
                reward = 10  # Task completed on time
            else:
                reward = -10  # Missed assigned deadline
            # Bonus for tight deadlines for procrastinators
            reward += (1 - deadline_fraction) * 5
        else:  # Active student
            # Active students complete tasks early or on time
            completion_time = np.random.randint(1, assigned_deadline + 1)
            if completion_time <= assigned_deadline:
                reward = 10
            else:
                reward = -10
            # Penalty for unnecessarily tight deadlines
            reward -= (1 - deadline_fraction) * 5

        # Update state
        self.time_remaining = max(1, self.time_remaining - assigned_deadline)
        self.urgency = 1 - (self.time_remaining / self.max_deadline)
        next_state = np.array([[self.urgency, self.student_type]])

        # Check if task is done
        done = self.time_remaining <= 0 or completion_time > self.time_remaining

        return next_state, reward, done, {"assigned_deadline": assigned_deadline}


# In[6]:


def train_dqn(episodes=1000):
    env = TaskSchedulerEnv()
    agent = DQN(state_size=env.state_size, action_size=env.action_size)
    batch_size = 32
    scores = []

    for e in range(episodes):
        state = env.reset()
        score = 0
        for time in range(100):  # Max steps per episode
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores.append(score)

        # Replay and update target model
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if e % 50 == 0:
            agent.update_target_model()

        print(f"Episode {e}/{episodes}, Score: {score}, Epsilon: {agent.epsilon:.2f}")

    return agent, scores


# In[7]:


# Test the trained model
def test_dqn(agent, episodes=10):
    env = TaskSchedulerEnv()
    for e in range(episodes):
        state = env.reset()
        student_type = "Procrastinator" if state[0][1] == 1 else "Active"
        print(f"\nEpisode {e+1}, Student Type: {student_type}, Initial Urgency: {state[0][0]:.2f}")
        done = False
        while not done:
            action = agent.act(state)
            deadline_fraction = env.action_space[action]
            next_state, reward, done, info = env.step(action)
            print(f"Assigned Deadline: {info['assigned_deadline']} days, Reward: {reward}")
            state = next_state
            if done:
                print("Task completed or deadline passed.")


# In[9]:


if __name__ == "__main__":
    # Train the model
    agent, scores = train_dqn(episodes=500)
    
    # Save the model
    agent.save("dqn_task_scheduler.weights.h5")
    
    # Test the model
    print("\nTesting the trained model:")
    test_dqn(agent, episodes=5)



def recommend_schedule(user_id):
    # TODO: Load user data and generate state
    # For now, simulate a dummy recommendation
    return {'user_id': user_id, 'recommended_slots': ['2025-04-18 10:00', '2025-04-18 14:00']}
