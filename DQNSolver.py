from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import random


GAMMA = 0.95
MEMORY_SIZE = 1000000
LEARNING_RATE = 0.001
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
BATCH_SIZE = 20


class DQNSolver:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.exploration_rate = EXPLORATION_MAX
        self._build_model()

    
    def _build_model(self):
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(self.observation_space,), activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    
    def save_observation(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    
    def get_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    
    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in batch:
            q_update = reward
            if not done:
                q_update = (reward + GAMMA * np.amax(self.model.predict(next_state)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)