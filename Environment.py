import gym
from DQNSolver import DQNSolver
import numpy as np
from statistics import mean
from constants import MAX_AVG_SCORE, CONSECUTIVE_RUNS_TO_SOLVE


class Environment:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.solver = DQNSolver(self.observation_space, self.action_space)
        self.scores = [0]

    
    def run(self):
        n_episode = 0
        while mean(self.scores[-CONSECUTIVE_RUNS_TO_SOLVE:]) < MAX_AVG_SCORE or \
            n_episode < CONSECUTIVE_RUNS_TO_SOLVE:
                state = self.env.reset()
                state = np.reshape(state, [1, self.observation_space])
                n_episode += 1
                self._run_episode(state, n_episode)

        self.scores.pop(0)
        return self.scores

    
    def _run_episode(self, state, n_episode):
        current_score = 0
        while True:
            current_score += 1
            self.env.render()
            action = self.solver.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            next_state = np.reshape(next_state, (1, self.observation_space))
            reward = reward if not done else -reward
            self.solver.save_observation(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                self.scores.append(current_score)
                print("Episode: {}, score: {}, avg score: {}, exploration rate: {}"
                    .format(n_episode, current_score, mean(self.scores[-CONSECUTIVE_RUNS_TO_SOLVE:]), self.solver.exploration_rate))
                return
            
            self.solver.train()