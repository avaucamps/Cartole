from Environment import Environment
from graph_builder import plot_graph
from constants import MAX_AVG_SCORE, CONSECUTIVE_RUNS_TO_SOLVE


ENV_NAME = "CartPole-v1"


if __name__ == "__main__":
    env = Environment(ENV_NAME)
    scores = env.run()
    plot_graph(ENV_NAME, scores, MAX_AVG_SCORE, CONSECUTIVE_RUNS_TO_SOLVE)