import matplotlib.pyplot as plt 
from statistics import mean


def plot_graph(env_name, scores, goal, goal_duration):
        x = []
        y = []
        for i in range(len(scores)):
                x.append(i+1)
                y.append(scores[i])

        plt.subplots()
        plt.plot(x, y, label="Score")
        plt.plot(x[-goal_duration:], [mean(y[-goal_duration:])] * len(x[-goal_duration:]), 
                linestyle="--", label="Last {} scores average".format(len(x[-goal_duration:])))
        plt.plot(x, [goal] * len(x), linestyle=":", label="Average score goal")
        plt.title(env_name)
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.legend(loc="upper left")
        plt.show()