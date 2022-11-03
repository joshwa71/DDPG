
import numpy as np
import matplotlib.pyplot as plt

def plot_reward_curve(scores, x, file_path):
    trailing_average = np.zeros(len(scores))
    for i in range(len(trailing_average)):
        trailing_average[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(trailing_average, x)
    plt.title('Trailing average of rewards for 100 games')
    plt.savefig(file_path)