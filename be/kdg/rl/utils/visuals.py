import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class QValuesVisual:
    def plot(q_values, count):
        plt.subplot(1, 2, 1)
        plt.suptitle(f'Episode {count}')
        plt.title('q-values')
        sns.heatmap(q_values, cmap="Blues", annot=True, cbar=False, square=False, vmin=0, vmax=1)

class PolicyVisual:
    def plot(policy, count):
        plt.subplot(1, 2, 2)
        plt.title('policy π')
        policy_by_action = np.argmax(np.transpose(policy), 1)
        action_mapping = {
            0: [-1, 0],     # Left
            1: [0, -1],     # Down
            2: [1, 0],      # Right
            3: [0, 1]       # Up
        }

        xy_values = [action_mapping[value] for value in policy_by_action]
        x_values = np.reshape([col[0] for col in xy_values], (4,4))
        y_values = np.reshape([col[1] for col in xy_values], (4,4))
        plt.quiver(x_values, y_values)
        plt.savefig(f'./output/images/qvals/episode_{count}.png')
        plt.clf()

class ReturnVisual:
    def plot(rewards, count):
        plt.subplot(1, 1, 1)
        plt.title('Average Reward in %')
        plt.plot(rewards.episode_nr, rewards.avg_reward)
        plt.savefig(f'./output/images/reward/episode_{count}.png')
        plt.clf()