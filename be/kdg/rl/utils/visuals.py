import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class QValuesVisual:
    def plot(q_values):
        p = sns.heatmap(q_values, cmap="Blues", annot=True, cbar=False, square=False, vmin=0, vmax=1)
        return p

class PolicyVisual:
    def plot(policy):
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
        p = plt.quiver(x_values, y_values)
        return p

class ReturnVisual:
    pass