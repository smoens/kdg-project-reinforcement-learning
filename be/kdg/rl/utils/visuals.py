import numpy as np
import seaborn as sns
    #https://www.python-graph-gallery.com/92-control-color-in-seaborn-heatmaps
    #https://seaborn.pydata.org/generated/seaborn.heatmap.html
import matplotlib.pyplot as plt
    #https://matplotlib.org/stable/tutorials/colors/colormaps.html


class QValuesVisual:
    def plot(q_values):
        p = sns.heatmap(q_values, cmap="Blues", annot=True, cbar=False, square=False, vmin=0, vmax=1)
        plt.show()

class PolicyVisual:
    #https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html
    #https://kivy-garden.github.io/graph/flower.html
    pass

class ReturnVisual:
    pass