import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
sns.set_style("whitegrid")
plt.rc('font', size=12)

class MakePlots(object):

    def __init__(
            self, 
            dataset=None, 
            path_export="",
            hue="",
            sample=""):
        
        self.palette_values = ['#026E81', '#00ABBD', '#FFB255', '#F45F74']
        self.colors = sns.color_palette(self.palette_values)

        self.dataset = dataset
        self.path_export = path_export
        self.hue = hue
        self.sample = sample
    
    def plot_by_stage(self, name_fig=None):
        if name_fig is None:
            name_fig = f"ml_classic_performance_by_Stage.png"

        fig = plt.figure(figsize=(15, 18))
        gs = GridSpec(3, 2, figure=fig)

        axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
        metrics = ["Accuracy", "Precision", "Recall", "F1-score", "MCC", "ROC-AUC"]

        for ax, metric in zip(axes, metrics):
            sns.boxplot(ax=ax, data=self.dataset, y="Stage", x=metric, hue=self.hue, fill=False, palette=self.colors)
            ax.set_title(metric)

        plt.tight_layout()
        plt.savefig(f"{self.path_export}/{name_fig}", dpi=300)
    
    def plot_by_algorithm(self, name_fig=None):
        name_fig = f"ml_classic_performance_by_algorithm_{self.sample}.png"

        fig = plt.figure(figsize=(15, 18))
        gs = GridSpec(3, 2, figure=fig)

        axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
        metrics = ["Accuracy", "Precision", "Recall", "F1-score", "MCC", "ROC-AUC"]

        for ax, metric in zip(axes, metrics):
            sns.boxplot(ax=ax, data=self.dataset, y="Algorithm", x=metric, hue=self.hue, fill=False, palette=self.colors)
            ax.set_title(metric)

        plt.tight_layout()
        plt.savefig(f"{self.path_export}/{name_fig}", dpi=300)