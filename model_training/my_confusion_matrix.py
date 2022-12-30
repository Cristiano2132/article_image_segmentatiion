from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import six
from matplotlib.gridspec import GridSpec


class My_Confusion_Matrix:
    def __init__(self, cf, save_path: str, group_names=None, categories='auto', count=True,
                 percent=True, cbar=True, xyticks=True, xyplotlabels=True,
                 sum_stats=True, figsize: tuple = (10, 15), title: str = 'Confusion Matrix', cmap='Blues'):
        self.cmap = cmap
        self.group_names = group_names
        self.categories = categories
        self.count = count
        self.percent = percent
        self.cbar = cbar
        self.xyticks = xyticks
        self.xyplotlabels = xyplotlabels
        self.sum_stats = sum_stats
        self.figsize = figsize
        self.title = title
        self.cf = cf
        self.__path = save_path
        self.__title_font_size = 20
        self.__text_font_size = 18

    def __make_text_inside_ech_squere(self):
        blanks = ['' for i in range(self.cf.size)]

        if self.group_names and len(self.group_names) == self.cf.size:
            group_labels = ["{}\n".format(value) for value in self.group_names]
        else:
            group_labels = blanks

        if self.count:
            group_counts = ["{0:0.0f}\n".format(
                value) for value in self.cf.flatten()]
        else:
            group_counts = blanks

        if self.percent:
            group_percentages = ["{0:.2%}".format(
                value) for value in self.cf.flatten() / np.sum(self.cf)]
        else:
            group_percentages = blanks

        box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(
            group_labels, group_counts, group_percentages)]
        box_labels = np.asarray(box_labels).reshape(
            self.cf.shape[0], self.cf.shape[1])
        return box_labels

    def __make_dataframe_sumary_statistics(self):
        if self.sum_stats:
            # Accuracy is sum of diagonal divided by total observations
            accuracy = np.trace(self.cf) / float(np.sum(self.cf))

            # if it is a binary confusion matrix, show some more stats
            if len(self.cf) == 2:
                # Metrics for Binary Confusion Matrices
                precision = self.cf[1, 1] / sum(self.cf[:, 1])
                recall = self.cf[1, 1] / sum(self.cf[1, :])
                f1_score = 2 * precision * recall / (precision + recall)
                stats_dict = {'Accuracy': [accuracy],
                              'Precision': [precision],
                              'Recall': [recall],
                              'F1Score': [f1_score]}
            else:
                stats_dict = {'Accuracy': [accuracy]}
        else:
            stats_dict = ""
        return pd.DataFrame(stats_dict).round(3)

    def __write_table(self, ax, dff: pd.DataFrame):
        tab = ax.table(cellText=dff.values, colLabels=dff.columns, loc='bottom',
                       cellLoc='center', colLoc='left', bbox=[0.0, 0, 1, 1]
                       )
        tab.auto_set_font_size(False)
        tab.set_fontsize(self.__text_font_size)
        header_color = '#40466e'
        row_colors = ['#f1f1f2', 'w']
        edge_color = 'w'
        for k, cell in six.iteritems(tab._cells):
            cell.set_edgecolor(edge_color)
            if k[0] == 0 or k[1] < 0:
                cell.set_text_props(weight='bold', color='w', size=self.__text_font_size)
                cell.set_facecolor(header_color)
            else:
                cell.set_facecolor(row_colors[k[1] % len(row_colors)])

    def make_confusion_matrix(self):

        box_labels = self.__make_text_inside_ech_squere()

        df_stats = self.__make_dataframe_sumary_statistics()

        # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
        if self.xyticks == False:
            self.categories = False

        # MAKE THE HEATMAP VISUALIZATION
        fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(nrows=2, ncols=1, width_ratios=[1], height_ratios=[5, 1])
        ax0 = fig.add_subplot(gs[0, :])
        ax1 = fig.add_subplot(gs[1, :])
        sns.set(font_scale=1.5)
        sns.heatmap(self.cf, annot=box_labels, fmt="", cmap=self.cmap, cbar=self.cbar, xticklabels=self.categories,
                    yticklabels=self.categories, annot_kws={"size": self.__text_font_size}, ax=ax0)

        if self.xyplotlabels:
            ax0.set_ylabel('True label', fontsize=self.__text_font_size)
            ax0.set_xlabel('Predicted label', fontsize=self.__text_font_size)
        else:
            ax0.set_xlabel('')

        if self.title:
            ax0.set_title(self.title, fontsize=self.__title_font_size)

        for tick in ax0.xaxis.get_major_ticks():
            tick.label.set_fontsize(self.__text_font_size)

        for tick in ax0.yaxis.get_major_ticks():
            tick.label.set_fontsize(self.__text_font_size)

        self.__write_table(ax=ax1, dff=df_stats)
        ax1.set_axis_off()
        plt.tight_layout()
        fig.savefig(self.__path, dpi=600)
