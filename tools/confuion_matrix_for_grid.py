from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import six
plt.style.use('seaborn-poster')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.edgecolor'] = '#ffffff'
plt.rcParams['axes.facecolor'] = '#ffffff'
plt.rcParams['figure.facecolor'] = '#ffffff'
plt.rcParams['patch.edgecolor'] = '#ffffff'
plt.rcParams['patch.facecolor'] = '#ffffff'
plt.rcParams['savefig.edgecolor'] = '#ffffff'
plt.rcParams['savefig.facecolor'] = '#ffffff'
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

class My_Confusion_Matrix:
    def __init__(self, ax0, ax1, cf, group_names=None, categories='auto', count=True,
                 percent=True, cbar=True, xyticks=True, xyplotlabels=True,
                 sum_stats=True, figsize: tuple = (10, 15), title: str = 'Confusion Matrix', cmap='Blues', xlabel = 'Predicted label', ylabel = 'True label'):
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
        self.__ax0 = ax0
        self.__ax1= ax1
        self.xlabel=xlabel
        self.ylabel=ylabel

    def __make_text_inside_ech_squere(self):
        blanks = ['' for i in range(self.cf.size)]

        if self.group_names and len(self.group_names) == self.cf.size:
            group_labels = ["{}\n".format(value) for value in self.group_names]
        else:
            group_labels = blanks

        if self.count:
            group_counts = ["{0:0.0f}\n".format(value) for value in self.cf.flatten()]
        else:
            group_counts = blanks

        if self.percent:
            group_percentages = ["{0:.2%}".format(value) for value in self.cf.flatten() / np.sum(self.cf)]
        else:
            group_percentages = blanks

        box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
        box_labels = np.asarray(box_labels).reshape(self.cf.shape[0], self.cf.shape[1])
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
        tab.set_fontsize(10)
        header_color = '#40466e'
        row_colors = ['#f1f1f2', 'w']
        edge_color = 'w'
        for k, cell in six.iteritems(tab._cells):
            cell.set_edgecolor(edge_color)
            if k[0] == 0 or k[1] < 0:
                cell.set_text_props(weight='bold', color='w', size=10)
                cell.set_facecolor(header_color)
            else:
                cell.set_facecolor(row_colors[k[1] % len(row_colors)])

    def make_confusion_matrix(self):

        box_labels = self.__make_text_inside_ech_squere()

        df_stats = self.__make_dataframe_sumary_statistics()
        

        # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
        if self.xyticks == False:
            self.categories = False

        sns.set(font_scale=0.9)
        sns.heatmap(self.cf, annot=box_labels, fmt="", cmap=self.cmap, cbar=self.cbar, xticklabels=self.categories,
                    yticklabels=self.categories, annot_kws={"size": 10}, ax=self.__ax0)

        if self.xyplotlabels:
            self.__ax0.set_ylabel(self.ylabel, fontsize=10)
            self.__ax0.set_xlabel(self.xlabel, fontsize=10)
        else:
            self.__ax0.set_xlabel('')

        if self.title:
            self.__ax0.set_title(self.title, fontsize=12)

        for tick in self.__ax0.xaxis.get_major_ticks():
            tick.label.set_fontsize(10)

        for tick in self.__ax0.yaxis.get_major_ticks():
            tick.label.set_fontsize(10)

        self.__write_table(ax=self.__ax1, dff=df_stats)
        self.__ax1.set_axis_off()