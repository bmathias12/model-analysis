from binaryclassifier.plotting import plot_roc_curve
from binaryclassifier.plotting import plot_prediction_density
from binaryclassifier.plotting.colors import THEME_COLORS

class Analyzer:
    
    def __init__(self, y_true, scores, preds):
        self.y_true = y_true
        self.scores = scores
        self.preds = preds

    def plot_roc_curve(self, color=THEME_COLORS[0]):
        return plot_roc_curve(self.y_true, self.scores, color=color)

    def plot_prediction_density(self, colors=THEME_COLORS):
        return plot_prediction_density(self.y_true, self.scores, colors=colors)



