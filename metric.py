import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay

def calculate_metrics(labels, preds):
        cm = confusion_matrix(labels, preds)

        accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
        precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
        recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        f1 = 2 * (precision * recall) / (precision + recall)

        return accuracy, precision, recall, f1, cm


def roc_auc_plt(labels, preds):
        fpr, tpr, thresholds = roc_curve(labels, preds)
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
        display.plot()
        plt.savefig('plt/roc.png')
