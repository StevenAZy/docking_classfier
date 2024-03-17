from sklearn.metrics import confusion_matrix

def calculate_metrics(labels, preds):
        cm = confusion_matrix(labels, preds)

        accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
        precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
        recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        f1 = 2 * (precision * recall) / (precision + recall)

        return accuracy, precision, recall, f1, cm