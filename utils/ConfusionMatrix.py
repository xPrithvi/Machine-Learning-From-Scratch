# Imports,
import numpy as np
import matplotlib.pyplot as plt

class ConfusionMatrix():
    """Class for the confusion matrix."""

    def __init__(self, clf, X, y):
        """Constructor method which computes the confusion matrix."""

        # Initialising confusion matrix,
        self.class_labels = np.unique(y)
        n_labels = len(self.class_labels)
        confusion_matrix = np.zeros(shape=(n_labels, n_labels), dtype=int)

        # Constructing confusion matrix,
        class_labels_preds = clf.predict(X) # <-- Predicted class labels.
        for class_label_pred, class_label in zip(class_labels_preds, y):
            class_label_pred, class_label = int(class_label_pred), int(class_label)
            confusion_matrix[class_label_pred, class_label] += 1

        # Assigning confusion matrix as an attribute,
        self.confusion_matrix = confusion_matrix

    def report(self, beta=1):
        """Prints a classification report."""

        # Computing metrics,
        precision = self.precision()
        recall = self.recall()
        support = self.support()
        fscores = self.fscore(beta)

        # Computing aggregate metrics,
        accuracy = self.accuracy()
        precision_mean, precision_weighted_mean  = np.mean(precision), np.average(precision, weights=support)#
        recall_mean, recall_weighted_mean  = np.mean(recall), np.average(recall, weights=support)
        fscores_mean, fscores_weighted_mean  = np.mean(fscores), np.average(fscores, weights=support)

        # Printing report,
        print(f"Classification Accuracy: {accuracy:.3f}")
        print(f"β: {beta}\n")
        print(f"{'Class':<12}{'Precision':>10}{'Recall':>10}{'Fβ-score':>10}{'Support':>10}")
        print("-" * 52)

        for class_idx, class_label in enumerate(self.class_labels):
            print(f"{str(class_label):<12}{precision[class_idx]:>10.3f}{recall[class_idx]:>10.3f}{fscores[class_idx]:>10.3f}{support[class_idx]:>10}")

        print("-" * 52)
        print(f"{'Average':<16}{precision_mean:>10.3f}{recall_mean:>10.3f}{fscores_mean:>10.3f}")
        print(f"{'Weighted-average':<16}{precision_weighted_mean:>10.3f}{recall_weighted_mean:>10.3f}{fscores_weighted_mean:>10.3f}")

        return None

    def accuracy(self):
        """Computes the classification accuracy from confusion matrix."""
        return np.sum(np.diagonal(self.confusion_matrix))/np.sum(self.confusion_matrix)

    def precision(self):
        """Computes the precision of the model for each class and returns them as an array."""
        return np.diagonal(self.confusion_matrix)/np.sum(self.confusion_matrix, axis=1)

    def recall(self):
        """Computes the recall of the model for each class and returns them as an array."""
        return np.diagonal(self.confusion_matrix)/np.sum(self.confusion_matrix, axis=0)
    
    def support(self):
        """Computes the number of data samples for each class"""
        return np.sum(self.confusion_matrix, axis=0)

    def fscore(self, beta=1):
        """Computes the weighted F-score with β=1 as the default (canonical case)."""

        # Computing the recall and precision vectors,
        recall, precision = self.precision(), self.precision()

        # Computing and returning the F-scores,
        return (1 + beta**2)*(precision * recall)/(precision*beta**2 + recall)

    def __repr__(self):
        """The presentation of the class."""
        return repr(self.confusion_matrix)

def PlotConfusionMatrix(conf_matrix_obj):
    """Displays the confusion matrix givem using Matplotlib."""

    # Extracting confusion matrix,
    conf_matrix = conf_matrix_obj.confusion_matrix

    # Creating figure,
    plt.figure(figsize=(6, 5))
    im = plt.imshow(conf_matrix, cmap="Reds")
    plt.colorbar(im)

    # Adding values,
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, conf_matrix[i, j], ha="center", va="center", color="black")

    # Adding class labels on axes,
    class_labels = conf_matrix_obj.class_labels
    if class_labels is not None:
        plt.xticks(range(len(class_labels)), class_labels)
        plt.yticks(range(len(class_labels)), class_labels)
    else:
        plt.xticks(range(conf_matrix.shape[1]))
        plt.yticks(range(conf_matrix.shape[0]))

    # Adding labels,
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()