import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
import seaborn as sns


class DataVisualization:
    """
    Class to visualize typical results in data science.

    Parameters
    ----------
    model : object
        The trained model.
    X_test : DataFrame or array-like
        Test features.
    y_test : array-like
        True labels for the test set.
    y_pred_prob : array-like
        Predicted probabilities for the test set.
    """

    def __init__(self, model, X_test, y_test, y_pred_prob):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred_prob = y_pred_prob

    def plot_roc_curve(self):
        """
        Plot ROC curve for binary classification.
        """
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_prob[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

    def plot_probability_density(self):
        """
        Plot histogram and density of predicted probabilities by label.
        """
        plt.figure()
        sns.histplot(self.y_pred_prob[:, 1][self.y_test == 0], color='red', label='Class 0', kde=True, stat="density",
                     bins=50)
        sns.histplot(self.y_pred_prob[:, 1][self.y_test == 1], color='blue', label='Class 1', kde=True, stat="density",
                     bins=50)
        plt.title('Predicted Probability Density')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.legend(loc='upper right')
        plt.show()

    def plot_calibration_curve(self):
        """
        Plot calibration curve to assess probability calibration.
        """
        prob_true, prob_pred = calibration_curve(self.y_test, self.y_pred_prob[:, 1], n_bins=10)

        plt.figure()
        plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Calibration curve')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
        plt.title('Calibration Curve')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.legend()
        plt.show()

    def plot_feature_importance(self, feature_names):
        """
        Plot feature importance for tree-based models.
        """
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            indices = np.argsort(importance)[::-1]
            plt.figure(figsize=(10, 6))
            plt.title('Feature Importance')
            plt.bar(range(len(importance)), importance[indices], align='center')
            plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.show()
        else:
            print("Feature importance is not available for this model.")

    def plot_confusion_matrix(self, confusion_matrix, class_names):
        """
        Plot confusion matrix.
        """
        plt.figure()
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                    yticklabels=class_names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()