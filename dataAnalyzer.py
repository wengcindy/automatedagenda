import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels


LABEL_NAMES = {  # For confusion matrix
    1: 'Pro',
    0: 'Neutral',
    -1: 'Con'
}


class DataAnalyzer:
    """
    Records all predicted and actual data from the model, both across all
    sessions and specific to each session.
    Gives accuracy and confusion matrices.
    """
    def __init__(self):
        self.predicted = []
        self.true = []
        self.session_accuracies = {}
        self.session_confusion_matrices = {}
        self.session_start_index = {}
        self.session_end_index = {}  # Exclusive
        self.current_session = None

    def accuracy_from_range(self, start_index=0, end_index=None):
        """
        Calculate the accuracy of a given range of data (e.g. a particular
        session).
        :param start_index: Start index (default 0)
        :param end_index: End index (default length of data)
        :return: Accuracy
        """
        if end_index is None:
            end_index = len(self.true)
        return accuracy_score(self.true[start_index:end_index],
                              self.predicted[start_index:end_index])

    @staticmethod
    def stringify_data(data):
        """
        Convert list of integers to strings for readability in confusion
        matrices.
        :param data: Data in integers
        :return: Data in strings
        """
        return [LABEL_NAMES.get(v, v) for v in data]

    def confusion_matrix_from_range(self, start_index=0, end_index=None):
        """
        Compute the confusion matrix of a given range of data (e.g. a
        particular session).
        :param start_index: Start index (default 0)
        :param end_index: End index (default length of data)
        :return: Confusion matrix
        """
        if end_index is None:
            end_index = len(self.true)
        return confusion_matrix(
            self.stringify_data(self.true[start_index:end_index]),
            self.stringify_data(self.predicted[start_index:end_index]),
            labels=list(LABEL_NAMES.values())
        )

    def new_session(self, session_name):
        """
        Starts a new session. (A session is one single "room" of the platform
        with several agenda items/sections.)
        This ends the previous session and stores its statistics.
        :param session_name: Name of new session
        :return: (accuracy, confusion matrix) of previous session, or
            (None, None) if this is the first session
        """
        prev_accuracy, prev_matrix = None, None
        if self.current_session:
            prev_accuracy, prev_matrix = self.end_session()
        self.current_session = session_name
        self.session_start_index[session_name] = len(self.true)
        return prev_accuracy, prev_matrix

    def end_session(self):
        """
        Ends the current session.
        This stores the session's statistics such as accuracy and confusion
        matrix.
        :return: (accuracy, confusion matrix) of current session
        """
        accuracy = self.accuracy_from_range(
            start_index=self.session_start_index[self.current_session])
        matrix = self.confusion_matrix_from_range(
            start_index=self.session_start_index[self.current_session])
        self.session_accuracies[self.current_session] = accuracy
        self.session_confusion_matrices[self.current_session] = matrix
        self.session_end_index[self.current_session] = len(self.true)
        self.current_session = None
        return accuracy, matrix

    def add(self, pred, true):
        """
        Add data to current session.
        :param pred: Predicted label
        :param true: True label
        """
        self.predicted.append(pred)
        self.true.append(true)

    def get_accuracy(self, session=None):
        """
        Get accuracy of a session, or the overall accuracy (if session is None).
        :param session: Session name
        :return: Accuracy of session, or overall accuracy if session=None
        """
        if session is None:
            return self.accuracy_from_range()
        return self.accuracy_from_range(
            start_index=self.session_start_index[session],
            end_index=self.session_end_index[session])

    def get_confusion_matrix(self, session=None):
        """
        Get confusion matrix of a session, or the overall accuracy (if session
        is None).
        :param session: Session name
        :return: Confusion matrix of session, or overall matrix if session=None
        """
        if session is None:
            return self.confusion_matrix_from_range()
        return self.confusion_matrix_from_range(
            start_index=self.session_start_index[session],
            end_index=self.session_end_index[session])

    def print_session_accuracies(self, end_current_session=True):
        """
        Print accuracies of all sessions to standard output.
        Caution: This function ends the current session by default.
        :param end_current_session: Whether current session should be
            terminated
        """
        if end_current_session:
            self.end_session()
        for session, accuracy in self.session_accuracies.items():
            print("Session %s \t Accuracy: %f" % (session, accuracy))

    def plot_confusion_matrix(self, session=None, filename=None, title=None):
        """
        Plot the confusion matrix for a session or the overall confusion
        matrix, and save it as an image file.
        :param session: Session name (None for overall)
        :param filename: File name (without .png)
        :param title: Title of plot
        """
        plot_confusion_matrix(cm=self.get_confusion_matrix(session),
                              title=title)
        plt.savefig((filename if filename else 'Confusion Matrix') + '.png')

    def reset(self):
        """
        Clear all stored data and reset the data analyzer.
        """
        self.predicted = []
        self.true = []
        self.session_accuracies = {}
        self.session_confusion_matrices = {}
        self.session_start_index = {}
        self.session_end_index = {}  # Exclusive
        self.current_session = None


def plot_confusion_matrix(y_true=[], y_pred=[], cm=None,
                          classes=list(LABEL_NAMES.values()),
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if cm is None:
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        #classes = classes[unique_labels(y_true, y_pred)]  # Problematic
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
