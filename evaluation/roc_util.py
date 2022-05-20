from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, auc
from sklearn.metrics import roc_auc_score

# n_classes = 5

def get_integer_mapping(le):
    '''
    Return a dict mapping labels to their integer values
    from an SKlearn LabelEncoder
    le = a fitted SKlearn LabelEncoder
    '''
    res = {}
    i = 0
    for cl in le.classes_:
        res.update({i: cl})
        i += 1

    return res



def precision_recall_graph(y_test, y_pred, y_pred_proba):
    """
    print precision recall graph with AUC and precision for each label
    :param y_test: test true values
    :param y_score: predicted probabilities
    :param classes: classes names
    :param method: name of the mthod performed / data kind (will be printed in the headline)
    """

    n_classes = np.append(np.unique(y_test), np.unique(y_pred))
    n_classes = len(set(n_classes))

    # Binarize the output
    lb = LabelBinarizer()
    lb.fit(np.concatenate([y_pred.values, y_test.values]))
    y_test = lb.transform(y_test)
    # y_score = lb.transform(y_pred)
    y_score = y_pred_proba

    if n_classes < 3:
        y_test = np.hstack((1 - y_test, y_test))

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = precision_score(y_test[:, i], np.round(y_score[:, i]))

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
                                                                    y_score.ravel())
    # average_precision["micro"] = precision_score(y_test, np.round(y_score), average="micro")

    auc_score = auc(recall["micro"], precision["micro"])
    print('PR AUC: %.3f' % auc_score)
    return auc_score

    # A "micro-average": quantifying score on all classes jointly
    # precision["macro"], recall["macro"], _ = precision_recall_curve(Y_test.ravel(),
    #                                                                 y_score.ravel())
    # average_precision["macro"] = average_precision_score(Y_test, y_score,
    #                                                      average="macro")

# save logs flow
#TODO:add common code to pr and auc*2
def calc_auc_roc(y_test, y_pred, y_pred_proba):
    # y_test = y_test_path
    # y_pred = y_pred_path
    n_classes = np.append(np.unique(y_test), np.unique(y_pred))
    n_classes = len(set(n_classes))

    # Binarize the output
    lb = LabelBinarizer()
    lb.fit(np.concatenate([y_pred.values, y_test.values]))
    y_test = lb.transform(y_test)
    # y_score = lb.transform(y_pred)
    y_score = y_pred_proba

    if n_classes < 3:
        y_test = np.hstack((1 - y_test, y_test))
        # y_score = np.hstack((y_score, 1 - y_score))
    # n_classes = len(y_test.unique())

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i], drop_intermediate=False)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    ##############################################################################
    # Plot ROC curves for the multiclass problem

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # print('macro auc: {}'.format(roc_auc["macro"]))
    # print('micro auc: {}'.format(roc_auc["micro"]))

    # get score for each class, replace dict keys to class names
    integerMapping = get_integer_mapping(lb)
    for i in range(n_classes):
        class_as_key = integerMapping.get(i)
        roc_auc[class_as_key] = roc_auc[i]
        del roc_auc[i]



    return roc_auc


# def calc_and_plot_roc_orig(y_test_path, y_pred_path,  save_path, model_name_title='', show =True):
#     y_test = pd.read_csv(y_test_path)
#     y_pred = pd.read_csv(y_pred_path)
#
#     # y_test = y_test_path
#     # y_pred = y_pred_path
#     n_classes = np.append(np.unique(y_test), np.unique(y_pred))
#     n_classes = len(set(n_classes))
#
#     # Binarize the output
#     lb = LabelBinarizer()
#     lb.fit(np.concatenate([y_pred.values, y_test.values]))
#     y_test = lb.transform(y_test)
#     y_score = lb.transform(y_pred)
#
#     if n_classes < 3:
#         y_test = np.hstack((y_test, 1 - y_test))
#         y_score = np.hstack((y_score, 1 - y_score))
#     # n_classes = len(y_test.unique())
#     integerMapping = get_integer_mapping(lb)
#
#     # Compute ROC curve and ROC area for each class
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     for i in range(n_classes):
#         fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
#
#     # Compute micro-average ROC curve and ROC area
#     fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
#     ##############################################################################
#     # Plot ROC curves for the multiclass problem
#
#     # Compute macro-average ROC curve and ROC area
#
#     # First aggregate all false positive rates
#     all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#
#     # Then interpolate all ROC curves at this points
#     mean_tpr = np.zeros_like(all_fpr)
#     for i in range(n_classes):
#         mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#
#     # Finally average it and compute AUC
#     mean_tpr /= n_classes
#
#     fpr["macro"] = all_fpr
#     tpr["macro"] = mean_tpr
#     roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#
#     # Plot all ROC curves
#     lw = 2
#     fig = plt.figure()
#     plt.plot(fpr["micro"], tpr["micro"],
#              label='micro-average ROC curve (area = {0:0.2f})'
#                    ''.format(roc_auc["micro"]),
#              color='deeppink', linestyle=':', linewidth=4)
#
#     plt.plot(fpr["macro"], tpr["macro"],
#              label='macro-average ROC curve (area = {0:0.2f})'
#                    ''.format(roc_auc["macro"]),
#              color='navy', linestyle=':', linewidth=4)
#
#     colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'navy'])
#     for i, color in zip(range(n_classes), colors):
#         plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#                  label='ROC curve of class {0} (area = {1:0.2f})'
#                  ''.format(integerMapping.get(i), roc_auc[i]))
#
#     plt.plot([0, 1], [0, 1], 'k--', lw=lw)
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(model_name_title + ' ROC curves per class and AUC scores')
#     plt.legend(loc="lower right")
#     if show:
#         plt.show()
#
#     fig.savefig(save_path)


def calc_and_plot_roc(y_test_path, y_pred_path, y_pred_proba_path,  save_path, model_name_title='', show=True,
                      should_plot=True):
    y_test = pd.read_csv(y_test_path)
    y_pred_proba = pd.read_csv(y_pred_proba_path)
    y_pred = pd.read_csv(y_pred_path)

    # y_test = y_test_path
    # y_pred = y_pred_path
    n_classes = np.append(np.unique(y_test), np.unique(y_pred))
    n_classes = len(set(n_classes))

    # # Binarize the output
    lb = LabelBinarizer()
    lb.fit(np.concatenate([y_pred.values, y_test.values]))
    y_test = lb.transform(y_test)
    # y_score = lb.transform(y_pred)
    y_score = y_pred_proba.values
    if n_classes < 3:
        y_test = np.hstack((1 - y_test, y_test))
        # y_score = np.hstack((y_score, 1 - y_score))
    # n_classes = len(y_test.unique())
    integerMapping = get_integer_mapping(lb)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i], drop_intermediate=False)
        roc_auc[i] = auc(fpr[i], tpr[i])


        # auc1 = roc_auc_score(y_test[:, i], y_score[:, i])
        # print(auc1)

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel(), drop_intermediate=False)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    ##############################################################################
    # Plot ROC curves for the multiclass problem

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    if should_plot:
        # Plot all ROC curves
        lw = 2
        fig = plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'navy'])
        for i, color in zip(range(n_classes), colors):
            print('{}:{}'.format(integerMapping.get(i), roc_auc[i]))
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.4f})'
                     ''.format(integerMapping.get(i), roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(model_name_title + ' ROC curves per class and AUC scores')
        plt.legend(loc="lower right")
        if show:
            plt.show()

        fig.savefig(save_path)
    print('macro auc: {}'.format(roc_auc["macro"]))
    print('micro auc: {}'.format(roc_auc["micro"]))
    return roc_auc["macro"], roc_auc["micro"]

