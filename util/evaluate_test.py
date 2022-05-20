from evaluation import confusion_matrix_util, roc_util
from sklearn.metrics import *
import pandas as pd


def evaluate_by_files(y_test_path, y_pred_path, y_pred_proba_path, class_names, target, save_plots_path_prefix):
    y_test_df = pd.read_csv(y_test_path)
    y_pred_df = pd.read_csv(y_pred_path)
    confusion_matrix_util.plot_confusion_matrix(y_test_path, y_pred_path, classes=class_names,
                                                save_path=save_plots_path_prefix + "lstm_confusion_m.png",
                                                title='helpers confusion matrix, without normalization ' + target)

    # Plot normalized confusion matrix
    confusion_matrix_util.plot_confusion_matrix(y_test_path, y_pred_path, classes=class_names,
                                                save_path=save_plots_path_prefix + "lstm_confusion_m_norm.png",
                                                normalize=True,
                                                title='helpers normalized confusion matrix ' + target)

    # calc accuracy
    accuracy = accuracy_score(y_test_df, y_pred_df)
    balanced_accuracy = balanced_accuracy_score(y_test_df, y_pred_df)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("Balanced Accuracy: %.2f%%" % (balanced_accuracy * 100.0))
    score_df = pd.DataFrame([])
    score_df['accuracy'] = accuracy
    score_df['balanced_accuracy'] = balanced_accuracy
    score_df.to_csv(save_plots_path_prefix + 'scores_' + target + '.csv',
                    index=False)

    # calc ROC and AUC
    roc_util.calc_and_plot_roc(y_test_path, y_pred_path, y_pred_proba_path,
                               save_path=save_plots_path_prefix + "lstm_roc_curves.png",
                               model_name_title="helpers " + target)
