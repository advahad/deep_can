import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, roc_auc_score, plot_roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
# matplotlib.use('agg')
from evaluation import roc_util

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'


def plot_epochs_metric(hist, file_name, metric='loss'):

    plt.figure()
    plt.plot(hist[metric])
    plt.plot(hist['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.show()
    plt.close()

def calc_3d_mse(y_test, y_pred):
    total_mse = 0
    for i in range(0, y_test.shape[0]):
        sample_mse = mean_squared_error(y_test[i], y_pred[i])
        total_mse = total_mse + sample_mse

    mean_total_mse = total_mse / y_test.shape[0]
    print('mse is: {}'.format(mean_total_mse))
    return mean_total_mse

def calculate_metrics(y_true, y_pred, y_pred_proba, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    precision = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)
    res['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    res['precision'] = precision

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    recall = recall_score(y_true, y_pred, average='macro')
    res['recall'] = recall

    from sklearn.metrics import auc


    # res['roc_auc_score'] = roc_auc_score(y_true, y_pred_proba)
    # res['precision_recall_curve'] = precision_recall_curve(y_pred)

    # precision, recall, _ = precision_recall_curve(y_true, pos_probs)
    # auc_score = auc(recall, precision)
    # res['auc_score'] = auc_score
    res['duration'] = duration

    auc_dict = roc_util.calc_auc_roc(y_true, y_pred, y_pred_proba)
    for key, auc_val in auc_dict.items():
        res[key] = auc_val


    pr_acore = roc_util.precision_recall_graph(y_true, y_pred, y_pred_proba)

    res['pr_auc'] = pr_acore

    return res

def calculate_metrics_ae(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 2), dtype=np.float), index=[0],
                       columns=['mse', 'duration'])
    res['mse'] = calc_3d_mse(y_true, y_pred)
    res['duration'] = duration

    return res


def save_logs(output_directory, hist_df, y_pred, y_pred_proba, y_true, duration, lr=True, y_true_val=None, y_pred_val=None):
    print("saving output logs")
    # hist_df = pd.DataFrame(hist_path.history)

    # hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, y_pred_proba, duration, y_true_val, y_pred_val)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)
    if hist_df.empty == False:
        index_best_model = hist_df['loss'].idxmin()
        row_best_model = hist_df.loc[index_best_model]

        df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                     columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                              'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

        df_best_model['best_model_train_loss'] = row_best_model['loss']
        df_best_model['best_model_val_loss'] = row_best_model['val_loss']
        df_best_model['best_model_train_acc'] = row_best_model['accuracy']
        df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
        if lr == True:
            df_best_model['best_model_learning_rate'] = row_best_model['lr']
        df_best_model['best_model_nb_epoch'] = index_best_model

        df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

        # for FCN there is no hyperparameters fine tuning - everything is static in code

        # plot losses
        plot_epochs_metric(hist_df, output_directory + 'epochs_loss.png')

    return df_metrics

def save_logs_ae(output_directory, hist_df, y_pred, y_true, duration, lr=True, y_true_val=None, y_pred_val=None):
    print("saving output logs")
    # hist_df = pd.DataFrame(hist_df.history)
    # hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics_ae(y_true, y_pred, duration, y_true_val, y_pred_val)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']

    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code

    # plot losses
    plot_epochs_metric(hist_df, output_directory + 'epochs_loss.png')

    return df_metrics