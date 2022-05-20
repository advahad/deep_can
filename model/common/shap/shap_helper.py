import matplotlib.pyplot as plt
import pandas as pd
import shap

from config.general_config import GeneralConfig
from util import pickle_util

general_conf_path = '../../european/config/general_config.json'

general_conf_obj = GeneralConfig(general_conf_path)


def get_backgroud(paths_conf_obj, output_handler, test_sets_dir, is_correct_preds, is_tree=False):
    y_test_predictions = pd.read_csv(test_sets_dir + paths_conf_obj.y_test_pred_file_name)
    y_test_predictions.columns = ['pred']

    if is_tree:
        y_test_true = pd.read_csv(
            output_handler.test_sets_dir_xgboost_online + general_conf_obj.tsfresh_online.y_NN_tsfreshed_file_name)
    else:
        y_test_true = pd.read_csv(output_handler.test_sets_dir_nn + paths_conf_obj.y_test_file_name)

    y_test_true.columns = ['true']

    y_test_merged = pd.concat([y_test_true, y_test_predictions], axis=1)

    if is_correct_preds:
        y_test_TP_TN = y_test_merged.loc[y_test_merged['true'] == y_test_merged['pred']]
        y_slice = y_test_TP_TN
    else:
        y_test_FP_FN = y_test_merged.loc[y_test_merged['true'] != y_test_merged['pred']]
        y_slice = y_test_FP_FN

    slice_indices = y_slice.index

    if is_tree:
        X_test = pd.read_csv(test_sets_dir + general_conf_obj.tsfresh_online.x_NN_tsfreshed_file_name)
        X_test_slice = X_test.iloc[slice_indices]
    else:
        X_test = pickle_util.load_obj(output_handler.test_sets_dir_nn + paths_conf_obj.x_test_file_name)
        X_test_slice = X_test[slice_indices]

    # pickle_util.save_obj(X_test_slice, paths_conf_obj.x_test_correct_preds_path)
    y_slice.reset_index(inplace=True, drop=True)
    # y_slice.to_csv(paths_conf_obj.y_test_correct_preds_path, index=False)
    return X_test_slice, y_slice


def get_background_per_original_and_predicted_class(paths_conf_obj, output_handler, test_sets_dir, original_class,
                                                    predicted_class, is_correct_preds, is_tree=False):
    X_slice, y_slice = get_backgroud(paths_conf_obj, output_handler, test_sets_dir, is_correct_preds, is_tree)

    y_filtered_per_class = y_slice.loc[(y_slice['true'] == original_class) & (y_slice['pred'] == predicted_class)]
    y_filtered_per_class_indices = y_filtered_per_class.index

    if is_tree:
        X_filtered_per_class = X_slice.iloc[y_filtered_per_class_indices]
    else:
        X_filtered_per_class = X_slice[y_filtered_per_class_indices]


    y_filtered_per_class.reset_index(inplace=True, drop=True)
    return X_filtered_per_class, y_filtered_per_class


def plot_bar(shap_values, test_values, features, class_names, plot_title, max_display, ds, show=True):
    plt.yticks(rotation=50)
    # plt.tight_layout()

    shap.summary_plot(shap_values, test_values, feature_names=features,
                      title=plot_title,
                      show=show,
                      plot_type='bar',
                      max_display=max_display,
                      plot_size=(11, 10),
                      class_names=class_names
                      )
    plt.savefig('plots/{}/{}_bar.png'.format(ds, plot_title))
    plt.clf()


def plot_dot(shap_values, test_values, features, class_names, plot_title, max_display, ds, show=True):
    plt.yticks(rotation=50)
    shap.summary_plot(shap_values, test_values, feature_names=features,
                      title=plot_title,
                      show=show,
                      plot_type='dot',
                      max_display=max_display,
                      plot_size=(13, 10)
                      )

    plt.savefig('plots/{}/{}_dots.png'.format(ds, plot_title))
    plt.clf()
