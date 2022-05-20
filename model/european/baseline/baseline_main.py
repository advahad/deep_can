import numpy as np
import pandas as pd
import tsfresh as ts
from sklearn.metrics import accuracy_score
from tsfresh import extract_features, select_features
from tsfresh import extract_relevant_features
from tsfresh.feature_extraction.settings import MinimalFCParameters, EfficientFCParameters
from xgboost import XGBClassifier

from config_classes.general_config import GeneralConfig
from config_classes.paths_config import PathsConfig
from evaluation import roc_util, confusion_matrix_util
from model.common.helpers import run_helper
from preprocessing.dataset.european import european_raw_preprocessing_NN_and_tsfresh_one_file as preprocess_util
from util import pickle_util
from sklearn.metrics import balanced_accuracy_score
from util import logs_util


# load configuration
general_conf_path = '../config/general_config_full_run.json'
paths_conf_path = '../config/paths_config_3_cls.json'
general_conf_obj = GeneralConfig(general_conf_path)
paths_conf_obj = PathsConfig(paths_conf_path)

output_handler = run_helper.OutputDirHandler(paths_conf_obj, general_conf_obj)


def load_data(path):
    data = pd.read_csv(path)
    # filling missing values
    for col in data.columns:
        data[col].interpolate(method='linear', inplace=True)
    return data


def save_kind_params(extracted_features_df, kind_to_fc_parameters_path):
    kind_to_fc_parameters = ts.feature_extraction.settings.from_columns(extracted_features_df)
    pickle_util.save_obj(kind_to_fc_parameters, kind_to_fc_parameters_path)


# extraction types: extract_only/ extract_select/ kind_extract
def extract_baseline_features(type, data_path, timeseries_extracted_df_path, y_path, kind_to_fc_parameters_path,
                              target_col,
                              series_col, sort_col, partition_type, drop_cols, settings, save_meta=True,
                              cols_order=None):
    data = load_data(data_path)
    meta_data = pd.read_csv(META_DATA_DIR_PATH + META_DATA_FILE_NAME)

    y = data.groupby([series_col]).first()[target_col]
    y.to_csv(y_path, index=False)

    timeseries = data.drop(drop_cols, axis=1)

    print("performing {}:".format(type))
    if type == 'extract_select':
        features_filtered_direct = extract_relevant_features(
            timeseries, y, column_id=series_col, column_sort=sort_col, default_fc_parameters=settings)
        extracted_features_df = features_filtered_direct
        save_kind_params(extracted_features_df, kind_to_fc_parameters_path)
    elif type == 'extract_only':
        extracted_features_df = extract_features(timeseries_container=timeseries, default_fc_parameters=settings,
                                                 column_id=series_col, column_sort=sort_col, chunksize=100)
    elif type == 'kind_extract':
        kind_to_fc_parameters = pickle_util.load_obj(kind_to_fc_parameters_path)
        extracted_features_df = extract_features(timeseries, column_id=series_col, column_sort=sort_col,
                                                 kind_to_fc_parameters=kind_to_fc_parameters)
        if cols_order is not None:
            extracted_features_df = extracted_features_df.loc[:, cols_order]

    else:
        print('please selection type of tsfresh operation')
        return

    if save_meta:
        print('adding meta features')
        extracted_features_df[DATA_ID] = meta_data[DATA_ID]
        extracted_features_df[partition_type] = meta_data[partition_type]
        extracted_features_df[target_col] = meta_data[target_col]
    print('saving extracted df')
    extracted_features_df.to_csv(timeseries_extracted_df_path, index=False)
    return extracted_features_df


def select_baseline_features(X_train_path, y_train_path, kind_to_fc_parameters_path, selected_features_names_path,
                             target_col):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    selected_features_df = select_features(X=X_train, y=y_train[target_col])
    save_kind_params(selected_features_df, kind_to_fc_parameters_path)
    selected_features = selected_features_df.columns
    selected_features_as_list = list(selected_features.values)
    pickle_util.save_obj(selected_features_as_list, selected_features_names_path)


def generate_train_test_sets(extracted_df, existing_partitioning, partitioning_type,
                             x_train_path, x_test_path, x_val_path,
                             y_train_path, y_test_path, y_val_path):
    train_set_part_col_ids = existing_partitioning['train_partitioning_col']
    test_set_part_col_ids = existing_partitioning['test_partitioning_col']
    val_set_part_col_ids = existing_partitioning['val_partitioning_col']

    # y_train = data.groupby(['series_num']).first()[label_col]

    # df_to_split = pd.read_csv(extracted_features_path)
    df_to_split = extracted_df
    train_set = df_to_split.loc[df_to_split[partitioning_type].isin(train_set_part_col_ids)]
    test_set = df_to_split.loc[df_to_split[partitioning_type].isin(test_set_part_col_ids)]
    val_set = df_to_split.loc[df_to_split[partitioning_type].isin(val_set_part_col_ids)]

    drop_list = [DATA_ID, partitioning_type, target_col]
    X_train = train_set.drop(drop_list, axis=1)
    X_test = test_set.drop(drop_list, axis=1)
    X_val = val_set.drop(drop_list, axis=1)

    y_train = train_set[target_col]
    y_test = test_set[target_col]
    y_val = val_set[target_col]

    X_train.to_csv(x_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    X_val.to_csv(x_val_path, index=False)

    y_train.to_csv(y_train_path, index=False)
    y_test.to_csv(y_test_path, index=False)
    y_val.to_csv(y_val_path, index=False)

    return X_train, X_test, X_val, y_train, y_test, y_val


def filter_dfs_and_save(selected_features_file_path, X_train_path, X_test_path, X_val_path):
    selected_features = pickle_util.load_obj(selected_features_file_path)

    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    X_val = pd.read_csv(X_val_path)

    # save only selected features
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    X_val = X_val[selected_features]

    X_train.to_csv(X_train_path, index=False)
    X_test.to_csv(X_test_path, index=False)
    X_val.to_csv(X_val_path, index=False)
    return X_train, X_test, X_val


def train_xgboost(X_train_path, y_train_path, model_path):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    model = XGBClassifier(max_depth=15, n_estimators=200)

    model.fit(X_train, y_train)
    # save model to file
    pickle_util.save_obj(model, model_path)

def get_feature_importance(X_test_path, model_path, top_num):
    bst_model = pickle_util.load_obj(model_path)

    X_test = pd.read_csv(X_test_path)

    # TODO: separate function of stats
    features_names = X_test.columns
    # ts_df_filtered = pd.read_csv(EXTRACTED_FEATURES_PATH)
    print(np.shape(features_names))
    features_importance = bst_model.feature_importances_
    print(np.shape(features_importance))
    features_names_and_importance = pd.DataFrame([])
    features_names_and_importance['feature_name'] = features_names
    features_names_and_importance['feature_impor'] = features_importance
    features_names_and_importance.sort_values(by='feature_impor', ascending=False, inplace=True)
    return features_names_and_importance['feature_name'].head(top_num).values

def predict(model_path, X_test_path, y_pred_path, y_pred_proba_path):
    # load model from file
    bst_model = pickle_util.load_obj(model_path)

    X_test = pd.read_csv(X_test_path)

    # TODO: separate function of stats
    features_names = X_test.columns
    # ts_df_filtered = pd.read_csv(EXTRACTED_FEATURES_PATH)
    print(np.shape(features_names))
    features_importance = bst_model.feature_importances_
    print(np.shape(features_importance))
    features_names_and_importance = pd.DataFrame([])
    features_names_and_importance['feature_name'] = features_names
    features_names_and_importance['feature_impor'] = features_importance
    features_names_and_importance.sort_values(by='feature_impor', ascending=False, inplace=True)
    print(features_names_and_importance.head(200))
    print('save feature importance in: {}'.format(output_handler.output_dir_data_xgboost_offline + 'feature_importance.csv'))
    features_names_and_importance.to_csv(output_handler.output_dir_data_xgboost_offline + 'feature_importance.csv')
    # features_names_and_importance = features_names_and_importance['feature_name'].str.split('_', expand=True)
    feature_name_and_stat = \
        features_names_and_importance['feature_name'].apply(lambda x: pd.Series(str(x).split(sep="_", maxsplit=1)))

    features_names_and_importance['feature_name'] = feature_name_and_stat[0]
    features_names_and_importance['feature_stat'] = feature_name_and_stat[1]

    features_names_and_importance = features_names_and_importance.sort_values(['feature_impor'], ascending=False)
    features_names_and_importance = features_names_and_importance.loc[
        features_names_and_importance['feature_impor'] != 0]
    # print(features_names_and_importance.shape)

    y_pred = bst_model.predict(X_test)
    y_pred_proba = bst_model.predict_proba(X_test)
    # save results
    # pd.DataFrame(y_test).to_csv(y_test_path, index=False)
    pd.DataFrame(y_pred).to_csv(y_pred_path, index=False)
    pd.DataFrame(y_pred_proba).to_csv(y_pred_proba_path, index=False)
    return y_pred_proba


# TODO:refactor according to output path
def evaluate(y_test_path, y_pred_path, y_pred_proba_path, class_names, plots_path_prefix, duration, logs_dir):
    y_test_encoded = pd.read_csv(y_test_path)
    y_pred_encoded = pd.read_csv(y_pred_path)
    y_pred_proba = pd.read_csv(y_pred_proba_path)


    # Plot non-normalized confusion matrix
    confusion_matrix_util.plot_confusion_matrix(y_test_path, y_pred_path, classes=class_names,
                                                save_path=plots_path_prefix + "baseline_confusion_m.png",
                                                title='Baseline confusion matrix, without normalization')

    # Plot normalized confusion matrix
    confusion_matrix_util.plot_confusion_matrix(y_test_path, y_pred_path, classes=class_names,
                                                save_path=plots_path_prefix + "baseline_confusion_m_norm.png",
                                                normalize=True,
                                                title='Baseline normalized confusion matrix')

    # calc accuracy
    accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
    balanced_accuracy = balanced_accuracy_score(y_test_encoded, y_pred_encoded)


    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("balanced_accuracy: %.2f%%" % (balanced_accuracy * 100.0))
    # balanced_acc = metrics.balanced_accuracy_score(y_test_encoded, y_pred_encoded)
    # print("Balanced Accuracy: %.2f%%" % (balanced_acc * 100.0))
    score_df = pd.DataFrame([accuracy, balanced_accuracy]).T
    score_df.columns = ["accuracy", "balanced_acc"]
    score_df.to_csv(plots_path_prefix + '/scores.csv', index=False)

    # calc ROC and AUC
    roc_util.calc_and_plot_roc(y_test_path, y_pred_path, y_pred_proba_path,
                               save_path=plots_path_prefix + "baseline_roc_curves.png",
                               model_name_title="Baseline")
    # TODO:May change y_pred_proba.vlaues to inne value type check
    logs_util.save_logs(logs_dir, pd.DataFrame([]), y_pred_encoded, y_pred_proba.values,
                        y_test_encoded, duration)


# general conf
SEQ = paths_conf_obj.seq_size
DATA_ID = 'sample_file_name'
target_col = paths_conf_obj.target
SERIES_COL = 'series_num'
SORT_COL = 'fid_x'
DROP_COLS = [target_col, 'sid', 'vid', 't_x']
DROP_COLS = [target_col, 'sid', 'vid', 't_x', 'sub_sid', 'sample_file_name', general_conf_obj.partitioning.type]

# offline configuration
STEP_OFFLINE_TRAIN = general_conf_obj.tsfresh_offline.step
META_DATA_DIR_PATH = '../../../results/raw_preprocessing/windows/seq_{}_step_{}/meta_merged_labels_3_cls/'.format(SEQ,
                                                                                                                  STEP_OFFLINE_TRAIN)
META_DATA_FILE_NAME = general_conf_obj.tsfresh_offline.meta_date_file_name
OFFLINE_TRAINING_ONE_FILE_DATA_PATH = \
    '../../../results/raw_preprocessing/windows/seq_{}_step_{}/samples_one_file/samples_one_file_dict.csv'.format(SEQ,
                                                                                                                  STEP_OFFLINE_TRAIN)
EXTRACTED_FEATURES_DF_PATH = '{}features_filtered_direct_european_seq_{}_step_{}.csv'.format(
    output_handler.output_dir_data_xgboost_offline, SEQ,
    STEP_OFFLINE_TRAIN)


# online ensemble configuration

STEP = paths_conf_obj.step_size

settings = EfficientFCParameters()
settings = MinimalFCParameters()

"""
1- extract features using tsfresh on smaller data set for running time saving
2- train test split on the feature extracted set
3- feature selection on the train data only
"""


def ts_fresh_offline_extract_and_select_features(partition_path, partition_type, X_train_path, X_test_path, X_val_path,
                                                 y_train_path, y_test_path, y_val_path, selected_features_file_path):
    # performing feature extraction
    partition = pickle_util.load_obj(partition_path)
    extracted_df = extract_baseline_features(type='extract_only',
                                             data_path=OFFLINE_TRAINING_ONE_FILE_DATA_PATH,
                                             timeseries_extracted_df_path=EXTRACTED_FEATURES_DF_PATH,
                                             y_path=output_dir_data_tsfresh_offlline + 'y_selecting_model.csv',
                                             kind_to_fc_parameters_path=None,
                                             target_col=target_col,
                                             series_col=SERIES_COL,
                                             sort_col=SORT_COL,
                                             partition_type=partition_type,
                                             drop_cols=DROP_COLS,
                                             settings=settings)

    # # performing feature selection
    # generating sets according to partition
    generate_train_test_sets(extracted_df=extracted_df,
                             existing_partitioning=partition,
                             partitioning_type=general_conf_obj.partitioning.type,
                             x_train_path=X_train_path,
                             x_test_path=X_test_path,
                             x_val_path=X_val_path,
                             y_train_path=y_train_path,
                             y_test_path=y_test_path,
                             y_val_path=y_val_path)

    # selecting features only by train examples
    select_baseline_features(X_train_path, y_train_path,
                             output_dir_data_tsfresh_offlline + general_conf_obj.tsfresh_offline.kind_params_file_name,
                             selected_features_file_path, target_col)
    filter_dfs_and_save(selected_features_file_path=selected_features_file_path, X_train_path=X_train_path,
                        X_test_path=X_test_path, X_val_path=X_val_path)


def create_aggregate_features(samples_dir_path, file_names_list, one_file_dir_path, one_file_name, y_path,
                              kind_params_path,
                              timeseries_extracted_df_path, cols_order=None, save_meta=False, generate_one_file=True):
    if generate_one_file:
        preprocess_util.create_one_windows_file(windows_path=samples_dir_path, one_file_dir_path=one_file_dir_path,
                                                one_file_name=one_file_name, files_names_list=file_names_list)

    extract_baseline_features('kind_extract',
                              data_path=one_file_dir_path + one_file_name,
                              timeseries_extracted_df_path=timeseries_extracted_df_path,
                              y_path=y_path,
                              kind_to_fc_parameters_path=kind_params_path,
                              target_col=target_col,
                              series_col=SERIES_COL,
                              sort_col=SORT_COL,
                              partition_type=general_conf_obj.partitioning.type,
                              drop_cols=DROP_COLS,
                              settings=settings,
                              save_meta=save_meta,
                              cols_order=cols_order)


def get_xgboost_cols_order(path):
    xgboost_train_cols_order = pd.read_csv(path, nrows=1).columns
    return xgboost_train_cols_order


# simulate flow of offline flow: features extraction, selection and model training and online flow: recieving teste set
if __name__ == '__main__':

    # mapping from old configuration to new
    output_dir_data_tsfresh_offlline = output_handler.output_dir_data_xgboost_offline
    output_dir_data = output_handler.output_dir
    train_sets_dir_tsfresh_offline = output_handler.train_sets_dir_xgboost_offline
    test_sets_dir_tsfresh_offline = output_handler.test_sets_dir_xgboost_offline
    val_sets_dir_tsfresh_offline = output_handler.val_sets_dir_xgboost_offline
    test_sets_dir_tsfresh_online = output_handler.test_sets_dir_xgboost_online


    keep_classes_list = run_helper.get_classes(paths_conf_obj.meta_data_dir_path + paths_conf_obj.meta_data_file_name,
                                               paths_conf_obj.target)

    partition = run_helper.get_partitioning(output_handler.output_dir + paths_conf_obj.partitioning_obj_file_name,
                                            keep_classes_list,
                                            paths_conf_obj.meta_data_file_prefix, paths_conf_obj.meta_data_dir_path,
                                            general_conf_obj.partitioning.type, paths_conf_obj.target,
                                            general_conf_obj.partitioning.sample_fraq,
                                            should_under_sample=general_conf_obj.partitioning.under_sample,
                                            under_sample_sets=general_conf_obj.partitioning.under_sample_sets,
                                            test_size=general_conf_obj.partitioning.test_set_size,
                                            val_size=general_conf_obj.partitioning.vat_set_size,
                                            try_load=general_conf_obj.partitioning.load,
                                            over_under_type='under')
    """
    offline flow: 
    1- extract features using tsfresh on smaller data set for running time saving
    
    2- train test split on the feature extracted set
    3- feature selection on the train data only
    """
    import time
    start = time.time()
    ts_fresh_offline_extract_and_select_features(
        partition_path=output_dir_data + paths_conf_obj.partitioning_obj_file_name,
        partition_type=general_conf_obj.partitioning.type,
        X_train_path=train_sets_dir_tsfresh_offline + general_conf_obj.tsfresh_offline.x_train_file_name,
        X_test_path=test_sets_dir_tsfresh_offline + general_conf_obj.tsfresh_offline.x_test_file_name,
        X_val_path=val_sets_dir_tsfresh_offline + general_conf_obj.tsfresh_offline.x_val_file_name,
        y_train_path=train_sets_dir_tsfresh_offline + general_conf_obj.tsfresh_offline.y_train_file_name,
        y_test_path=test_sets_dir_tsfresh_offline + general_conf_obj.tsfresh_offline.y_test_file_name,
        y_val_path=val_sets_dir_tsfresh_offline + general_conf_obj.tsfresh_offline.y_val_file_name,
        selected_features_file_path=
        output_dir_data_tsfresh_offlline + general_conf_obj.tsfresh_offline.selected_features_file_name)
    duration = time.time() - start
    print('duration is: {}'.format(duration))
    # train xgboost model
    train_xgboost(X_train_path=train_sets_dir_tsfresh_offline + general_conf_obj.tsfresh_offline.x_train_file_name,
                  y_train_path=train_sets_dir_tsfresh_offline + general_conf_obj.tsfresh_offline.y_train_file_name,
                  model_path=output_dir_data_tsfresh_offlline + general_conf_obj.tsfresh_offline.xgboost_model_name)

    # # # TODO:add evaluate
    """
    Online flow:
    """
    # parameter from NN
    # partitioning = pickle_util.load_obj(output_dir_data + paths_conf_obj.partitioning_obj_file_name)


    # online_raw_samples_main_path = '../../../results/raw_preprocessing/windows/seq_{}_step_{}/'.format(SEQ, STEP)

    # parameter from NN
    xgboost_train_cols_order = get_xgboost_cols_order(
        train_sets_dir_tsfresh_offline + general_conf_obj.tsfresh_offline.x_train_file_name)

    # create aggregated features from NN test set
    create_aggregate_features(samples_dir_path=paths_conf_obj.ts_fresh_samples_path,
                              file_names_list=partition['test'],
                              one_file_dir_path=test_sets_dir_tsfresh_online,
                              one_file_name=general_conf_obj.tsfresh_online.x_NN_one_file_name,
                              y_path=test_sets_dir_tsfresh_online + general_conf_obj.tsfresh_online.y_NN_tsfreshed_file_name,
                              kind_params_path=output_dir_data_tsfresh_offlline + general_conf_obj.tsfresh_offline.kind_params_file_name,
                              timeseries_extracted_df_path=test_sets_dir_tsfresh_online +
                                                           general_conf_obj.tsfresh_online.x_NN_tsfreshed_file_name,
                              save_meta=False,
                              generate_one_file=general_conf_obj.running.generate_tsfresh_test_one_file,
                              cols_order=xgboost_train_cols_order)

    predict(model_path=output_dir_data_tsfresh_offlline + general_conf_obj.tsfresh_offline.xgboost_model_name,
            X_test_path=test_sets_dir_tsfresh_online + general_conf_obj.tsfresh_online.x_NN_tsfreshed_file_name,
            y_pred_path=test_sets_dir_tsfresh_online + general_conf_obj.tsfresh_online.y_pred_file_name,
            y_pred_proba_path=test_sets_dir_tsfresh_offline + general_conf_obj.tsfresh_online.y_pred_proba_file_name)

    # parameter from NN
    keep_classes_list = run_helper.get_classes(paths_conf_obj.meta_data_dir_path + paths_conf_obj.meta_data_file_name,
                                               paths_conf_obj.target)

    # TODO: refactor base on new output dir
    evaluate(y_test_path=test_sets_dir_tsfresh_online + general_conf_obj.tsfresh_online.y_NN_tsfreshed_file_name,
             y_pred_path=test_sets_dir_tsfresh_online + general_conf_obj.tsfresh_online.y_pred_file_name,
             y_pred_proba_path=test_sets_dir_tsfresh_offline + general_conf_obj.tsfresh_online.y_pred_proba_file_name,
             class_names=keep_classes_list,
             plots_path_prefix=output_handler.output_dir_plots_xgboost_online)
