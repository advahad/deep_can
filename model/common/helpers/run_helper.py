import pandas as pd
from sklearn import preprocessing
import os

from model.common.helpers import data_generator
from preprocessing.utils.partitioning import data_partitioning_util_gss as partitioning
from util import pickle_util
from pathlib import Path

from model.common.FCN.fcn_ae_model_european import FcnAeSingle

# helpers
# types: NN/ensemble
def generate_output_dirs_bkp(type, paths_conf_obj, general_conf_obj):
    output_dir = '../../../output/seq_' + str(paths_conf_obj.seq_size) + \
                 '_step_' + str(paths_conf_obj.step_size) + '/' + \
                 general_conf_obj.running.architecture_name + '/' + \
                 'scaler_' + general_conf_obj.data.scaler + '_scaling_type_' + general_conf_obj.data.scaling_type + \
                 '_pattern_features_' + str(general_conf_obj.data.pattern_features) + \
                 '_part_' + general_conf_obj.partitioning.type + \
                 '_frac_' + str(general_conf_obj.partitioning.sample_fraq) + \
                 '_under_sample_' + str(general_conf_obj.partitioning.under_sample) + '/'

    if str(general_conf_obj.data.gap) != '':
        output_dir = output_dir + 'gap_' + str(general_conf_obj.data.gap) + '/'
    else:
        output_dir = output_dir + 'gap_regular/'

    if general_conf_obj.features.keep_features == 'all':
        output_dir = output_dir + 'all_features/'
    else:
        output_dir = output_dir + 'partial_features/'

    # general data, plots, logs output dir
    output_dir_data = output_dir + 'data/'
    output_dir_plots = output_dir + 'plots/'
    output_dir_logs = output_dir + 'logs/'

    # train, test, val dirs
    train_sets_dir = output_dir_data + 'train_sets/'
    test_sets_dir = output_dir_data + 'test_sets/'
    val_sets_dir = output_dir_data + 'val_sets/'

    # tsfresh offline
    output_dir_data_tsfresh_offline = output_dir + 'data_tsfresh_offline/'
    train_sets_dir_tsfresh_offline = output_dir_data_tsfresh_offline + 'train_sets/'
    test_sets_dir_tsfresh_offline = output_dir_data_tsfresh_offline + 'test_sets/'
    val_sets_dir_tsfresh_offline = output_dir_data_tsfresh_offline + 'val_sets/'

    # tsfresh online
    output_dir_data_tsfresh_online = output_dir + 'data_tsfresh_online/'
    test_sets_dir_tsfresh_online = output_dir_data_tsfresh_online + 'test_sets/'

    # auto encoder
    output_dir_data_ae = output_dir + 'data_auto_encoder/'
    train_sets_dir_ae = output_dir_data_ae + 'train_sets/'
    test_sets_dir_ae = output_dir_data_ae + 'test_sets/'
    val_sets_dir_ae = output_dir_data_ae + 'val_sets/'

    # create paths
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir_data).mkdir(parents=True, exist_ok=True)
    Path(output_dir_plots).mkdir(parents=True, exist_ok=True)
    Path(output_dir_logs).mkdir(parents=True, exist_ok=True)

    Path(train_sets_dir).mkdir(parents=True, exist_ok=True)
    Path(test_sets_dir).mkdir(parents=True, exist_ok=True)
    Path(val_sets_dir).mkdir(parents=True, exist_ok=True)

    # tsfresh
    Path(output_dir_data_tsfresh_offline).mkdir(parents=True, exist_ok=True)
    Path(train_sets_dir_tsfresh_offline).mkdir(parents=True, exist_ok=True)
    Path(test_sets_dir_tsfresh_offline).mkdir(parents=True, exist_ok=True)
    Path(val_sets_dir_tsfresh_offline).mkdir(parents=True, exist_ok=True)
    Path(test_sets_dir_tsfresh_online).mkdir(parents=True, exist_ok=True)

    # auto encoder
    Path(output_dir_data_ae).mkdir(parents=True, exist_ok=True)
    Path(train_sets_dir_ae).mkdir(parents=True, exist_ok=True)
    Path(test_sets_dir_ae).mkdir(parents=True, exist_ok=True)
    Path(val_sets_dir_ae).mkdir(parents=True, exist_ok=True)

    print('output dir is: \n{}'.format(output_dir))

    return output_dir, output_dir_data, output_dir_logs, output_dir_plots, train_sets_dir, test_sets_dir, val_sets_dir, \
           output_dir_data_tsfresh_offline, train_sets_dir_tsfresh_offline, test_sets_dir_tsfresh_offline, \
           val_sets_dir_tsfresh_offline, test_sets_dir_tsfresh_online, output_dir_data_ae, train_sets_dir_ae, \
           test_sets_dir_ae, val_sets_dir_ae


def create_paths(paths):
    for folder_path in paths:
        Path(folder_path).mkdir(parents=True, exist_ok=True)


class OutputDirHandler:
    def __init__(self, paths_conf_obj, general_conf_obj, output_folder_name=None):
        self.generate_output_dirs(paths_conf_obj, general_conf_obj, output_folder_name)

    def generate_output_dirs(self, paths_conf_obj, general_conf_obj, output_folder_name=None):
        if output_folder_name == None:
            output_folder_name = 'output'

        output_dir = '../../../' + output_folder_name + '/seq_' + str(paths_conf_obj.seq_size) + \
                     '_step_' + str(paths_conf_obj.step_size) + '/' + \
                     general_conf_obj.running.architecture_name + '/' + \
                     'scaler_' + general_conf_obj.data.scaler + '_scaling_type_' + general_conf_obj.data.scaling_type + \
                     '_pattern_features_' + str(general_conf_obj.data.pattern_features) + \
                     '_part_' + general_conf_obj.partitioning.type + \
                     '_frac_' + str(general_conf_obj.partitioning.sample_fraq) + \
                     '_under_sample_' + str(general_conf_obj.partitioning.under_sample) + '/'



        if str(general_conf_obj.data.gap) != '':
            output_dir = output_dir + 'gap_' + str(general_conf_obj.data.gap) + '/'
        else:
            output_dir = output_dir + 'gap_regular/'

        if general_conf_obj.features.keep_features == 'all':
            output_dir = output_dir + 'all_features/'
        else:
            output_dir = output_dir + 'partial_features/'

        output_dir = '../../../' + output_folder_name
        print('output dir is: {}'.format(output_dir))
        self.output_dir = output_dir

        """
        NN folders with ae
        """
        # NN data, plots, logs output dir
        self.output_dir_nn = output_dir + 'nn/'
        self.output_dir_data_nn = self.output_dir_nn + 'data/'
        self.output_dir_plots_nn = self.output_dir_nn + 'plots/'
        self.output_dir_logs_nn = self.output_dir_nn + 'logs/'
        self.output_dir_model_nn = self.output_dir_nn + 'best_model/'

        # NN train, test, val dirs
        self.train_sets_dir_nn = self.output_dir_data_nn + 'train_sets/'
        self.test_sets_dir_nn = self.output_dir_data_nn + 'test_sets/'
        self.val_sets_dir_nn = self.output_dir_data_nn + 'val_sets/'

        create_paths(
            [self.output_dir_plots_nn, self.output_dir_logs_nn, self.output_dir_model_nn, self.train_sets_dir_nn,
             self.test_sets_dir_nn, self.val_sets_dir_nn])



        """
        NN folders without ae
        """
        # NN data, plots, logs output dir
        self.output_dir_nn_basic = output_dir + 'nn_basic/'
        self.output_dir_data_nn_basic = self.output_dir_nn_basic + 'data/'
        self.output_dir_plots_nn_basic = self.output_dir_nn_basic + 'plots/'
        self.output_dir_logs_nn_basic = self.output_dir_nn_basic + 'logs/'
        self.output_dir_model_nn_basic = self.output_dir_nn_basic + 'best_model/'

        # NN train, test, val dirs
        self.train_sets_dir_nn_basic = self.output_dir_data_nn_basic + 'train_sets/'
        self.test_sets_dir_nn_basic = self.output_dir_data_nn_basic + 'test_sets/'
        self.val_sets_dir_nn_basic = self.output_dir_data_nn_basic + 'val_sets/'

        create_paths(
            [self.output_dir_plots_nn_basic, self.output_dir_logs_nn_basic, self.output_dir_model_nn_basic,
             self.train_sets_dir_nn_basic,
             self.test_sets_dir_nn_basic, self.val_sets_dir_nn_basic])

        """
        xgboost folders
        """
        # xgboost
        self.output_dir_xgboost = self.output_dir + 'xgboost/'
        # xgboost offline
        self.output_dir_xgboost_offline = self.output_dir_xgboost + 'offline/'
        self.output_dir_data_xgboost_offline = self.output_dir_xgboost_offline + 'data/'
        self.output_dir_plots_xgboost_offline = self.output_dir_xgboost_offline + 'plots/'
        self.output_dir_logs_xgboost_offline = self.output_dir_xgboost_offline + 'logs/'
        self.output_dir_model_xgboost_offline = self.output_dir_xgboost_offline + 'best_model/'

        # xgboost_offline train, test, val dirs
        self.train_sets_dir_xgboost_offline = self.output_dir_data_xgboost_offline + 'train_sets/'
        self.test_sets_dir_xgboost_offline = self.output_dir_data_xgboost_offline + 'test_sets/'
        self.val_sets_dir_xgboost_offline = self.output_dir_data_xgboost_offline + 'val_sets/'

        create_paths([self.output_dir_plots_xgboost_offline, self.output_dir_logs_xgboost_offline,
                      self.output_dir_model_xgboost_offline,
                      self.train_sets_dir_xgboost_offline, self.test_sets_dir_xgboost_offline,
                      self.val_sets_dir_xgboost_offline])

        # xgboost online
        self.output_dir_xgboost_online = self.output_dir_xgboost + 'online/'
        self.output_dir_data_xgboost_online = self.output_dir_xgboost_online + 'data/'
        self.output_dir_plots_xgboost_online = self.output_dir_xgboost_online + 'plots/'
        self.output_dir_logs_xgboost_online = self.output_dir_xgboost_online + 'logs/'
        self.output_dir_model_xgboost_online = self.output_dir_xgboost_online + 'best_model/'

        # xgboost_online train, test, val dirs
        self.train_sets_dir_xgboost_online = self.output_dir_data_xgboost_online + 'train_sets/'
        self.test_sets_dir_xgboost_online = self.output_dir_data_xgboost_online + 'test_sets/'
        self.val_sets_dir_xgboost_online = self.output_dir_data_xgboost_online + 'val_sets/'

        create_paths([self.output_dir_plots_xgboost_online, self.output_dir_logs_xgboost_online,
                      self.output_dir_model_xgboost_online,
                      self.train_sets_dir_xgboost_online, self.test_sets_dir_xgboost_online,
                      self.val_sets_dir_xgboost_online])

        """
        ae folders
        """
        # ae data, plots, logs output dir
        self.output_dir_ae = output_dir + 'ae/'
        self.output_dir_data_ae = self.output_dir_ae + 'data/'
        self.output_dir_plots_ae = self.output_dir_ae + 'plots/'
        self.output_dir_logs_ae = self.output_dir_ae + 'logs/'
        self.output_dir_model_ae = self.output_dir_ae + 'best_model/'

        # ae train, test, val dirs
        self.train_sets_dir_ae = self.output_dir_data_ae + 'train_sets/'
        self.test_sets_dir_ae = self.output_dir_data_ae + 'test_sets/'
        self.val_sets_dir_ae = self.output_dir_data_ae + 'val_sets/'

        create_paths(
            [self.output_dir_plots_ae, self.output_dir_logs_ae, self.output_dir_model_ae, self.test_sets_dir_ae,
             self.train_sets_dir_ae, self.val_sets_dir_ae,
             self.test_sets_dir_ae, self.val_sets_dir_ae])

        """
        ensemble folders
        """
        # ae data, plots, logs output dir
        self.output_dir_ensemble = output_dir + 'ensemble/'
        self.output_dir_data_ensemble = self.output_dir_ensemble + 'data/'
        self.output_dir_plots_ensemble = self.output_dir_ensemble + 'plots/'
        self.output_dir_logs_ensemble = self.output_dir_ensemble + 'logs/'
        self.output_dir_model_ensemble = self.output_dir_ensemble + 'best_model/'

        # ae train, test, val dirs
        self.train_sets_dir_ensemble = self.output_dir_data_ensemble + 'train_sets/'
        self.test_sets_dir_ensemble = self.output_dir_data_ensemble + 'test_sets/'
        self.val_sets_dir_ensemble = self.output_dir_data_ensemble + 'val_sets/'

        create_paths(
            [self.output_dir_plots_ensemble, self.output_dir_logs_ensemble, self.output_dir_model_ensemble, self.test_sets_dir_ensemble,
             self.train_sets_dir_ensemble, self.val_sets_dir_ensemble,
             self.test_sets_dir_ensemble, self.val_sets_dir_ensemble])


def create_or_load_label_encoder(classes, generate_le=False, path_to_load=None):
    if generate_le == False:
        print("loading le for target")
        le = pickle_util.load_obj(path_to_load)
    else:
        print("generating le for target")
        le = preprocessing.LabelEncoder()
        le.fit(pd.DataFrame(list(classes)))
        pickle_util.save_obj(le, path_to_load)
    return le


def get_classes(meta_path, target):
    meta_df = pd.read_csv(meta_path)
    meta_df_classes = meta_df[target].unique()
    all_classes = list(meta_df_classes)
    print('all classes are: {}'.format(all_classes))
    return all_classes


def get_scaler(samples_files, samples_path, non_features_cols, path, try_load=True):
    if try_load and os.path.isfile(path):
        scaler = pickle_util.load_obj(path)
    else:
        scaler = preprocessing.MinMaxScaler()
        for idx, file_name in enumerate(samples_files):
            if idx % 10000 == 0:
                print('scaler idx {}/{}'.format(idx, len(samples_files)))
            sample_df = pd.read_csv(samples_path + file_name)
            scaler.partial_fit(sample_df.drop(non_features_cols, axis=1))
        pickle_util.save_obj(scaler, path)
    return scaler


def calc_num_of_features(samples_files, samples_path, non_features_cols, remove_features, keep_features):
    features = get_features_names(samples_files, samples_path, non_features_cols, remove_features, keep_features)
    print('features are: {}'.format(features))
    return len(features)


# TODO: change to wllk
# for root, dirs, files in os.walk(".", topdown=False):
#    for name in files:
#       print(os.path.join(root, name))
def get_features_names(samples_files, samples_path, non_features_cols, remove_features, keep_features):
    sample_file = samples_files[0]
    sample_df = pd.read_csv(samples_path + sample_file)
    sample_df = sample_df.drop(non_features_cols, axis=1)
    sample_df = sample_df.drop(remove_features, axis=1)
    if keep_features != 'all':
        sample_df = sample_df[keep_features]
    return sample_df.columns


def generate_x_y_sets(sets_path, partition_files_names, x_file_name, y_file_name, y_encoded_file_name, samples_path,
                      non_features_cols, remove_features, keep_features,
                      seq_size, label_encoder):
    print('\ngenerating sets')
    x, y_encoded = data_generator.batch_generation(samples_path, partition_files_names, non_features_cols,
                                                   remove_features, keep_features,
                                                   seq_size, label_encoder)
    # if is_auto_encoder == False:
    y = label_encoder.inverse_transform(y_encoded)
    # save as csv
    pd.DataFrame(y).to_csv(sets_path + y_file_name, index=False)
    pd.DataFrame(y_encoded).to_csv(sets_path + y_encoded_file_name, index=False)
    # else:
    #     y = x
    #     # save as pickle due to dimensionality
    #     pickle_util.save_obj(y, sets_path + y_file_name)
    #     pickle_util.save_obj(y_encoded, sets_path + y_encoded_file_name)

    pickle_util.save_obj(x, sets_path + x_file_name)

    print("done generating sets")
    return x, y, y_encoded


def generate_x_y_sets_ts_fresh(sets_path, partition_files_names, x_file_name, y_file_name, y_encoded_file_name,
                               samples_path,
                               non_features_cols, remove_features, keep_features,
                               seq_size, label_encoder):
    print('\ngenerating sets')
    x, y_encoded = data_generator.batch_generation(samples_path, partition_files_names, non_features_cols,
                                                   remove_features, keep_features,
                                                   seq_size, label_encoder, should_reshape=False)
    y = label_encoder.inverse_transform(y_encoded)

    pickle_util.save_obj(x, sets_path + x_file_name)
    pd.DataFrame(y).to_csv(sets_path + y_file_name, index=False)
    pd.DataFrame(y_encoded).to_csv(sets_path + y_encoded_file_name, index=False)
    print("done generating sets")
    return x, y, y_encoded


def get_partitioning(partitioning_obj_path, keep_classes_list, meta_data_file_name_prefix, meta_data_dir_path,
                     partitioning_type, target,
                     sample_fraq,
                     should_under_sample,
                     under_sample_sets,
                     test_size,
                     val_size,
                     try_load,
                     over_under_type,
                     filter_rpm=False):
    test_size = test_size
    val_size = val_size
    if try_load == True:
        try:
            partition = pickle_util.load_obj(partitioning_obj_path)
            print("load partitioning from file successfully")
        except Exception as e:
            print("couldn't load, preparing partitioning")
            print(e)
            if partitioning_type == 'time':
                partition = partitioning.create_partitioning_based_on_time(meta_data_dir_path,
                                                                           meta_data_file_name_prefix, target,
                                                                           sample_fraq)
                partition = partitioning.create_partitioning(meta_data_file_name_prefix,
                                                             meta_data_dir_path,
                                                             partitioning_type,
                                                             target_col=target,
                                                             test_set_size=test_size,
                                                             val_set_size=val_size,
                                                             results_path=partitioning_obj_path,
                                                             keep_classes_list=keep_classes_list,
                                                             should_under_sample=should_under_sample,
                                                             under_sample_sets=under_sample_sets,
                                                             sample_frac=sample_fraq,
                                                             over_under_type=over_under_type,
                                                             filter_rpm=filter_rpm)

            else:
                pass

    else:
        print("preparing partitioning")
        if partitioning_type == 'time':
            partition = partitioning.create_partitioning_based_on_time(meta_data_dir_path,
                                                                       meta_data_file_name_prefix, target,
                                                                       sample_fraq)
        else:
            partition = partitioning.create_partitioning(meta_data_file_name_prefix,
                                                         meta_data_dir_path,
                                                         partitioning_type,
                                                         target_col=target,
                                                         test_set_size=test_size,
                                                         val_set_size=val_size,
                                                         results_path=partitioning_obj_path,
                                                         keep_classes_list=keep_classes_list,
                                                         should_under_sample=should_under_sample,
                                                         under_sample_sets=under_sample_sets,
                                                         sample_frac=sample_fraq,
                                                         over_under_type=over_under_type,
                                                         filter_rpm=filter_rpm)

    # TODO-b: bug fix UnboundLocalError: local variable 'partition' referenced before assignment,when new output path and load == True
    return partition

def load_fcn_model(output_dir_model, output_dir_model_ae, paths_conf_obj, general_conf_obj):
    model = FcnAeSingle(model_path=output_dir_model + paths_conf_obj.best_model_name,
                        initialize_weights_with_model=True,
                        ae_best_model_path=output_dir_model_ae + paths_conf_obj.best_model_name,
                        concat_ae=general_conf_obj.running.concat_ae)
    return model


def create_class_to_idx_mapping(general_conf_obj, paths_conf_obj, output_handler):
    keep_classes_list = get_classes(paths_conf_obj.meta_data_dir_path + paths_conf_obj.meta_data_file_name,
                                               paths_conf_obj.target)


    label_encoder = create_or_load_label_encoder(keep_classes_list,
                                                            generate_le=general_conf_obj.running.generate_target_label_encoder,
                                                            path_to_load=output_handler.output_dir +
                                                                         paths_conf_obj.target_label_encoder_file_name)
    class_to_idx_dict = {}
    for idx, class_name in enumerate(label_encoder.classes_):
        class_to_idx_dict[class_name] = idx
    return class_to_idx_dict

from sklearn.utils import class_weight
import numpy as np
def calc_class_weights(y_train):
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train['0'].values)
    class_labels = np.unique(y_train)
    class_weights_dict = {}
    for idx, label in enumerate(class_labels):
        class_weights_dict[label] = class_weights[idx]
    return class_weights_dict

