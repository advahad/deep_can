from config_classes.general_config import GeneralConfig
from config_classes.paths_config import PathsConfig

from model.common.helpers import run_helper
import numpy as np

import pandas as pd
from util import pickle_util

from matplotlib import pyplot as plt

def load_configs_european():
    general_conf_path = '../../european/config/general_config.json'
    paths_conf_path = '../../european/config/paths_config_3_cls.json'

    general_conf_obj = GeneralConfig(general_conf_path)
    paths_conf_obj = PathsConfig(paths_conf_path)

    return general_conf_obj, paths_conf_obj

def load_configs_shrp():
    general_conf_path = '../../SHRP2/config/general_config_full_run.json'
    paths_conf_path = '../../SHRP2/config/paths_config_3_cls.json'
    paths_conf_path = '../../SHRP2/config/paths_config_3_cls_150.json'

    general_conf_obj = GeneralConfig(general_conf_path)
    paths_conf_obj = PathsConfig(paths_conf_path)

    return general_conf_obj, paths_conf_obj


def get_features_names(general_conf_obj):
    merged_preds_X = pd.read_csv(output_handler.train_sets_dir_ensemble + 'X_merged.csv')
    # y_correct.to_csv(output_handler.train_sets_dir_ensemble + 'y.csv', index=False)
    # y_correct_encoded.to_csv(output_handler.train_sets_dir_ensemble + 'y_encoded.csv', index=False)

    features = merged_preds_X.columns.values
    new_features = []
    for feature_name in features:
        new_feature_name = feature_name.replace('_standard_deviation', 'std')
        new_feature_name = new_feature_name.replace('_minimum', 'min')
        new_feature_name = new_feature_name.replace('_maximum', 'max')
        new_feature_name = new_feature_name.replace('_variance', 'var')
        new_feature_name = new_feature_name.replace('_median', 'median')
        new_feature_name = new_feature_name.replace('_sum_values', 'sum')
        new_feature_name = new_feature_name.replace('_mean', 'mean')


        new_features.append(new_feature_name)
    return new_features




################################################CONFS################################################################
classes_names = ['residential', 'secondary', 'motorway']

original_class = 'residential'
original_class = 'secondary'
original_class = 'motorway'




############################### DS ####################################
DS = 'european'
DS = 'shrp'


if DS == 'european':
    seed = '12'
    output_dir = 'final_output/european/seed_{}_sefi/'.format(seed)
    general_conf_obj, paths_conf_obj = load_configs_european()
else:
    seed = '12_100'
    output_dir = 'final_output/shrp/seed_{}/'.format(seed)
    general_conf_obj, paths_conf_obj = load_configs_shrp()

output_handler = run_helper.OutputDirHandler(paths_conf_obj, general_conf_obj, output_dir)

classes_mapping = run_helper.create_class_to_idx_mapping(general_conf_obj, paths_conf_obj, output_handler)

features = get_features_names(general_conf_obj)
if DS == 'european':
    MODEL_DIR = output_handler.output_dir_model_ensemble
    model_to_check_path = MODEL_DIR + paths_conf_obj.meta_learner_best_model_name
else:
    model_to_check_path = '../../../final_output/shrp/seed_12_100/ensemble/best_model/meta_best_logistic.pickle'

TEST_SETS_DIR = output_handler.test_sets_dir_xgboost_online


# model_to_check_path = MODEL_DIR + paths_conf_obj.meta_learner_best_model_name


################################ ref for explaination ###################################
# https://github.com/slundberg/shap/issues/457


##############################################################################################

print('loading model {}'.format(model_to_check_path))
ensamble_model = pickle_util.load_obj(model_to_check_path)

# read X for feature names:

###############################rename features#########################################################
if DS == 'european':
    import configuration.signals_mapping_short as european_signals_mapping
    new_features = []
    signlas_ids_mapping = european_signals_mapping.map
    for feature_name_with_signal_id in features:
        splitted = feature_name_with_signal_id.split('_')
        if splitted[1] not in ['ae'] and splitted[0] not in classes_names:
            feature_signal_id = splitted[0]
            signal_name = signlas_ids_mapping.get(feature_signal_id)
            if signal_name == None:
                print(signal_name)
                print(feature_signal_id)

            new_feature = feature_name_with_signal_id.replace(feature_signal_id, signal_name)
            new_features.append(new_feature)
        else:
            new_features.append(feature_name_with_signal_id)
    # print(new_features)
    features = new_features

################################all feature importance#########################################
# if DS == 'european':
coef_all = ensamble_model.coef_
ensemble_coef_all_abs = np.abs(ensamble_model.coef_)
ensemble_coef_all_sum = np.sum(ensemble_coef_all_abs, axis=0)
# else:
#     ensemble_coef_all_sum = ensamble_model.feature_importances_
#     coef_all= ensemble_coef_all_sum
feature_importance_all_dict = dict(zip(features, ensemble_coef_all_sum))
feature_importance_all_dict_sorted = \
    {k: v for k, v in sorted(feature_importance_all_dict.items(), key=lambda item: item[1])}
print('')

slim_dict_all = {}
# keep feature of nn_basic and xgboost only
keep_features = [s + '_nn_basic' for s in classes_names] + [s + '_xg' for s in classes_names]

for feature_name, importance in feature_importance_all_dict_sorted.items():
    if feature_name in keep_features:
        slim_dict_all[feature_name] = importance

print('')


################################ per class #########################################
# coef_per_class = coef_all[classes_mapping[original_class]]
coef_per_class = np.abs(coef_all[classes_mapping[original_class]])
feature_importance_per_class_dict = dict(zip(features, coef_per_class))

#full feature set dict
feature_importance_per_class_dict_sorted = \
    {k: v for k, v in sorted(feature_importance_per_class_dict.items(), key=lambda item: item[1])}
print('')


#slim dict
slim_dict_per_class = {}
for feature_name, importance in feature_importance_per_class_dict_sorted.items():
    if feature_name in keep_features:
        slim_dict_per_class[feature_name] = importance



#############################################plot feature importance per class##########################################

def plot_feature_importance(feature_importance_dict, plot_path, full_size=True):
    plt.clf()
    if full_size:
        plt.gcf().set_size_inches(12, 30)
    else:
        plt.gcf().set_size_inches(12, 8)
    features_keys = list(feature_importance_dict.keys())
    features_values = list(feature_importance_dict.values())
    y_pos = np.arange(len(features_keys))
    plt.barh(y_pos, features_values, 0.7, align='center')
    plt.yticks(y_pos, fontsize=13)
    plt.gca().set_yticklabels(features_keys)
    plt.yticks(rotation=50)
    plt.savefig(plot_path)
    plt.show()
    plt.clf()
#########################################plot stuff########################################
###### plot all feature importance
plot_title = 'all'
plot_path_all = f'plots/{DS}/ensemble/{plot_title}'
plot_feature_importance(feature_importance_all_dict_sorted, plot_path_all)

###### plot all feature importance slim
plot_title = 'all_slim'
plot_path_all = f'plots/{DS}/ensemble/{plot_title}'
plot_feature_importance(slim_dict_all, plot_path_all, full_size=False)

###### plot per class importnace
plot_title = f'{original_class}'
plot_path_per_class = f'plots/{DS}/ensemble/{plot_title}'
plot_feature_importance(feature_importance_per_class_dict_sorted, plot_path_per_class)

###### plot per class slim importnace
plot_title = f'{original_class}_slim'
plot_path_per_class_slim = f'plots/{DS}/ensemble/{plot_title}'
plot_feature_importance(slim_dict_per_class, plot_path_per_class_slim, full_size=False)


























