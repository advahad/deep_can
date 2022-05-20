import shap
from config_classes.general_config import GeneralConfig
from config_classes.paths_config import PathsConfig

from model.common.helpers import run_helper
import numpy as np
import shap_helper
import pandas as pd
from util import pickle_util


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
    train_data = pd.read_csv(
        output_handler.train_sets_dir_xgboost_online + general_conf_obj.tsfresh_online.x_NN_tsfreshed_file_name)
    features = train_data.columns.values
    new_features = []
    for feature_name in features:
        new_feature_name = feature_name.replace('_standard_deviation', 'std')
        new_feature_name = new_feature_name.replace('_minimum', 'min')
        new_feature_name = new_feature_name.replace('_maximum', 'max')
        new_feature_name = new_feature_name.replace('_variance', 'var')
        new_feature_name = new_feature_name.replace('_median', 'median')
        new_feature_name = new_feature_name.replace('_sum_values', 'sum')


        new_features.append(new_feature_name)
    return new_features




################################################CONFS################################################################
classes_names = ['residential', 'secondary', 'tertiary', 'motorway']
original_class = 'tertiary'
original_class = 'residential'
original_class = 'motorway'
original_class = 'secondary'


predicted_class = 'tertiary'
predicted_class = 'residential'
predicted_class = 'motorway'
predicted_class = 'secondary'

############################### DS ####################################
DS = 'european'
DS = 'shrp'


if DS == 'european':
    seed = '12'
    output_dir = 'final_output/european/seed_{}_sefi/'.format(seed)
    general_conf_obj, paths_conf_obj = load_configs_european()
    CALCS_DIR = 'calcs/european/'
else:
    seed = '12_100'
    output_dir = 'final_output/shrp/seed_{}/'.format(seed)
    general_conf_obj, paths_conf_obj = load_configs_shrp()
    CALCS_DIR = 'calcs/shrp/'

output_handler = run_helper.OutputDirHandler(paths_conf_obj, general_conf_obj, output_dir)

classes_mapping = run_helper.create_class_to_idx_mapping(general_conf_obj, paths_conf_obj, output_handler)

features = get_features_names(general_conf_obj)


if DS == 'european':
    import configuration.signals_mapping_short as european_signals_mapping
    new_features = []
    signlas_ids_mapping = european_signals_mapping.map
    for feature_name_with_signal_id in features:
        feature_signal_id = feature_name_with_signal_id.split('_')[0]
        signal_name = signlas_ids_mapping.get(feature_signal_id)
        new_feature = feature_name_with_signal_id.replace(feature_signal_id, signal_name)
        new_features.append(new_feature)
    # print(new_features)
    features = new_features


MODEL_DIR = output_handler.output_dir_model_xgboost_online
TEST_SETS_DIR = output_handler.test_sets_dir_xgboost_online
BACK_SIZE = 1000
TEST_SIZE = 1000

model_to_check_path = MODEL_DIR + general_conf_obj.tsfresh_offline.xgboost_model_name





##############################################################################################

print('loading model {}'.format(model_to_check_path))
bst_model = pickle_util.load_obj(model_to_check_path)



############################### all records ##############################################
plot_title_all = 'all'
background_all = pd.read_csv(output_handler.train_sets_dir_xgboost_online + general_conf_obj.tsfresh_online.x_NN_tsfreshed_file_name)
X_test_all = pd.read_csv(output_handler.test_sets_dir_xgboost_online + general_conf_obj.tsfresh_online.x_NN_tsfreshed_file_name)

background_all = background_all.iloc[np.random.choice(background_all.shape[0],
                                         min(background_all.shape[0], BACK_SIZE), replace=False)]
X_test_all = X_test_all.iloc[np.random.choice(X_test_all.shape[0],
                                                   min(X_test_all.shape[0], TEST_SIZE), replace=False)]

BACKGROUND_ALL_FILE_NAME = 'background_all.pickle'
X_TEST_ALL_FILE_NAME = 'X_test_all.pickle'
pickle_util.save_obj(background_all, CALCS_DIR + BACKGROUND_ALL_FILE_NAME)
pickle_util.save_obj(X_test_all, CALCS_DIR + X_TEST_ALL_FILE_NAME)


############################### per class records ##############################################
plot_title_per_class = "original_{}_predicted_{}".format(original_class, predicted_class)
# predicted_class = other_class_name
title = "original_{}_predicted_{}".format(original_class, predicted_class)
print("original: {}\npredicted: {}".format(original_class, predicted_class))

# get correct background
background, _ = shap_helper.get_background_per_original_and_predicted_class(
    paths_conf_obj=paths_conf_obj,
    output_handler=output_handler,
    test_sets_dir=TEST_SETS_DIR,
    original_class=original_class,
    predicted_class=original_class,
    is_correct_preds=True,
    is_tree=True
)


background = background.iloc[np.random.choice(background.shape[0],
                                         min(background.shape[0], BACK_SIZE), replace=False)]

print("size of background {}".format(background.shape[0]))

# get test sets to check
X_test_to_check_per_class, y_test_miss_classify = shap_helper.get_background_per_original_and_predicted_class(
    paths_conf_obj=paths_conf_obj,
    output_handler=output_handler,
    test_sets_dir=TEST_SETS_DIR,
    original_class=original_class,
    predicted_class=original_class,
    is_correct_preds=True,
    is_tree=True)

X_test_to_check_per_class = X_test_to_check_per_class.iloc[np.random.choice(X_test_to_check_per_class.shape[0],
                                                                            min(X_test_to_check_per_class.shape[0], TEST_SIZE), replace=False)]
test_size = X_test_to_check_per_class.shape[0]
print("size of test {}".format(test_size))
if test_size == 0:
    print(title)
    print("test size is {}, skipping".format(test_size))

BACKGROUND_PER_CLASS_FILE_NAME = f'background_{plot_title_per_class}.pickle'
X_TEST_TO_CHECK_PER_CLASS_FILE_NAME = f'X_test_to_check_per_class_{plot_title_per_class}.pickle'
pickle_util.save_obj(background, CALCS_DIR + BACKGROUND_PER_CLASS_FILE_NAME)
pickle_util.save_obj(X_test_to_check_per_class, CALCS_DIR + X_TEST_TO_CHECK_PER_CLASS_FILE_NAME)

######################## create Explainer all ########################
print('loading explainer all')
explainer_all = shap.TreeExplainer(bst_model, background_all)
# calc shap values
print('calc shap values')
shap_values_all = explainer_all.shap_values(X_test_all)
EXPLAINER_ALL_FILE_NAME = 'explainer_all.pickle'
SHAP_VALUES_ALL_FILE_NAME = 'shap_values_all.pickle'
pickle_util.save_obj(explainer_all, CALCS_DIR + EXPLAINER_ALL_FILE_NAME)
pickle_util.save_obj(explainer_all, CALCS_DIR + SHAP_VALUES_ALL_FILE_NAME)


# print('extract shap values for specified class')
# if type(shap_values_all) is list:
#     shap_values_all = shap_values_all[classes_mapping[predicted_class]]





######################## create Explainer per class ########################
print('loading explainer')
explainer_per_class = shap.TreeExplainer(bst_model, background)
# calc shap values
print('calc shap values')
shap_values_per_class = explainer_per_class.shap_values(X_test_to_check_per_class, check_additivity=False)



print('extract shap values for specified class')
# if type(shap_values_per_class) is list:
if DS == 'european':
    shap_values_per_class = shap_values_per_class[classes_mapping[predicted_class]]

EXPLAINER_ALL_FILE_NAME = f'explainer_per_class_{plot_title_per_class}.pickle'
SHAP_VALUES_ALL_FILE_NAME = f'shap_values_per_class_{plot_title_per_class}.pickle'
pickle_util.save_obj(explainer_all, CALCS_DIR + EXPLAINER_ALL_FILE_NAME)
pickle_util.save_obj(explainer_all, CALCS_DIR + SHAP_VALUES_ALL_FILE_NAME)









#####################################plot######################################################
num_of_features_to_plot = 20
show = False
# plot bar for all
shap_helper.plot_bar(shap_values_all, X_test_all, features, list(classes_mapping.keys()), plot_title_all,
                     num_of_features_to_plot,  DS + '/tree', show=show)

# plot bar for per class
shap_helper.plot_bar(shap_values_per_class, X_test_to_check_per_class, features, list(classes_mapping.keys()),
                     plot_title_per_class, num_of_features_to_plot, DS + '/tree', show=show)

shap_helper.plot_dot(shap_values_per_class, X_test_to_check_per_class, features, list(classes_mapping.keys()),
                     plot_title_per_class, num_of_features_to_plot, DS + '/tree', show=show)




