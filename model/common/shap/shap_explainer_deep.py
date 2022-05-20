import shap
from config.general_config import GeneralConfig
from config.paths_config import PathsConfig

from model.common.helpers import run_helper
import numpy as np
import os
import shap_helper
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

def get_features_names(general_conf_obj, paths_conf_obj):
    samples_files = os.listdir(paths_conf_obj.samples_path)
    features = run_helper.get_features_names(samples_files=samples_files,
                                             samples_path=paths_conf_obj.samples_path,
                                             non_features_cols=general_conf_obj.features.non_features_cols,
                                             remove_features=general_conf_obj.features.remove_features,
                                             keep_features=general_conf_obj.features.keep_features)

    return features

################################################CONFS################################################################
classes_names = ['residential', 'secondary', 'tertiary', 'motorway']

original_class = 'residential'
original_class = 'motorway'
original_class = 'secondary'

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
else:
    seed = '12_100'
    output_dir = 'final_output/shrp/seed_{}/'.format(seed)
    general_conf_obj, paths_conf_obj = load_configs_shrp()

output_handler = run_helper.OutputDirHandler(paths_conf_obj, general_conf_obj, output_dir)

classes_mapping = run_helper.create_class_to_idx_mapping(general_conf_obj, paths_conf_obj, output_handler)
print(classes_mapping)

features = get_features_names(general_conf_obj, paths_conf_obj)


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

MODEL_DIR = output_handler.output_dir_model_nn_basic
TEST_SETS_DIR = output_handler.test_sets_dir_nn_basic
BACK_SIZE = 1000
TEST_SIZE = 1000

model_to_check_path = MODEL_DIR + paths_conf_obj.best_model_name





##############################################################################################

print('loading model {}'.format(model_to_check_path))
pipeline = run_helper.load_fcn_model(MODEL_DIR, output_handler.output_dir_model_ae,
                                     paths_conf_obj, general_conf_obj)


############################### all records ##############################################
plot_title_all = 'all'
background_all = pickle_util.load_obj(output_handler.train_sets_dir_nn + paths_conf_obj.x_train_file_name)
X_test_all = pickle_util.load_obj(output_handler.test_sets_dir_nn + paths_conf_obj.x_test_file_name)
background_all = background_all[np.random.choice(background_all.shape[0],
                                         min(background_all.shape[0], BACK_SIZE), replace=False)]
X_test_all = X_test_all[np.random.choice(X_test_all.shape[0],
                                                   min(X_test_all.shape[0], TEST_SIZE), replace=False)]



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
    is_correct_preds=True
)


background = background[np.random.choice(background.shape[0],
                                         min(background.shape[0], BACK_SIZE), replace=False)]

print("size of background {}".format(background.shape[0]))

# get test sets to check
X_test_to_check_per_class, y_test_miss_classify = shap_helper.get_background_per_original_and_predicted_class(
    paths_conf_obj=paths_conf_obj,
    output_handler=output_handler,
    test_sets_dir=TEST_SETS_DIR,
    original_class=original_class,
    predicted_class=original_class,
    is_correct_preds=True)

X_test_to_check_per_class = X_test_to_check_per_class[np.random.choice(X_test_to_check_per_class.shape[0],
                                                                       min(X_test_to_check_per_class.shape[0], TEST_SIZE), replace=False)]
test_size = X_test_to_check_per_class.shape[0]
print("size of test {}".format(test_size))
if test_size == 0:
    print(title)
    print("test size is {}, skipping".format(test_size))



######################## create Explainer all ########################
print('loading explainer all')
explainer_all = shap.GradientExplainer(pipeline.model, background_all)
# calc shap values
print('calc shap values')
shap_values_all = explainer_all.shap_values(X_test_all)


# print('extract shap values for specified class')
# if type(shap_values_all) is list:
#     shap_values_all = shap_values_all[classes_mapping[predicted_class]]

# calc shap valus sum for all classes
shap_values_sum_all = [np.sum(vals_per_class, axis=1) for vals_per_class in shap_values_all]
X_test_all_sum_all = np.sum(X_test_all, axis=1)





######################## create Explainer per class ########################
print('loading explainer')
explainer_per_class = shap.GradientExplainer(pipeline.model, background)
# calc shap values
print('calc shap values')
shap_values_per_class = explainer_per_class.shap_values(X_test_to_check_per_class)



print('extract shap values for specified class')
# if type(shap_values_per_class) is list:
shap_values_per_class = shap_values_per_class[classes_mapping[predicted_class]]

# sum shap value over the timesteps dimention from: <instances, timesteps, features> to <instances, features>
shap_values_sum_per_class_sum = np.sum(shap_values_per_class, axis=1)
test_sum_per_class_sum = np.sum(X_test_to_check_per_class, axis=1)
print('shape of summarized data is {}'.format(test_sum_per_class_sum.shape))







#####################################plot######################################################
num_of_features_to_plot = 20
show = False
# plot bar for all
shap_helper.plot_bar(shap_values_sum_all, X_test_all_sum_all, features, list(classes_mapping.keys()), plot_title_all,
                     num_of_features_to_plot, DS + '/deep', show=show)

# plot bar for subset
shap_helper.plot_bar(shap_values_sum_per_class_sum, test_sum_per_class_sum, features, list(classes_mapping.keys()),
                     plot_title_per_class, num_of_features_to_plot,  DS + '/deep', show=show)

shap_helper.plot_dot(shap_values_sum_per_class_sum, test_sum_per_class_sum, features, list(classes_mapping.keys()),
                     plot_title_per_class, num_of_features_to_plot,  DS + '/deep', show=show)




