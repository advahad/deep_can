import os
import time
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from config.general_config import GeneralConfig
from config.paths_config import PathsConfig
from model.common.AE.ae_main import TimeSeriesAutoEncoder
from model.common.helpers import run_helper, global_seeds
from model.common.tpot import tpot_classifier
from model.common.FCN.fcn_ae_model_shrp import FcnAeSingle
from model.SHRP2.baseline import baseline_main as baseline_util
from util import logs_util
from util import pickle_util

paths_conf_path = '../config/paths_config_3_cls_150.json'
general_conf_path = '../config/general_config_full_run.json'
general_conf_path = '../config/general_config.json'
general_conf_obj = GeneralConfig(general_conf_path)
general_conf_obj.print()
paths_conf_obj = PathsConfig(paths_conf_path)




SAMPLES_FILES = os.listdir(paths_conf_obj.samples_path)
seed = '27_sefi'
seed = 12

print('###########################seed is: {}'.format(seed))

global_seeds.set_seeds(seed)
output_dir = 'final_output/shrp/seed_{}_100/'.format(seed)

output_handler = run_helper.OutputDirHandler(paths_conf_obj, general_conf_obj, output_dir)
def get_train_test_sets(generate_sets, partition, label_encoder, initialize_weights_with_model):
    def load_sets():
        print("\nload sets:")
        x_train = pickle_util.load_obj(output_handler.train_sets_dir_nn + paths_conf_obj.x_train_file_name)
        y_train_encoded = pd.read_csv(output_handler.train_sets_dir_nn + paths_conf_obj.y_train_encoded_file_name)
        print("done load train")

        x_val = pickle_util.load_obj(output_handler.val_sets_dir_nn + paths_conf_obj.x_val_file_name)
        y_val_encoded = pd.read_csv(output_handler.val_sets_dir_nn + paths_conf_obj.y_val_encoded_file_name)
        print("done load val")

        x_test = pickle_util.load_obj(output_handler.test_sets_dir_nn + paths_conf_obj.x_test_file_name)
        y_test = pd.read_csv(output_handler.test_sets_dir_nn + paths_conf_obj.y_test_file_name)
        print("done load test")
        return x_train, y_train_encoded, x_val, y_val_encoded, x_test, y_test

    if initialize_weights_with_model:
        print("initialize_weights_with_model")
        x_train, y_train_encoded, x_val, y_val_encoded, x_test, y_test = load_sets()
    elif generate_sets:
        print("generating sets:")
        print("generate train")
        x_train, y_train, y_train_encoded = run_helper.generate_x_y_sets(output_handler.train_sets_dir_nn,
                                                                         partition['train'],
                                                                         paths_conf_obj.x_train_file_name,
                                                                         paths_conf_obj.y_train_file_name,
                                                                         paths_conf_obj.y_train_encoded_file_name,
                                                                         paths_conf_obj.samples_path,
                                                                         general_conf_obj.features.non_features_cols,
                                                                         general_conf_obj.features.remove_features,
                                                                         general_conf_obj.features.keep_features,
                                                                         paths_conf_obj.seq_size,
                                                                         label_encoder)

        print("generate val")
        x_val, y_val, y_val_encoded = run_helper.generate_x_y_sets(output_handler.val_sets_dir_nn,
                                                                   partition['val'],
                                                                   paths_conf_obj.x_val_file_name,
                                                                   paths_conf_obj.y_val_file_name,
                                                                   paths_conf_obj.y_val_encoded_file_name,
                                                                   paths_conf_obj.samples_path,
                                                                   general_conf_obj.features.non_features_cols,
                                                                   general_conf_obj.features.remove_features,
                                                                   general_conf_obj.features.keep_features,
                                                                   paths_conf_obj.seq_size,
                                                                   label_encoder)

        # generate test set
        print("generate test")
        x_test, y_test, y_test_encoded = run_helper.generate_x_y_sets(output_handler.test_sets_dir_nn,
                                                                      partition['test'],
                                                                      paths_conf_obj.x_test_file_name,
                                                                      paths_conf_obj.y_test_file_name,
                                                                      paths_conf_obj.y_test_encoded_file_name,
                                                                      paths_conf_obj.samples_path,
                                                                      general_conf_obj.features.non_features_cols,
                                                                      general_conf_obj.features.remove_features,
                                                                      general_conf_obj.features.keep_features,
                                                                      paths_conf_obj.seq_size,
                                                                      label_encoder)
    else:  # load sets
        print("load sets")
        x_train, y_train_encoded, x_val, y_val_encoded, x_test, y_test = load_sets()

    return x_train, y_train_encoded, x_val, y_val_encoded, x_test, y_test


def train_ae(num_of_classes, features_size, initialize_weights_with_model, x_train, x_val, x_test):
    ae_model = TimeSeriesAutoEncoder(paths_conf_obj.seq_size, features_size, num_of_classes,
                                     use_generator=True,
                                     train_generator=None,
                                     val_generator=None,
                                     processor_type='GPU',
                                     model_path=output_handler.output_dir_model_ae + paths_conf_obj.best_model_name,
                                     initialize_weights_with_model=initialize_weights_with_model)

    # TODO: get log dir of tensoeboard
    ae_model.train(x_train, x_val, x_test,
                   output_handler.output_dir_model_ae + paths_conf_obj.best_model_name,
                   general_conf_obj.network.epochs, general_conf_obj.network.batch_size,
                   general_conf_obj.network.log_dir, paths_conf_obj.target,
                   output_handler.output_dir_logs_ae + general_conf_obj.network.hist_file_name,
                   output_handler.output_dir_logs_ae + general_conf_obj.network.duration_file_name)

def load_ae_model():
    ts_ae_model = TimeSeriesAutoEncoder(model_path=output_handler.output_dir_model_ae + paths_conf_obj.best_model_name,
                                        initialize_weights_with_model=True,
                                        es_patience=general_conf_obj.network.es_patience)
    return ts_ae_model

def predict_ae(x_test_path, y_pred_path):
    ts_ae_model = load_ae_model()
    x_test = pickle_util.load_obj(x_test_path)
    ts_ae_model.predict_using_generator(x_test, y_pred_path)


def evaluate_ae(hist_path, duration_path):
    hist = pd.read_csv(hist_path)
    duration = pickle_util.load_obj(duration_path)
    y_pred_encoded = pickle_util.load_obj(output_handler.test_sets_dir_ae + paths_conf_obj.y_test_pred_file_name)
    y_test_encoded = pickle_util.load_obj(output_handler.test_sets_dir_nn + paths_conf_obj.x_test_file_name)
    logs_util.save_logs_ae(output_handler.output_dir_logs_ae, hist, y_pred_encoded, y_test_encoded, duration)


def train_fcn(num_of_classes, features_size, initialize_weights_with_model, x_train, x_val, x_test, y_train, y_val,
              y_test, input_shape_ae, concat_ae,
              output_dir_model=output_handler.output_dir_model_nn + paths_conf_obj.best_model_name,
              output_dir_logs=output_handler.output_dir_logs_nn):

    fcn_model = FcnAeSingle(paths_conf_obj.seq_size, features_size, num_of_classes,
                            use_generator=True,
                            train_generator=None,
                            val_generator=None,
                            processor_type='GPU',
                            model_path=output_dir_model + paths_conf_obj.best_model_name,
                            initialize_weights_with_model=initialize_weights_with_model,
                            es_patience=general_conf_obj.network.es_patience,
                            concat_ae=concat_ae,
                            input_shape_ae=input_shape_ae,
                            ae_best_model_path=output_handler.output_dir_model_ae + paths_conf_obj.best_model_name)

    if concat_ae:
        fcn_model.train_fcn_ae(x_train, y_train, x_val, y_val, x_test, y_test,
                               output_dir_model + paths_conf_obj.best_model_name,
                               general_conf_obj.network.epochs, general_conf_obj.network.batch_size,
                               general_conf_obj.network.log_dir, paths_conf_obj.target,
                               output_dir_logs + general_conf_obj.network.hist_file_name,
                               output_dir_logs + general_conf_obj.network.duration_file_name)
    else:
        fcn_model.train(x_train, y_train, x_val, y_val, x_test, y_test,
                        output_dir_model + paths_conf_obj.best_model_name,
                        general_conf_obj.network.epochs, general_conf_obj.network.batch_size,
                        general_conf_obj.network.log_dir, paths_conf_obj.target,
                        output_dir_logs + general_conf_obj.network.hist_file_name,
                        output_dir_logs + general_conf_obj.network.duration_file_name)



def predict_fcn(x_test_path, y_test_path, y_pred_path, y_pred_encoded_path, y_pred_path_proba, le, output_dir_model,
                output_dir_model_ae):
    model = run_helper.load_fcn_model(output_dir_model, output_dir_model_ae, paths_conf_obj, general_conf_obj)
    x_test = pickle_util.load_obj(x_test_path)
    y_test = pd.read_csv(y_test_path)
    if general_conf_obj.running.concat_ae:
        y_pred_prob = model.predict_using_generator_ae(x_test, y_test, y_test_path, y_pred_path, y_pred_encoded_path,
                                                       y_pred_path_proba, le)
    else:
        y_pred_prob = model.predict_using_generator(x_test, y_test, y_test_path, y_pred_path, y_pred_encoded_path,
                                                    y_pred_path_proba, le)
    return y_pred_prob


def evaluate_fcn(hist_path, duration_path, classes, predictions_dir, output_dir_model, output_dir_model_ae,
                 output_dir_logs, plots_dir):
    model = run_helper.load_fcn_model(output_dir_model, output_dir_model_ae, paths_conf_obj, general_conf_obj)
    y_test= pd.read_csv(output_handler.test_sets_dir_nn + paths_conf_obj.y_test_file_name)
    y_pred = pd.read_csv(predictions_dir + paths_conf_obj.y_test_pred_file_name)
    y_pred_proba = pd.read_csv(predictions_dir + paths_conf_obj.y_test_pred_proba_file_name)


    hist = pd.read_csv(hist_path)
    duration = pickle_util.load_obj(duration_path)
    logs_util.save_logs(output_dir_logs, hist, y_pred, y_pred_proba.values, y_test, duration)

    model.evaluate_trips(output_handler.test_sets_dir_nn + paths_conf_obj.y_test_file_name,
                         predictions_dir + paths_conf_obj.y_test_pred_file_name,
                         predictions_dir + paths_conf_obj.y_test_pred_proba_file_name,
                         classes,
                         paths_conf_obj.target,
                         plots_dir)


def evaluate_ensemble(y_test_path, y_pred_path, y_pred_proba_path):
    model = run_helper.load_fcn_model(output_handler.output_dir_model_nn, output_handler.output_dir_model_ae,
                                      paths_conf_obj, general_conf_obj)

    model.evaluate_trips(y_test_path,
                         y_pred_path,
                         y_pred_proba_path,
                         classes,
                         paths_conf_obj.target,
                         output_handler.output_dir_plots_ensemble)


def train_xgboost(X_train_path, y_train_path, model_path):
    print("start xgboost training")
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)


    model = XGBClassifier(max_depth=15, n_estimators=200)



    model.fit(X_train, y_train)

    # save model to file
    pickle_util.save_obj(model, model_path)


def get_input_shape_ae(features_size):
    return int(features_size / 2)

def create_final_decision_features(label_encoder, y_preds_nn_path, y_preds_ts_xgboost_path, y_correct_preds):
    y_correct = pd.read_csv(y_correct_preds)
    y_correct_encoded = label_encoder.transform(y_correct)

    y_pred_proba_NN = pd.read_csv(y_preds_nn_path)
    y_pred_proba_tsfresh = pd.read_csv(y_preds_ts_xgboost_path)

    max_class_nn = y_pred_proba_NN.apply(lambda x: x.argmax(), axis=1)
    max_class_tsfresh = y_pred_proba_tsfresh.apply(lambda x: x.argmax(), axis=1)

    max_class_nn.columns = pd.Series(['pred_nn'])
    max_class_tsfresh.columns = pd.Series(['pred_tsfresh'])

    merged_preds_X = pd.concat([max_class_nn, max_class_tsfresh], axis=1)
    return merged_preds_X


def create_data_for_meta_learner_two_features(label_encoder, y_preds_nn_path, y_preds_ts_xgboost_path, y_correct_preds):
    y_correct = pd.read_csv(y_correct_preds)
    y_correct_encoded = label_encoder.transform(y_correct)

    y_pred_proba_NN = pd.read_csv(y_preds_nn_path)
    y_pred_proba_tsfresh = pd.read_csv(y_preds_ts_xgboost_path)

    max_class_nn = y_pred_proba_NN.apply(lambda x: x.argmax(), axis=1)
    max_class_tsfresh = y_pred_proba_tsfresh.apply(lambda x: x.argmax(), axis=1)

    max_class_nn.columns = pd.Series(['pred_nn'])
    max_class_tsfresh.columns = pd.Series(['pred_tsfresh'])

    merged_preds_X = pd.concat([max_class_nn, max_class_tsfresh], axis=1)

    return merged_preds_X, y_correct, y_correct_encoded


def select_features(X_train, y_train):

    from sklearn.feature_selection import f_classif
    from sklearn.feature_selection import SelectKBest
    k = 3
    fs = SelectKBest(score_func=f_classif, k=3)
    # learn relationship from training data
    fs.fit(X_train, y_train)

    scores = pd.DataFrame([X_train.columns.values, fs.scores_], index=pd.Index(['feature_name', 'score'])).T
    scores.sort_values(by='score', ascending=False, inplace=True)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    X_train_fs = pd.DataFrame(X_train_fs, columns=scores['feature_name'].values[:k])
    # transform test input data

    return X_train_fs, fs

def create_data_for_meta_learner(label_encoder, y_preds_nn_path, y_preds_nn_basic_path, y_preds_ts_xgboost_path,
                                 embedding_ae_path, y_correct_preds_path,
                                 xgboost_features_path, load_scaler=False):
    y_correct = pd.read_csv(y_correct_preds_path)
    y_correct_encoded = label_encoder.transform(y_correct)
    y_correct_encoded = pd.DataFrame(y_correct_encoded, columns=y_correct.columns)

    # get proba from each classifier
    y_pred_proba_NN_AE = pd.read_csv(y_preds_nn_path)
    y_pred_proba_NN_basic = pd.read_csv(y_preds_nn_basic_path)
    y_pred_proba_tsfresh = pd.read_csv(y_preds_ts_xgboost_path)

    # select top xgboost features
    xgboost_features = pd.read_csv(xgboost_features_path)

    top_features = \
        baseline_util.get_feature_importance(X_test_path=output_handler.val_sets_dir_xgboost_online + general_conf_obj.tsfresh_online.x_NN_tsfreshed_file_name,
                                             model_path=output_handler.output_dir_model_xgboost_online + general_conf_obj.tsfresh_offline.xgboost_model_name,
                                             top_num=10000)

    xgboost_features = xgboost_features[top_features]



    # get embedding features
    embedding_features = pd.read_csv(embedding_ae_path)
    embedding_features.columns = embedding_features.columns + '_ae'

    columns_names = label_encoder.inverse_transform(y_pred_proba_NN_AE.columns.astype(int))
    columns_names_nn = columns_names + '_nn'
    columns_names_nn_basic = columns_names + '_nn_basic'
    columns_names_xgboost = columns_names + '_xg'

    y_pred_proba_NN_AE.columns = columns_names_nn
    y_pred_proba_NN_basic.columns = columns_names_nn_basic
    y_pred_proba_tsfresh.columns = columns_names_xgboost


    # concat all feature
    merged_preds_X = pd.concat([xgboost_features, y_pred_proba_tsfresh], axis=1)
    merged_preds_X = pd.concat([embedding_features, xgboost_features, y_pred_proba_NN_AE, y_pred_proba_NN_basic, y_pred_proba_tsfresh], axis=1)
    x_cols = merged_preds_X.columns
    scaler_path = 'meta_learner_scaler.pickle'
    if load_scaler:
        print('loading scaler')
        scaler = pickle_util.load_obj(scaler_path)
    else:
        print('preparing scaler')
        # scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = StandardScaler()
        scaler.fit(merged_preds_X)
        pickle_util.save_obj(scaler, scaler_path)
    merged_preds_X = scaler.transform(merged_preds_X)
    merged_preds_X = pd.DataFrame(merged_preds_X, columns=x_cols)

    return merged_preds_X, y_correct, y_correct_encoded


def train_meta_learning_two_features(label_encoder, y_train_preds_nn_path, y_train_preds_ts_xgboost_path,
                                     y_train_correct_preds, meta_learner_best_model_path):
    merged_preds_X, y_correct, y_correct_encoded = create_data_for_meta_learner_two_features(label_encoder,
                                                                                             y_train_preds_nn_path,
                                                                                             y_train_preds_ts_xgboost_path,
                                                                                             y_train_correct_preds)

    xgboost = SGDClassifier()


    # xgboost = XGBClassifier()
    print('balanced_accuracy_score')
    xgboost.fit(merged_preds_X, y_correct)

    pickle_util.save_obj(xgboost, meta_learner_best_model_path)

    # evaluate
    meta_learning_predictions = xgboost.predict(merged_preds_X)
    acc = balanced_accuracy_score(y_correct, meta_learning_predictions)
    print("balanced acc for two features: {}".format(acc))


def train_meta_learning(label_encoder, y_train_preds_nn_path, y_train_preds_nn_basic_path, y_train_preds_ts_xgboost_path,
                        embedding_ae_path, y_train_correct_preds, meta_learner_best_model_path, tpot_model_path,
                        xgboost_features_path):
    merged_preds_X, y_correct, y_correct_encoded = create_data_for_meta_learner(label_encoder,
                                                                                y_train_preds_nn_path,
                                                                                y_train_preds_nn_basic_path,
                                                                                y_train_preds_ts_xgboost_path,
                                                                                embedding_ae_path,
                                                                                y_train_correct_preds,
                                                                                xgboost_features_path)
    should_train_tpot = False
    if should_train_tpot == True:
        tpot_classifier.find_best_pipline_tpot(merged_preds_X, y_correct, y_correct_encoded, tpot_model_path)

    exported_pipeline = LogisticRegression(
        class_weight={'motorway': 0.6, 'secondary': 1, 'residential': 0.6},
                                    max_iter=1000)

    exported_pipeline = LogisticRegression(max_iter=1000)
    exported_pipeline.fit(merged_preds_X, y_correct)

    print(meta_learner_best_model_path)
    meta_learner_best_model_path = '../../../final_output/shrp/seed_12_100/ensemble/best_model/meta_best_logistic.pickle'
    pickle_util.save_obj(exported_pipeline, meta_learner_best_model_path)


    meta_learning_predictions = exported_pipeline.predict(merged_preds_X)
    acc = balanced_accuracy_score(y_correct, meta_learning_predictions)
    print("balanced acc: {}".format(acc))




def majority_voting_and_evaluation_by_meta_learner(y_test_pred_proba_nn_path, y_test_pred_proba_nn_basic_path,
                                                   y_test_pred_proba_ts_xgboost_path, embedding_ae_path,
                                                   y_test_correct_preds_path, meta_learner_best_model_path,
                                                   xgboost_features_path, duration):
    merged_preds_X, y_correct, y_correct_encoded = create_data_for_meta_learner(label_encoder,
                                                                                y_test_pred_proba_nn_path,
                                                                                y_test_pred_proba_nn_basic_path,
                                                                                y_test_pred_proba_ts_xgboost_path,
                                                                                embedding_ae_path,
                                                                                y_test_correct_preds_path,
                                                                                xgboost_features_path,
                                                                                load_scaler=True)

    meta_learning_model = pickle_util.load_obj(meta_learner_best_model_path)

    ensemble_predictions = meta_learning_model.predict(merged_preds_X)
    ensemble_predictions_encoded = label_encoder.transform(ensemble_predictions)
    ensemble_predictions_proba = meta_learning_model.predict_proba(merged_preds_X)
    # ensemble_predictions_inverse = label_encoder.inverse_transform(ensemble_predictions)

    y_preds_inverse_ensemble_path = output_handler.test_sets_dir_ensemble + 'y_preds_final_ensemble.csv'
    preds = pd.DataFrame(ensemble_predictions)
    preds.to_csv(y_preds_inverse_ensemble_path, index=False)

    y_preds_encoded_ensemble_path = output_handler.test_sets_dir_ensemble + 'y_preds_encoded_final_ensemble.csv'
    preds_encoded = pd.DataFrame(ensemble_predictions_encoded)
    preds_encoded.to_csv(y_preds_encoded_ensemble_path, index=False)

    y_preds_proba_path = output_handler.test_sets_dir_ensemble + 'y_preds_proba_final_ensemble.csv'
    preds_proba = pd.DataFrame(ensemble_predictions_proba)
    preds_proba.to_csv(y_preds_proba_path, index=False)

    evaluate_ensemble(y_test_correct_preds_path, y_preds_inverse_ensemble_path, y_preds_proba_path)
    logs_util.save_logs(output_handler.output_dir_logs_ensemble, pd.DataFrame([]),
                        preds, ensemble_predictions_proba, y_correct, duration)
    return ensemble_predictions


def majority_voting_and_evaluation_by_meta_learner_two_features(y_test_pred_proba_nn_path,
                                                                y_test_pred_proba_ts_xgboost_path,
                                                                y_test_correct_preds, meta_learner_best_model_path):
    merged_preds_X, y_correct, y_correct_encoded = create_data_for_meta_learner_two_features(label_encoder,
                                                                                             y_test_pred_proba_nn_path,
                                                                                             y_test_pred_proba_ts_xgboost_path,
                                                                                             y_test_correct_preds)

    meta_learning_model = pickle_util.load_obj(meta_learner_best_model_path)

    ensemble_predictions = meta_learning_model.predict(merged_preds_X)
    # ensemble_predictions_inverse = label_encoder.inverse_transform(ensemble_predictions)

    y_preds_inverse_ensemble_path = output_handler.test_sets_dir_ensemble + 'y_preds_final_ensemble.csv'
    pd.DataFrame(ensemble_predictions).to_csv(y_preds_inverse_ensemble_path, index=False)

    evaluate_ensemble(y_test_correct_preds, y_preds_inverse_ensemble_path)
    return ensemble_predictions


def majority_voting_and_evaluation(label_encoder, y_test_path):
    y_pred_proba_NN = pd.read_csv(output_handler.test_sets_dir_nn + paths_conf_obj.y_test_pred_proba_file_name)
    y_pred_proba_tsfresh = pd.read_csv(output_handler.test_sets_dir_xgboost_online +
                                       general_conf_obj.tsfresh_online.y_pred_proba_file_name)

    y_pred_by_voting = y_pred_proba_tsfresh * 0.4 + y_pred_proba_NN * 0.6

    y_pred_encoded = y_pred_by_voting.values.argmax(axis=1)

    y_pred_inverse = label_encoder.inverse_transform(y_pred_encoded)

    # TODO: file name in config_classes
    y_preds_inverse_ensemble_path = output_handler.test_sets_dir_ensemble + 'y_preds_ensemble.csv'
    pd.DataFrame(y_pred_inverse).to_csv(y_preds_inverse_ensemble_path, index=False)

    evaluate_ensemble(y_test_path, y_preds_inverse_ensemble_path)


def generate_predictions_data_for_meta_learner(set_type):
    if set_type == 'val':
        print('generating predictions for val set:')
        # xgboost val set generation
        xgboost_train_cols_order = baseline_util.get_xgboost_cols_order(
                output_handler.train_sets_dir_xgboost_online + general_conf_obj.tsfresh_online.x_NN_tsfreshed_file_name)
        baseline_util.create_aggregate_features(samples_dir_path=paths_conf_obj.ts_fresh_samples_path,
                                                file_names_list=partition['val'],
                                                one_file_dir_path=output_handler.val_sets_dir_xgboost_online,
                                                one_file_name=general_conf_obj.tsfresh_online.x_NN_one_file_name,
                                                y_path=output_handler.val_sets_dir_xgboost_online + general_conf_obj.tsfresh_online.y_NN_tsfreshed_file_name,
                                                kind_params_path='../../../output_shrp_baseline/xgboost/offline/data/' + general_conf_obj.tsfresh_offline.kind_params_file_name,
                                                timeseries_extracted_df_path=output_handler.val_sets_dir_xgboost_online +
                                                                             general_conf_obj.tsfresh_online.x_NN_tsfreshed_file_name,
                                                save_meta=False,
                                                generate_one_file=general_conf_obj.running.generate_tsfresh_test_one_file,
                                                cols_order=xgboost_train_cols_order)

        baseline_util.predict(
            model_path=output_handler.output_dir_model_xgboost_online + general_conf_obj.tsfresh_offline.xgboost_model_name,
            X_test_path=output_handler.val_sets_dir_xgboost_online + general_conf_obj.tsfresh_online.x_NN_tsfreshed_file_name,
            y_pred_path=output_handler.val_sets_dir_xgboost_online + general_conf_obj.tsfresh_online.y_pred_file_name,
            y_pred_proba_path=output_handler.val_sets_dir_xgboost_online + general_conf_obj.tsfresh_online.y_pred_proba_file_name)

        # fcn AE val set generation
        predict_fcn(output_handler.val_sets_dir_nn + paths_conf_obj.x_val_file_name,
                    output_handler.val_sets_dir_nn + paths_conf_obj.y_val_encoded_file_name,
                    output_handler.val_sets_dir_nn + paths_conf_obj.y_pred_file_name,
                    output_handler.val_sets_dir_nn + paths_conf_obj.y_pred_encoded_file_name,
                    output_handler.val_sets_dir_nn + paths_conf_obj.y_pred_proba_file_name,
                    label_encoder,
                    output_dir_model=output_handler.output_dir_model_nn,
                    output_dir_model_ae=output_handler.output_dir_model_ae)

        # fcn basic val set generation
        predict_fcn(output_handler.val_sets_dir_nn + paths_conf_obj.x_val_file_name,
                    output_handler.val_sets_dir_nn + paths_conf_obj.y_val_encoded_file_name,
                    output_handler.val_sets_dir_nn_basic + paths_conf_obj.y_pred_file_name,
                    output_handler.val_sets_dir_nn_basic + paths_conf_obj.y_pred_encoded_file_name,
                    output_handler.val_sets_dir_nn_basic + paths_conf_obj.y_pred_proba_file_name,
                    label_encoder,
                    output_dir_model=output_handler.output_dir_model_nn_basic,
                    output_dir_model_ae=output_handler.output_dir_model_ae)
        nn_set_dir = output_handler.val_sets_dir_nn
        nn_basic_set_dir = output_handler.val_sets_dir_nn_basic
        tsfresh_set_dir = output_handler.val_sets_dir_xgboost_online
        ae_set_dir = output_handler.val_sets_dir_ae

        # ae embedding generation
        ae_model = load_ae_model()
        ae_model.generate_embeddings(output_handler.val_sets_dir_nn + paths_conf_obj.x_val_file_name,
                                     output_handler.val_sets_dir_ae + paths_conf_obj.embedding_file_name, label_encoder)




    else:
        print('generating predictions for train set:')
        baseline_util.predict(
            model_path=output_handler.output_dir_model_xgboost_online + general_conf_obj.tsfresh_offline.xgboost_model_name,
            X_test_path=output_handler.train_sets_dir_xgboost_online + general_conf_obj.tsfresh_online.x_NN_tsfreshed_file_name,
            y_pred_path=output_handler.train_sets_dir_xgboost_online + general_conf_obj.tsfresh_online.y_pred_file_name,
            y_pred_proba_path=output_handler.train_sets_dir_xgboost_online + general_conf_obj.tsfresh_online.y_pred_proba_file_name)

        # fcn val set generation
        predict_fcn(output_handler.train_sets_dir_nn + paths_conf_obj.x_train_file_name,
                    output_handler.train_sets_dir_nn + paths_conf_obj.y_train_encoded_file_name,
                    output_handler.train_sets_dir_nn + paths_conf_obj.y_pred_file_name,
                    output_handler.train_sets_dir_nn + paths_conf_obj.y_pred_encoded_file_name,
                    output_handler.train_sets_dir_nn + paths_conf_obj.y_pred_proba_file_name,
                    label_encoder, output_dir_model=output_handler.output_dir_model_nn,
                    output_dir_model_ae=output_handler.output_dir_model_ae)

        # ae embedding generation
        ae_model = load_ae_model()
        ae_model.generate_embeddings(output_handler.train_sets_dir_nn + paths_conf_obj.x_train_file_name,
                                     output_handler.train_sets_dir_ae + paths_conf_obj.y_train_file_name, label_encoder)

        nn_set_dir = output_handler.train_sets_dir_nn
        nn_basic_set_dir = output_handler.train_sets_dir_nn_basic
        tsfresh_set_dir = output_handler.train_sets_dir_xgboost_online
        ae_set_dir = output_handler.train_sets_dir_ae

    # generate embedding to test time also, there is no other 'predictions' like in classifiers
    ae_model.generate_embeddings(output_handler.test_sets_dir_nn + paths_conf_obj.x_test_file_name,
                                 output_handler.test_sets_dir_ae + paths_conf_obj.embedding_file_name, label_encoder)

    return nn_set_dir, nn_basic_set_dir, tsfresh_set_dir, ae_set_dir

def fcn_flow(concat_ae):
    # class_weights = run_helper.calc_class_weights(y_train_encoded)
    if concat_ae:
        output_dir_model = output_handler.output_dir_model_nn
        output_dir_logs = output_handler.output_dir_logs_nn
        prediction_dir = output_handler.test_sets_dir_nn
        plots_dir = output_handler.output_dir_plots_nn
    else:
        output_dir_model = output_handler.output_dir_model_nn_basic
        output_dir_logs = output_handler.output_dir_logs_nn_basic
        prediction_dir = output_handler.test_sets_dir_nn_basic
        plots_dir = output_handler.output_dir_plots_nn_basic

    train_fcn(num_of_classes=num_of_classes, features_size=features_size,
              initialize_weights_with_model=general_conf_obj.network.initialize_weights_with_model,
              x_train=x_train, x_val=x_val, x_test=x_test, y_train=y_train_encoded, y_test=y_test,
              y_val=y_val_encoded,
              input_shape_ae=input_shape_ae,
              concat_ae=concat_ae,
              output_dir_model=output_dir_model,
              output_dir_logs=output_dir_logs)

    start = time.time()

    predict_fcn(output_handler.test_sets_dir_nn + paths_conf_obj.x_test_file_name,
                output_handler.test_sets_dir_nn + paths_conf_obj.y_test_file_name,
                prediction_dir + paths_conf_obj.y_test_pred_file_name,
                prediction_dir + paths_conf_obj.y_test_pred_encoded_file_name,
                prediction_dir + paths_conf_obj.y_test_pred_proba_file_name,
                label_encoder,
                output_dir_model,
                output_handler.output_dir_model_ae)

    duration = time.time() - start
    print('fcn  concat ae {} test duration {}'.format(concat_ae, duration))

    evaluate_fcn(hist_path=output_dir_logs + general_conf_obj.network.hist_file_name,
                 duration_path=output_dir_logs + general_conf_obj.network.duration_file_name,
                 classes=classes, predictions_dir=prediction_dir, output_dir_model=output_dir_model,
                 output_dir_model_ae=output_handler.output_dir_model_ae, output_dir_logs=output_dir_logs,
                 plots_dir=plots_dir)


if __name__ == '__main__':
    # general initialization, relevant tp all models in pipeline


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
                                            over_under_type=general_conf_obj.running.is_over,
                                            filter_rpm=False)

    classes = keep_classes_list

    label_encoder = run_helper.create_or_load_label_encoder(classes,
                                                            generate_le=general_conf_obj.running.generate_target_label_encoder,
                                                            path_to_load=output_handler.output_dir +
                                                                         paths_conf_obj.target_label_encoder_file_name)

    num_of_classes = len(classes)

    x_train, y_train_encoded, x_val, y_val_encoded, x_test, y_test = \
        get_train_test_sets(generate_sets=general_conf_obj.running.generate_train_test_val_sets, partition=partition,
                            label_encoder=label_encoder,
                            initialize_weights_with_model=general_conf_obj.network.initialize_weights_with_model)

    features_size = run_helper.calc_num_of_features(SAMPLES_FILES,
                                                    paths_conf_obj.samples_path,
                                                    general_conf_obj.features.non_features_cols,
                                                    general_conf_obj.features.remove_features,
                                                    general_conf_obj.features.keep_features)
    print('num of feature: {}'.format(features_size))

    if general_conf_obj.running.train.ae:
        print("###########start training ae############")
        train_ae(num_of_classes=num_of_classes, features_size=features_size,
                 initialize_weights_with_model=general_conf_obj.network.initialize_weights_with_model,
                 x_train=x_train, x_val=x_val, x_test=x_test)

        predict_ae(output_handler.test_sets_dir_nn + paths_conf_obj.x_test_file_name, output_handler.test_sets_dir_ae +
                   paths_conf_obj.y_test_pred_file_name)

        evaluate_ae(hist_path=output_handler.output_dir_logs_ae + general_conf_obj.network.hist_file_name,
                    duration_path=output_handler.output_dir_logs_ae + general_conf_obj.network.duration_file_name)
    if general_conf_obj.running.train.fcn:
        print("###########start training fcn############")
        classes_mapping = run_helper.create_class_to_idx_mapping(general_conf_obj, paths_conf_obj, output_handler)
        input_shape_ae = get_input_shape_ae(features_size)

        fcn_flow(concat_ae=True)
        fcn_flow(concat_ae=False)


    if general_conf_obj.running.train.tsfresh:
        print("###########start training xgboost############")
        # create aggregation features for train
        if general_conf_obj.running.generate_tsfresh_aggregation_set:
            baseline_util.create_aggregate_features(samples_dir_path=paths_conf_obj.ts_fresh_samples_path,
                                                    file_names_list=partition['train'],
                                                    one_file_dir_path=output_handler.train_sets_dir_xgboost_online,
                                                    one_file_name=general_conf_obj.tsfresh_online.x_NN_one_file_name,
                                                    y_path=output_handler.train_sets_dir_xgboost_online + general_conf_obj.tsfresh_online.y_NN_tsfreshed_file_name,
                                                    kind_params_path='../../../output_shrp_baseline/xgboost/offline/data/' + general_conf_obj.tsfresh_offline.kind_params_file_name,
                                                    timeseries_extracted_df_path=output_handler.train_sets_dir_xgboost_online +
                                                                                 general_conf_obj.tsfresh_online.x_NN_tsfreshed_file_name,
                                                    save_meta=False,
                                                    generate_one_file=general_conf_obj.running.generate_tsfresh_test_one_file,
                                                    cols_order=None)
        start_time = time.time()
        train_xgboost(
            X_train_path=output_handler.train_sets_dir_xgboost_online + general_conf_obj.tsfresh_online.x_NN_tsfreshed_file_name,
            y_train_path=output_handler.train_sets_dir_xgboost_online + general_conf_obj.tsfresh_online.y_NN_tsfreshed_file_name,
            model_path=output_handler.output_dir_model_xgboost_online + general_conf_obj.tsfresh_offline.xgboost_model_name)
        duration = time.time() - start_time
        # create aggregation features for test
        if general_conf_obj.running.generate_tsfresh_aggregation_set:
            xgboost_train_cols_order = baseline_util.get_xgboost_cols_order(
                output_handler.train_sets_dir_xgboost_online + general_conf_obj.tsfresh_online.x_NN_tsfreshed_file_name)
            start_pred_tsf = time.time()
            baseline_util.create_aggregate_features(samples_dir_path=paths_conf_obj.ts_fresh_samples_path,
                                                    file_names_list=partition['test'],
                                                    one_file_dir_path=output_handler.test_sets_dir_xgboost_online,
                                                    one_file_name=general_conf_obj.tsfresh_online.x_NN_one_file_name,
                                                    y_path=output_handler.test_sets_dir_xgboost_online + general_conf_obj.tsfresh_online.y_NN_tsfreshed_file_name,
                                                    kind_params_path='../../../output_shrp_baseline/xgboost/offline/data/' + general_conf_obj.tsfresh_offline.kind_params_file_name,
                                                    timeseries_extracted_df_path=output_handler.test_sets_dir_xgboost_online +
                                                                                 general_conf_obj.tsfresh_online.x_NN_tsfreshed_file_name,
                                                    save_meta=False,
                                                    generate_one_file=general_conf_obj.running.generate_tsfresh_test_one_file,
                                                    cols_order=xgboost_train_cols_order)
            duration = time.time() - start_time


        baseline_util.predict(
            model_path=output_handler.output_dir_model_xgboost_online + general_conf_obj.tsfresh_offline.xgboost_model_name,
            X_test_path=output_handler.test_sets_dir_xgboost_online + general_conf_obj.tsfresh_online.x_NN_tsfreshed_file_name,
            y_pred_path=output_handler.test_sets_dir_xgboost_online + general_conf_obj.tsfresh_online.y_pred_file_name,
            y_pred_proba_path=output_handler.test_sets_dir_xgboost_online + general_conf_obj.tsfresh_online.y_pred_proba_file_name)
        duration_pred_tsf = time.time() - start_pred_tsf
        print('duration preds tsf: {}'.format(duration_pred_tsf))
        # duration = 0

        baseline_util.evaluate(
            y_test_path=output_handler.test_sets_dir_xgboost_online + general_conf_obj.tsfresh_online.y_NN_tsfreshed_file_name,
            y_pred_path=output_handler.test_sets_dir_xgboost_online + general_conf_obj.tsfresh_online.y_pred_file_name,
            y_pred_proba_path=output_handler.test_sets_dir_xgboost_online + general_conf_obj.tsfresh_online.y_pred_proba_file_name,
            class_names=keep_classes_list,
            plots_path_prefix=output_handler.output_dir_plots_xgboost_online,
            duration=duration,
            logs_dir=output_handler.output_dir_logs_xgboost_online)

    if general_conf_obj.running.train.meta_learner:
        proba_set_for_meta_learner = 'val'
        nn_set_dir, nn_basic_set_dir, tsfresh_set_dir, ae_set_dir = generate_predictions_data_for_meta_learner(proba_set_for_meta_learner)
        # nn_set_dir = output_handler.val_sets_dir_nn
        # nn_basic_set_dir = output_handler.val_sets_dir_nn_basic
        # tsfresh_set_dir = output_handler.val_sets_dir_xgboost_online
        # ae_set_dir = output_handler.val_sets_dir_ae


        start_time = time.time()
        train_meta_learning(label_encoder,
                            nn_set_dir + paths_conf_obj.y_pred_proba_file_name,
                            nn_basic_set_dir + paths_conf_obj.y_pred_proba_file_name,
                            tsfresh_set_dir + general_conf_obj.tsfresh_online.y_pred_proba_file_name,
                            ae_set_dir + paths_conf_obj.embedding_file_name,
                            tsfresh_set_dir + general_conf_obj.tsfresh_online.y_NN_tsfreshed_file_name,
                            output_handler.output_dir_model_ensemble + paths_conf_obj.meta_learner_best_model_name,
                            output_handler.output_dir_model_ensemble + 'tpot_50_gen.py',
                            tsfresh_set_dir + general_conf_obj.tsfresh_online.x_NN_tsfreshed_file_name)
        duration = time.time() - start_time
        print('majority_voting_and_evaluation meta learner')
        majority_voting_and_evaluation_by_meta_learner(
            output_handler.test_sets_dir_nn + paths_conf_obj.y_test_pred_proba_file_name,
            output_handler.test_sets_dir_nn_basic + paths_conf_obj.y_test_pred_proba_file_name,
            output_handler.test_sets_dir_xgboost_online +
            general_conf_obj.tsfresh_online.y_pred_proba_file_name,
            output_handler.test_sets_dir_ae + paths_conf_obj.embedding_file_name,
            output_handler.test_sets_dir_xgboost_online + general_conf_obj.tsfresh_online.y_NN_tsfreshed_file_name,
            output_handler.output_dir_model_ensemble + paths_conf_obj.meta_learner_best_model_name,
            output_handler.test_sets_dir_xgboost_online + general_conf_obj.tsfresh_online.x_NN_tsfreshed_file_name,
            duration)



