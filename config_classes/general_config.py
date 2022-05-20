from config.referenced_dict import load_json_file_to_my_dict
import json
from types import SimpleNamespace


class GeneralConfig:
    def __init__(self, json_config_path):
        with open(json_config_path) as json_file:
            conf_dict = json.load(json_file)
            # Parse JSON into an object with attributes corresponding to dict keys.
            conf = json.loads(json.dumps(conf_dict), object_hook=lambda d: SimpleNamespace(**d))
        # self.data = self.Data(conf.data.seq_size, conf.data.step_size, conf.data.task, conf.data.target)
        train = self.Running.Train(conf.running.train.ae, conf.running.train.fcn, conf.running.train.tsfresh,
                                   conf.running.train.meta_learner)
        self.running = self.Running(conf.running.architecture_name,
                                    conf.running.concat_ae,
                                    conf.running.predict,
                                    conf.running.generate_target_label_encoder,
                                    conf.running.generate_train_test_val_sets,
                                    conf.running.generate_tsfresh_test_one_file,
                                    conf.running.generate_tsfresh_aggregation_set,
                                    train,
                                    conf.running.is_over)

        self.data = self.Data(conf.data.scaler, conf.data.scaling_type, conf.data.pattern_features, conf.data.gap)
        self.network = self.Network(conf.network.epochs, conf.network.batch_size,
                                    conf.network.log_dir, conf.network.initialize_weights_with_model,
                                    conf.network.hist_file_name, conf.network.duration_file_name,
                                    conf.network.es_patience)
        self.partitioning = self.Partitioning(conf.partitioning.sample_fraq, conf.partitioning.type,
                                              conf.partitioning.under_sample,
                                              conf.partitioning.under_sample_sets,
                                              conf.partitioning.test_set_size,
                                              conf.partitioning.vat_set_size,
                                              conf.partitioning.load)

        self.features = self.Features(conf.features.remove_features, conf.features.keep_features,
                                      conf.features.non_features_cols, conf.features.dummy)

        self.tsfresh_offline = self.TSFreshOffline(conf.tsfresh_offline.step,
                                                   conf.tsfresh_offline.meta_date_file_name,
                                                   conf.tsfresh_offline.x_train_file_name,
                                                   conf.tsfresh_offline.x_test_file_name,
                                                   conf.tsfresh_offline.x_val_file_name,
                                                   conf.tsfresh_offline.y_train_file_name,
                                                   conf.tsfresh_offline.y_test_file_name,
                                                   conf.tsfresh_offline.y_val_file_name,
                                                   conf.tsfresh_offline.kind_params_file_name,
                                                   conf.tsfresh_offline.selected_features_file_name,
                                                   conf.tsfresh_offline.xgboost_model_name)

        self.tsfresh_online = self.TSFreshOnline(conf.tsfresh_online.x_NN_one_file_name,
                                                 conf.tsfresh_online.x_NN_tsfreshed_file_name,
                                                 conf.tsfresh_online.y_NN_tsfreshed_file_name,
                                                 conf.tsfresh_online.y_pred_file_name,
                                                 conf.tsfresh_online.y_pred_proba_file_name)

    class Data:
        def __init__(self, scaler, scaling_type, pattern_features, gap):
            self.scaler = scaler
            self.scaling_type = scaling_type
            self.pattern_features = pattern_features
            self.gap = gap

    class Running:
        def __init__(self, architecture_name, concat_ae, predict, generate_target_label_encoder,
                     generate_train_test_val_sets, generate_tsfresh_test_one_file, generate_tsfresh_aggregation_set,
                     train, is_over):
            self.architecture_name = architecture_name
            self.concat_ae = concat_ae
            self.train = train
            self.predict = predict
            self.generate_target_label_encoder = generate_target_label_encoder
            self.generate_train_test_val_sets = generate_train_test_val_sets
            self.generate_tsfresh_test_one_file = generate_tsfresh_test_one_file
            self.generate_tsfresh_aggregation_set = generate_tsfresh_aggregation_set
            self.is_over = is_over

        class Train:
            def __init__(self, ae, fcn, tsfresh, meta_learner):
                self.ae = ae
                self.fcn = fcn
                self.tsfresh = tsfresh
                self.meta_learner = meta_learner

    class Network:
        def __init__(self, epochs, batch_size, log_dir, initialize_weights_with_model,
                     hist_file_name, duration_file_name, es_patience):
            self.epochs = epochs
            self.batch_size = batch_size
            self.log_dir = log_dir
            self.initialize_weights_with_model = initialize_weights_with_model
            self.hist_file_name = hist_file_name
            self.duration_file_name = duration_file_name
            self.es_patience = es_patience

    class Partitioning:
        def __init__(self, sample_fraq, type, under_sample, under_sample_sets, test_set_size, vat_set_size, load):
            self.sample_fraq = sample_fraq
            self.type = type
            self.under_sample = under_sample
            self.under_sample_sets = under_sample_sets
            self.test_set_size = test_set_size
            self.vat_set_size = vat_set_size
            self.load = load

    class Features:
        def __init__(self, remove_features, keep_features, non_features_cols, dummy):
            self.remove_features = remove_features
            self.keep_features = keep_features
            self.non_features_cols = non_features_cols
            self.dummy = dummy

    class TSFreshOffline:
        def __init__(self, step, meta_date_file_name, x_train_file_name, x_test_file_name, x_val_file_name,
                     y_train_file_name, y_test_file_name, y_val_file_name, kind_params_file_name,
                     selected_features_file_name, xgboost_model_name):
            self.step = step
            self.meta_date_file_name = meta_date_file_name
            self.x_train_file_name = x_train_file_name
            self.x_test_file_name = x_test_file_name
            self.x_val_file_name = x_val_file_name

            self.y_train_file_name = y_train_file_name
            self.y_test_file_name = y_test_file_name
            self.y_val_file_name = y_val_file_name

            self.kind_params_file_name = kind_params_file_name
            self.selected_features_file_name = selected_features_file_name

            self.xgboost_model_name = xgboost_model_name

    class TSFreshOnline:
        def __init__(self, x_NN_one_file_name, x_NN_tsfreshed_file_name, y_NN_tsfreshed_file_name,
                     y_pred_file_name, y_pred_proba_file_name):
            self.x_NN_one_file_name = x_NN_one_file_name
            self.x_NN_tsfreshed_file_name = x_NN_tsfreshed_file_name
            self.y_NN_tsfreshed_file_name = y_NN_tsfreshed_file_name

            self.y_pred_file_name = y_pred_file_name
            self.y_pred_proba_file_name = y_pred_proba_file_name

    def print(self):
        running_obj = self.running
        print('running config_classes: {}'.format(self.running))
        print('architecture_name: {}'.format(running_obj.architecture_name))
        print('concat_ae: {}'.format(running_obj.concat_ae))
        print('generate_train_test_val_sets: {}'.format(running_obj.generate_train_test_val_sets))
        print('generate_target_label_encoder: {}'.format(running_obj.generate_target_label_encoder))
        print('generate_tsfresh_test_one_file: {}'.format(running_obj.generate_tsfresh_test_one_file))
        print('generate_tsfresh_aggregation_set: {}'.format(running_obj.generate_tsfresh_aggregation_set))
        print('train_ae: {}'.format(running_obj.train.ae))
        print('train fcn: {}'.format(running_obj.train.fcn))
        print('train tsfresh: {}'.format(running_obj.train.tsfresh))
        print('train meta_learner: {}'.format(running_obj.train.meta_learner))
        print('train is_over: {}'.format(running_obj.is_over))
        print('partitioning-load: {}'.format(self.partitioning.load))
        print('test size: {}'.format(self.partitioning.test_set_size))
        print('val size: {}'.format(self.partitioning.vat_set_size))





        # print('')
        # print('samples_'
        # print("num of epochs {}".format(general_conf_obj.network.epochs))
        # print('#################################################')
        # print('run fit generator on seq size: {}, step: {}'.format(SEQ_SIZE, STEP_SIZE))
        # print('partioning type is: {}'.format(PARTITIONING_TYPE))
        # print('data FRAQ is: {}'.format(SAMPLE_FRAQ))
        # print('data EPOCHS is: {}'.format(EPOCHS))
        # print('data BATCH_SIZE is: {}'.format(BATCH_SIZE))
