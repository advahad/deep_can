from config.referenced_dict import load_json_file_to_my_dict


class PathsConfig:
    def __init__(self, json_config_path):
        paths_config_ref_dict = load_json_file_to_my_dict(json_config_path)
        self.seq_size = paths_config_ref_dict['seq_size']
        self.step_size = paths_config_ref_dict['step_size']

        self.target = paths_config_ref_dict['target']
        self.best_model_name = paths_config_ref_dict['best_model_name']

        self.windows_dir = paths_config_ref_dict['windows_dir']
        self.base_data_dir = paths_config_ref_dict['base_data_dir']
        self.samples_path = paths_config_ref_dict['samples_path']

        self.ts_fresh_samples_path = paths_config_ref_dict["ts_fresh_samples_path"]

        self.meta_data_dir_path = paths_config_ref_dict['meta_data_dir_path']
        self.meta_data_file_name = paths_config_ref_dict['meta_data_file_name']
        self.meta_data_file_prefix = paths_config_ref_dict['meta_data_file_prefix']

        self.partitioning_obj_file_name = paths_config_ref_dict['partitioning_obj_file_name']
        self.target_label_encoder_file_name = paths_config_ref_dict['target_label_encoder_file_name']

        # test paths
        self.x_test_file_name = paths_config_ref_dict['x_test_file_name']

        self.y_test_file_name = paths_config_ref_dict['y_test_file_name']
        self.y_test_encoded_file_name = paths_config_ref_dict['y_test_encoded_file_name']

        self.y_test_pred_file_name = paths_config_ref_dict['y_test_pred_file_name']
        self.y_test_pred_encoded_file_name = paths_config_ref_dict['y_test_pred_encoded_file_name']
        self.y_test_pred_proba_file_name = paths_config_ref_dict['y_test_pred_proba_file_name']

        # val paths
        self.x_val_file_name = paths_config_ref_dict['x_val_file_name']

        self.y_val_file_name = paths_config_ref_dict['y_val_file_name']
        self.y_val_encoded_file_name = paths_config_ref_dict['y_val_encoded_file_name']

        self.y_pred_file_name = paths_config_ref_dict['y_pred_file_name']
        self.y_pred_encoded_file_name = paths_config_ref_dict['y_pred_encoded_file_name']
        self.y_pred_proba_file_name = paths_config_ref_dict['y_pred_proba_file_name']


        # train pathes
        self.x_train_file_name = paths_config_ref_dict['x_train_file_name']

        self.y_train_file_name = paths_config_ref_dict['y_train_file_name']
        self.y_train_encoded_file_name = paths_config_ref_dict['y_train_encoded_file_name']

        # predictions paths
        self.x_test_correct_preds_file_name = paths_config_ref_dict['x_test_correct_preds_file_name']
        self.y_test_correct_preds_file_name = paths_config_ref_dict['y_test_correct_preds_file_name']

        self.meta_learner_best_model_name = paths_config_ref_dict['meta_learner_best_model_name']

        self.embedding_file_name = paths_config_ref_dict['embedding_file_name']


    def print(self):
        print('')

# json_file_path = 'paths_config.json'
# conf = PathsConfig(json_file_path)
# print(conf)
