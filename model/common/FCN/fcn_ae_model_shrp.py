import os
import time
from pathlib import Path

import pandas as pd
import tensorflow.keras as keras
from tensorflow.keras.models import load_model

from util import evaluate_test
from util import pickle_util
from model.common.helpers import gpu_test
import tensorflow as ts
from model.common.AE.ae_main import TimeSeriesAutoEncoder

gpu_test.get_test_info()


class FcnAeSingle:
    def __init__(self, seq_size=None, features_size=None, num_of_classes=None, model_path=None, use_generator=False,
                 train_generator=None, val_generator=None, processor_type=None,
                 initialize_weights_with_model=False, es_patience=100, concat_ae=False, input_shape_ae=None,
                 ae_best_model_path=None):
        if initialize_weights_with_model and model_path is not None:
            print('initialize model from: {}'.format(model_path))
            self.model = keras.models.load_model(model_path)
        else:
            if concat_ae:
                # self.model = self.make_model(seq_size, features_size, num_of_classes, processor_type)
                self.model = self.make_model_concat_ae(seq_size, features_size, num_of_classes, input_shape_ae)
            else:
                self.model = self.make_model(seq_size, features_size, num_of_classes, processor_type)
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.use_generator = use_generator
        self.es_patience = es_patience

        if concat_ae:
            ts_ae_model = TimeSeriesAutoEncoder(model_path=ae_best_model_path,
                                                initialize_weights_with_model=True)
            layer_name = "embedding"
            self.ecoder_intermediate_layer_model = keras.models.Model(inputs=ts_ae_model.model.input,
                                                                      outputs=ts_ae_model.model.get_layer(
                                                                          layer_name).output)

    def make_model_concat_ae(self, seq_size, features_size, num_of_classes, features_size_ae):

        print("FCN Concat AE model")
        input_shape = (seq_size, features_size)
        input_layer = keras.layers.Input(input_shape)
        filter_size = 8
        conv1 = keras.layers.Conv1D(filters=filter_size, kernel_size=8, padding='same', kernel_regularizer=keras.regularizers.l2(0.08))(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)

        conv2 = keras.layers.Conv1D(filters=filter_size, kernel_size=5, padding='same', kernel_regularizer=keras.regularizers.l2(0.08))(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(filter_size, kernel_size=3, padding='same', kernel_regularizer=keras.regularizers.l2(0.08))(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

        input_layer_ae_embedding_ts = keras.layers.Input((features_size_ae,))
        dense_ae = keras.layers.Dense(10, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(
            input_layer_ae_embedding_ts)
        concat = keras.layers.concatenate([gap_layer, dense_ae])

        output_layer = keras.layers.Dense(num_of_classes, activation='softmax')(concat)

        model = keras.models.Model(inputs=[input_layer, input_layer_ae_embedding_ts], outputs=output_layer)

        model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                      metrics=['accuracy'], weighted_metrics=['accuracy'])
        model.summary()
        return model

    def make_model(self, seq_size, features_size, num_of_classes, processor_type=None):

        print("FCN model")
        input_shape = (seq_size, features_size)
        input_layer = keras.layers.Input(input_shape)
        filter_size = 150
        regulaizer = keras.regularizers.l2(0.001)
        conv1 = keras.layers.Conv1D(filters=filter_size, kernel_size=8, padding='same', kernel_regularizer=keras.regularizers.l2(0.1))(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)

        conv2 = keras.layers.Conv1D(filters=filter_size, kernel_size=5, padding='same', kernel_regularizer=keras.regularizers.l2(0.1))(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(filter_size, kernel_size=3, padding='same',kernel_regularizer=keras.regularizers.l2(0.1))(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(num_of_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                      metrics=['accuracy'], weighted_metrics=['accuracy'])
        model.summary()
        return model

    def prepare_callbacks(self, best_model_path, csv_logger_path, include_tb=False, tb_logdir=None):
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15,
                                                      min_lr=0.0001)
        e_s = keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.es_patience)

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=best_model_path,
                                                           save_best_only=True)

        callbacks = [reduce_lr, model_checkpoint, e_s]
        # callbacks = [reduce_lr, model_checkpoint]
        return callbacks

    def train_fcn_ae(self, x_train, y_train_encoded, x_val, y_val_encoded, x_test, y_test_encoded, best_model_path,
                     epochs,
                     batch_size, tb_logdir, target, hist_path, duration_path, classes_weights=None):
        callbacks = self.prepare_callbacks(best_model_path=best_model_path, csv_logger_path='csv_logger_path',
                                           include_tb=False, tb_logdir=tb_logdir)

        # ts_ae_model = TimeSeriesAutoEncoder(seq_size=x_train.shape[1], features_size=x_train.shape[2])

        x_train_ae_representation = self.ecoder_intermediate_layer_model.predict(x_train)
        x_val_ae_representation = self.ecoder_intermediate_layer_model.predict(x_val)

        start_time = time.time()

        hist = self.model.fit([x_train, x_train_ae_representation], y_train_encoded,
                              validation_data=([x_val, x_val_ae_representation], y_val_encoded),
                              epochs=epochs, batch_size=batch_size, callbacks=callbacks,
                              # validation_split=0.2,
                              verbose=2, shuffle=True, class_weight=classes_weights)
        duration = time.time() - start_time
        # Evaluate the model
        # loaded_model = load_model(best_model_path)
        # scores = self.model.evaluate(x_test, y_test_encoded, verbose=1)
        #
        # print('\nAccurancy: {:.3f}'.format(scores[1]))

        # # Save the model
        # self.model.save('../../../best_models/' + data_repository + '/helpers/' + target + '/helpers-{:.3f}.h5'.format(
        #     (scores[1] * 100)))
        hist_df = pd.DataFrame(hist.history)
        hist_df.to_csv(hist_path, index=False)

        pickle_util.save_obj(duration, duration_path)

    def train(self, x_train, y_train_encoded, x_val, y_val_encoded, x_test, y_test_encoded, best_model_path, epochs,
              batch_size, tb_logdir, target, hist_path, duration_path, classes_weights=None):
        callbacks = self.prepare_callbacks(best_model_path=best_model_path, csv_logger_path='csv_logger_path',
                                           include_tb=False, tb_logdir=tb_logdir)


        start_time = time.time()
        hist = self.model.fit([x_train], y_train_encoded, validation_data=([x_val], y_val_encoded),
                              epochs=epochs, batch_size=batch_size, callbacks=callbacks,
                              # validation_split=0.2,
                              verbose=2, shuffle=True)
        duration = time.time() - start_time
        # Evaluate the model
        # loaded_model = load_model(best_model_path)
        # scores = self.model.evaluate(x_test, y_test_encoded, verbose=1)
        #
        # print('\nAccurancy: {:.3f}'.format(scores[1]))

        # # Save the model
        # self.model.save('../../../best_models/' + data_repository + '/helpers/' + target + '/helpers-{:.3f}.h5'.format(
        #     (scores[1] * 100)))
        hist_df = pd.DataFrame(hist.history)
        hist_df.to_csv(hist_path, index=False)

        pickle_util.save_obj(duration, duration_path)




    def train_with_generator(self, x_test, y_test, best_model_path, epochs, batch_size, tb_logdir, data_repository,
                             target, use_callbacks=False):
        best_model_dir = os.path.dirname(best_model_path)  ## directory of file
        Path(best_model_dir).mkdir(parents=True, exist_ok=True)
        csv_logger_path = '../../../training_history/european/' + target + '.csv'

        val_steps = int(len(self.val_generator.list_IDs) / self.val_generator.batch_size)
        # val_steps = len(self.val_generator.list_IDs)
        print('val steps size {}'.format(val_steps))

        fit_params = {
            'generator': self.train_generator,
            'validation_data': self.val_generator,
            'epochs': epochs,
            'verbose': 2,
            'use_multiprocessing': False,
            'validation_steps': val_steps
        }

        if use_callbacks:
            callbacks = self.prepare_callbacks(best_model_path=best_model_path, csv_logger_path=csv_logger_path,
                                               include_tb=False, tb_logdir=tb_logdir)
            fit_params['callbacks'] = callbacks
        self.model.fit_generator(**fit_params)

        # Evaluate the model
        scores = self.model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
        print('\nAccurancy: {:.3f}'.format(scores[1]))

        # Save the model
        # self.model.save('../../../best_models/' + data_repository + '/helpers/' + target + '/helpers-{:.3f}.h5'.format(
        #     (scores[1] * 100)))

    def predict(self, data, best_model_path, y_test_path, y_pred_path, y_pred_path_proba, meta_data_path):
        new_model = load_model(best_model_path)
        predictions = new_model.predict(data.x_test)

        predictions_probs_df = pd.DataFrame(predictions)

        y_pred_encoded = predictions.argmax(axis=1)

        y_pred_inverse = data.le.inverse_transform(y_pred_encoded)
        # save predictions to csv
        pd.DataFrame(data.y_test).to_csv(y_test_path, index=False)
        pd.DataFrame(y_pred_inverse).to_csv(y_pred_path, index=False)

        # change columns names from encoded before savind
        predictions_probs_df.columns = data.le.inverse_transform(predictions_probs_df.columns)
        predictions_probs_df.to_csv(y_pred_path_proba, index=False)

        pickle_util.save_obj(data.file_id_event_id_dict, meta_data_path)

        print("Done prediction")

    def predict_using_generator_ae(self, x_test, y_test, y_test_path, y_pred_path, y_pred_encoded_path, y_pred_path_proba,
                                le):
        new_model = self.model

        x_test_encoded_ae = self.ecoder_intermediate_layer_model.predict(x_test)
        predictions = new_model.predict([x_test, x_test_encoded_ae])

        predictions_probs_df = pd.DataFrame(predictions)

        y_pred_encoded = predictions.argmax(axis=1)

        y_pred_inverse = le.inverse_transform(y_pred_encoded)
        # save predictions to csv
        pd.DataFrame(y_test).to_csv(y_test_path, index=False)
        pd.DataFrame(y_pred_encoded).to_csv(y_pred_encoded_path, index=False)
        pd.DataFrame(y_pred_inverse).to_csv(y_pred_path, index=False)

        # change columns names from encoded before savind
        # predictions_probs_df.columns = le.inverse_transform(predictions_probs_df.columns)
        predictions_probs_df.to_csv(y_pred_path_proba, index=False)

        # pickle_util.save_obj(data.file_id_event_id_dict)
        ts.keras.backend.clear_session()
        print("Done prediction")

    def predict_using_generator(self, x_test, y_test, y_test_path, y_pred_path, y_pred_encoded_path, y_pred_path_proba,
                                le):
        new_model = self.model

        predictions = new_model.predict(x_test)

        predictions_probs_df = pd.DataFrame(predictions)

        y_pred_encoded = predictions.argmax(axis=1)

        y_pred_inverse = le.inverse_transform(y_pred_encoded)
        # save predictions to csv
        pd.DataFrame(y_test).to_csv(y_test_path, index=False)
        pd.DataFrame(y_pred_encoded).to_csv(y_pred_encoded_path, index=False)
        pd.DataFrame(y_pred_inverse).to_csv(y_pred_path, index=False)

        # change columns names from encoded before savind
        # predictions_probs_df.columns = le.inverse_transform(predictions_probs_df.columns)
        predictions_probs_df.to_csv(y_pred_path_proba, index=False)

        # pickle_util.save_obj(data.file_id_event_id_dict)
        ts.keras.backend.clear_session()
        print("Done prediction")

    def evaluate_trips(self, y_test_path, y_pred_path, class_names, target, save_plots_path_prefix):
        evaluation_type = 'trips'
        save_plots_path_prefix = save_plots_path_prefix + evaluation_type + '/'
        Path(save_plots_path_prefix).mkdir(parents=True, exist_ok=True)
        evaluate_test.evaluate_by_files(y_test_path, y_pred_path, class_names, target, save_plots_path_prefix)

    def get_majority_class_for_baselines(self, complete_test_df, file_id, event_id):
        baselines = complete_test_df.groupby([file_id, event_id])
        true_baselines = []
        predicted_baselines = []
        for baseline_name, baseline_df in baselines:
            # choose majority
            common_class_true = baseline_df['true'].value_counts().index[0]
            common_class_predicted = baseline_df['predicted'].value_counts().index[0]
            true_baselines.append(common_class_true)
            predicted_baselines.append(common_class_predicted)
        return true_baselines, predicted_baselines

    def get_distribution_summation_class_for_baselines(self, complete_test_df, file_id, event_id, class_names):
        baselines = complete_test_df.groupby([file_id, event_id])
        true_baselines = []
        predicted_baselines = []
        for baseline_name, baseline_df in baselines:
            # choose majority
            class_true = baseline_df['true'].value_counts().index[0]
            common_class_predicted = baseline_df[class_names].mean().argmax(axis=1)
            true_baselines.append(class_true)
            predicted_baselines.append(common_class_predicted)
        return true_baselines, predicted_baselines

    def evaluate_baselines(self, y_test_path, y_pred_path, y_pred_path_proba, meta_data_dir_path, y_test_path_baseline,
                           y_pred_path_baseline, class_names, target, save_plots_path_prefix):
        # prepare baseline test set dataframe
        FILE_ID = 'file_id'
        EVENT_ID = 'event_id'
        # load file_id and event_id corresponding to test instances
        file_id_event_id_dict = pickle_util.load_obj(meta_data_dir_path)
        file_ids_test = file_id_event_id_dict.get('y_test').get(FILE_ID)
        event_ids_test = file_id_event_id_dict.get('y_test').get(EVENT_ID)

        # create baseline dataset for evaluation
        complete_baselines_test_df = pd.DataFrame([])
        complete_baselines_test_df[FILE_ID] = file_ids_test
        complete_baselines_test_df[EVENT_ID] = event_ids_test
        complete_baselines_test_df['true'] = pd.read_csv(y_test_path)
        complete_baselines_test_df['predicted'] = pd.read_csv(y_pred_path)
        probas = pd.read_csv(y_pred_path_proba)
        complete_baselines_test_df = pd.concat([complete_baselines_test_df, probas], axis=1)

        true_baselines, predicted_baselines = \
            self.get_distribution_summation_class_for_baselines(complete_baselines_test_df,
                                                                FILE_ID, EVENT_ID, class_names)

        # true_baselines, predicted_baselines = self.get_majority_class_for_baselines(complete_baselines_test_df, FILE_ID,
        #                                                                             EVENT_ID)
        #

        pd.DataFrame(true_baselines).to_csv(y_test_path_baseline, index=False)
        pd.DataFrame(predicted_baselines).to_csv(y_pred_path_baseline, index=False)

        evaluation_type = 'baselines'
        save_plots_path_prefix = save_plots_path_prefix + target + '/' + evaluation_type + '/'
        evaluate_test.evaluate_by_files(y_test_path_baseline, y_pred_path_baseline, class_names, target,
                                        save_plots_path_prefix)

# train_predict_SHRP2()
