from sklearn.model_selection import StratifiedShuffleSplit
from tpot import TPOTClassifier


def find_best_pipline_tpot(merged_preds_X, y_correct_encoded, path_to_selected_pipeline):
    target_col = 'aug.road_type'
    tpot = TPOTClassifier(generations=50, population_size=100, verbosity=2, n_jobs=-1)
    # # del tpot.default_config_dict['sklearn.feature_selection.RFE']

    split_list = list(StratifiedShuffleSplit(n_splits=1, test_size=0.1).split(merged_preds_X, y_correct_encoded))
    train_indices = split_list[0][0]
    test_indices = split_list[0][1]

    X_train = merged_preds_X.iloc[train_indices].values
    y_train_encoded = y_correct_encoded.iloc[train_indices][target_col].values

    X_test = merged_preds_X.iloc[test_indices].values
    y_test_encoded = y_correct_encoded.iloc[test_indices][target_col].values

    tpot.fit(X_train, y_train_encoded)  # train

    print(tpot.score(X_test, y_test_encoded))  # test

    print(tpot.score(X_train, y_train_encoded))
    tpot.export(path_to_selected_pipeline)
