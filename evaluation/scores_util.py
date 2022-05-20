import pandas as pd
from sklearn.metrics import accuracy_score


# calc accuracy
def calc_accuracy(y_test_file_name, y_pred_file_name, plots_path_prefix, target):
    y_test_encoded = pd.read_csv(y_test_file_name)
    y_pred_encoded = pd.read_csv(y_pred_file_name)
    accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    # balanced_acc = metrics.balanced_accuracy_score(y_test_encoded, y_pred_encoded)
    # print("Balanced Accuracy: %.2f%%" % (balanced_acc * 100.0))
    score_df = pd.DataFrame([accuracy])
    score_df.to_csv(plots_path_prefix + '/' + target + '/' + 'scores.csv', header=["accuracy"], index=False)
