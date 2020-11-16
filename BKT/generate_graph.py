import os, sys
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd

def get_student_list(path_to_clean_data) -> list:
    with open(path_to_clean_data, 'r') as f:
        content = f.readlines()
    
    result = []
    for line in content:
        student_id = line.split("\t")[1]
        if student_id not in result:
            result.append(student_id)
    
    return result

def get_BKT_true_labels(student_id) -> list:
    file_path = f"./BKT/inputs/predict/clean_data KC (Original){student_id}_predicting.txt"
    with open(file_path, 'r') as f:
        content = f.readlines()
        labels = [line.split("\t")[0] for line in content]

    return labels

def get_BKT_predict_score(student_id) -> list:
    file_path = f"./BKT/output/{student_id}_predict.txt"
    with open(file_path,'r') as f:
        content = f.readlines()
        labels_for_correct = [line.split("\t")[0] for line in content]
    
    return labels_for_correct

def get_BKT_roc_curve(student_list: list) -> tuple:
    true_labels = []
    predict_scores = []
    for student in student_list:
        true_labels += [1 if int(i)==1 else 0 for i in get_BKT_true_labels(student)]
        predict_scores += [float(i) for i in get_BKT_predict_score(student)]
        
    return (*roc_curve(true_labels,predict_scores, pos_label=1), roc_auc_score(true_labels, predict_scores, labels=[0,1]) )

def generate_roc_curves(student_list):
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    
    plt.plot([0,1],[0,1], '--', color="gray", label="random")
    
    # BKT ROC curve
    fpr_bkt, tpr_bkt, threshold, auc_bkt = get_BKT_roc_curve(student_list)
    plt.plot(fpr_bkt,tpr_bkt, label=f"BKT (AUC={auc_bkt:.3f})")
    
    # Simple RNN ROC curve
    simple_rnn_path = "./rnn_roc.csv"
    fpr_rnn, tpr_rnn, threshold_rnn, auc_rnn = get_rnn_roc_curve(simple_rnn_path)
    plt.plot(fpr_rnn, tpr_rnn, label=f"simple RNN (AUC={auc_rnn:.3f})")

    # LSTM RNN ROC curve
    lstm_rnn_path = "./lstm_roc.csv"
    fpr_lstm, tpr_lstm, threshold_lstm, auc_lstm = get_rnn_roc_curve(lstm_rnn_path)
    plt.plot(fpr_lstm, tpr_lstm, label=f"LSTM RNN (AUC={auc_lstm:.3f})")

    plt.legend()
    plt.show()

def get_rnn_roc_curve(file_path: str) -> tuple:
    df = pd.read_csv(file_path)
    true_labels = df['y_actual'].tolist()
    predict_scores = df['y_pred'].tolist()
    fpr, tpr, threshold = roc_curve(true_labels, predict_scores)
    auc_socre = roc_auc_score (true_labels, predict_scores)

    return (fpr, tpr, threshold, auc_socre)




if __name__ == "__main__":
    student_list = get_student_list("./BKT/inputs/clean_data KC (Original).txt")
    
    generate_roc_curves(student_list)