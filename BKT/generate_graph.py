import os, sys
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def get_student_list(path_to_clean_data) -> list:
    with open(path_to_clean_data, 'r') as f:
        content = f.readlines()
    
    result = []
    for line in content:
        student_id = line.split("\t")[1]
        if student_id not in result:
            result.append(student_id)
    
    return result

def get_true_labels(student_id) -> list:
    file_path = f"./BKT/inputs/predict/clean_data KC (Original){student_id}_predicting.txt"
    with open(file_path, 'r') as f:
        content = f.readlines()
        labels = [line.split("\t")[0] for line in content]

    return labels

def get_predict_score(student_id) -> list:
    file_path = f"./BKT/output/{student_id}_predict.txt"
    with open(file_path,'r') as f:
        content = f.readlines()
        labels_for_correct = [line.split("\t")[0] for line in content]
    
    return labels_for_correct
if __name__ == "__main__":
    student_list = get_student_list("./BKT/inputs/clean_data KC (Original).txt")
    true_labels = [1 if int(i)==1 else 0 for i in get_true_labels("Stu_0dc180ab4196c3c75b62bfc5e01e6ee2")]
    predict_scores = [float(i) for i in get_predict_score("Stu_0dc180ab4196c3c75b62bfc5e01e6ee2")]
    tp, fp, threshold = roc_curve(true_labels,predict_scores, pos_label=1)
    print(roc_auc_score(true_labels, predict_scores))
    plt.plot(fp,tp)
    # for student in student_list:
    #     true_labels = [1 if int(i)==1 else 0 for i in get_true_labels(student)]
    #     predict_scores = [float(i) for i in get_predict_score(student)]
        
    #     tp, fp, threshold = roc_curve(true_labels,predict_scores, pos_label=1)
    #     print(roc_auc_score(true_labels, predict_scores, labels=[0,1]))
    #     plt.plot(fp,tp)
    plt.show()