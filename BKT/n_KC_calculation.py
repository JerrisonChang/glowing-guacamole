import os
import math
import random
import pandas as pd
import numpy

def get_KC_list(clean_data_path: str) -> list:
    KC_list = []
    with open(clean_data_path, 'r') as f:
        for i in f.readlines():
            KC = i.split("\t")[3].strip()
            
            if KC not in KC_list:
                KC_list.append(KC)

    return KC_list

def create_data_with_n_KC(KC_list,data: list, n:int):
    include_list = KC_list[:n]
    with open(f'./BKT/inputs/clean_data_{n}KC.txt','w') as f:
        for row in data:
            kc = row.split('\t')[3].strip()
            if kc in include_list:
                f.write(row)
    
def create_train_predict(n:int):
    RAW_DATA_PATH = f"./BKT/inputs/clean_data_{n}KC.txt"
    ratio = 0.2

    with open(RAW_DATA_PATH, 'r') as f:
        content = f.readlines()
    
    head, tail = os.path.split(RAW_DATA_PATH)

    # arrange the interaction by student
    student_to_interaction:dict = {}
    for line in content:
        student_id = line.split("\t")[1]
        if student_id not in student_to_interaction:
            student_to_interaction[student_id] = [line]
        else:
            student_to_interaction[student_id].append(line)
    
    # create train test sets for each student
    for student, interactions in student_to_interaction.items():
        predict_amount = math.ceil(len(interactions)*ratio)
        predict_set = random.sample(interactions, predict_amount)
        
        for i in predict_set:
            interactions.remove(i) # remove testing sets from training set
    
        training_name = tail.replace('.txt', f'{student}_training.txt')
        training_set_path = os.path.join(head, f"{n}_train", training_name)
        if not os.path.isdir(os.path.join(head, f"{n}_train")):
            os.makedirs(os.path.join(head, f"{n}_train"))
        
        with open(training_set_path, 'w') as f:
            f.writelines(interactions)
        
        predicting_name = tail.replace('.txt', f'{student}_predicting.txt')
        predicting_set_path = os.path.join(head, f"{n}_predict" ,predicting_name)
        if not os.path.isdir(os.path.join(head, f"{n}_predict")):
            os.makedirs(os.path.join(head, f"{n}_predict"))
        
        with open(predicting_set_path, 'w') as f:
            f.writelines(predict_set)

def generate_model(student_list: list, n: int):
    for student_id in student_list:
        os.system(f"./BKT/trainhmm -s 1.2 -d ~ -m 1 -p 1 ./BKT/inputs/{n}_train/clean_data_{n}KC{student_id}_training.txt ./BKT/model/n_KCs/{n}_{student_id}_model.txt ./BKT/model/n_KCs/predict_result/{n}_{student_id}_predict.txt")

def get_student_list(n: int) -> list:
    PATH = f"./BKT/inputs/clean_data_{n}KC.txt"
    with open(PATH, 'r') as f:
        content = f.readlines()
    
    result = []
    for line in content:
        student_id = line.split("\t")[1]
        if student_id not in result:
            result.append(student_id)
    
    return result

def generate_predict_files(student_list: list, n: int):
    for student_id in student_list:
        os.system(f"./BKT/predicthmm -p 1 ./BKT/inputs/{n}_predict/clean_data_{n}KC{student_id}_predicting.txt ./BKT/model/n_KCs/{n}_{student_id}_model.txt ./BKT/output/n_KCs/{n}_{student_id}_predict.txt")


def get_accuracy(student_list: list, n: int) -> float:
    accuracies = []
    for student_id in student_list:
        try:
            df_actual = pd.read_csv(f'./BKT/inputs/{n}_predict/clean_data_{n}KC{student_id}_predicting.txt', delimiter="\t", header=None)
            df_predict = pd.read_csv(f'./BKT/output/n_KCs/{n}_{student_id}_predict.txt', delimiter="\t", header=None)
        except:
            continue
        actual_labels = [1 if i == 1 else 0 for i in df_actual[0].tolist() ]
        predict_labels = [1 if i>= 0.5 else 0 for i in df_predict[0].tolist()]
        corrects = [1 if y == y_hat else 0 for y, y_hat in zip(actual_labels, predict_labels)]
        accuracies.append(sum(corrects)/len(corrects))
            
    return numpy.average(accuracies)

if __name__ == "__main__":
    clean_data_path = "./BKT/inputs/clean_data KC (Original).txt"
    
    kc_list = get_KC_list(clean_data_path)
    # with open(clean_data_path, 'r') as f:
    #     data = f.readlines()

    #     create_data_with_n_KC(kc_list, data, i)
    accuracies = []
    for i in range(1,len(kc_list)+1):
        # create_train_predict(i)
        
        student_list = get_student_list(i)
        # # generate model
        # generate_model(student_list,i)

        # # generate predict model
        # generate_predict_files(student_list, i)

        # # get accuracty score
        accuracies.append(get_accuracy(student_list,i))
    
    df = pd.DataFrame({
        'KC number': [i for i in range(1, len(kc_list)+1)],
        'accuracy': accuracies
    })
    
    df.to_csv('./BKT/bkt_accuracies.csv',index=False)