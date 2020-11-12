# python 3.7
import csv
import os
import random
import math
def clean_data(raw_data: str, output_name:str, KC_name: str):
    """
    This function will prepare the raw data and output the data that is ready to be used in BKT model.
    """
    data_path = f"./{raw_data}"
    output_content = []
    with open(data_path, 'r', newline="") as f:
        whole_data = csv.DictReader(f,delimiter= "\t")
        
        for i in whole_data:
            correctness = '2' if i['First Attempt'].lower() == "correct" else '1'
            student_id = i['Anon Student Id']
            question = f"{i['Problem Hierarchy']}-{i['Problem Name']}-{i['Problem View']}-{i['Step Name']}"
            KC = i[KC_name]
            output_content.append('\t'.join([correctness,student_id,question,KC]))
    
    head, tail = os.path.split(output_name)
    if not os.path.isdir(head):
        os.makedirs(head)

    with open(output_name, 'w') as f:
        f.writelines('\n'.join(output_content))

def create_train_and_predict_data(clean_data_path: str, ratio: float = 0.2):
    with open(clean_data_path, 'r') as f:
        content = f.readlines()
    
    head, tail = os.path.split(clean_data_path)

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
        training_set_path = os.path.join(head, "train", training_name)
        with open(training_set_path, 'w') as f:
            f.writelines(interactions)
        
        predicting_name = tail.replace('.txt', f'{student}_predicting.txt')
        predicting_set_path = os.path.join(head, "predict" ,predicting_name)
        with open(predicting_set_path, 'w') as f:
            f.writelines(predict_set)


if __name__ == "__main__":
    data_path = "./data_sets/ds76_tx_2020_0918_151755/ds76_student_step_All_Data_74_2020_0926_034727.txt"
    KC_name = "KC (Original)"
    output_path = f"./BKT/inputs/clean_data {KC_name}.txt"
    # clean_data(data_path, output_path, KC_name)

    create_train_and_predict_data(output_path)