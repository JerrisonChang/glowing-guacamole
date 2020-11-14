import csv
import os

def clean_data(file_path:str , output_path:str, KC_name: str):
    with open(file_path, 'r', newline="") as f:
        whole_data = csv.DictReader(f, delimiter="\t")
    
        output_content = []
        student_mapping = {}
        item_mapping = {}
        for i in whole_data:
            correctness = 1 if i['First Attempt'].lower() == "correct" else 0
            student_id = i['Anon Student Id']
            question = f"{i['Problem Hierarchy']}-{i['Problem Name']}-{i['Problem View']}-{i['Step Name']}"

            if student_id in student_mapping:
                serialized_user = student_mapping[student_id]
            else:
                new_serialized_number = len(student_mapping) + 1
                student_mapping[student_id] = new_serialized_number
                serialized_user = new_serialized_number

            if question in item_mapping:
                serialized_item = item_mapping[question]
            else:
                new_serialized_item = len(item_mapping) +1
                item_mapping[question] = new_serialized_item
                serialized_item = new_serialized_item

            KC = i[KC_name]
            factorized_data = []
            output_content.append([serialized_user, serialized_item, KC, correctness])
    
    
    head, tail = os.path.split(output_path)
    if not os.path.isdir(head):
        os.makedirs(head)

    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(output_content)

    with open("./KTM/mapping/user_mapping.csv",'w') as f:
        writer = csv.writer(f)
        for user, serialized_id in student_mapping.items():
            writer.writerow([user,serialized_id])
    
    with open("./KTM/mapping/item_mapping.csv", 'w') as f:
        writer = csv.writer(f)
        for question, serialized_id in item_mapping.items():
            writer.writerow([question, serialized_id])
    
        
    

if __name__ == "__main__":
    RAW_DATA_PATH = "./data_sets/ds76_tx_2020_0918_151755/ds76_student_step_All_Data_74_2020_0926_034727.txt"
    # RAW_DATA_PATH = "./data_sets/ds76_tx_2020_0918_151755/All_data.csv"
    KC_NAME = "KC (Original)"
    clean_data(RAW_DATA_PATH, f"./KTM/clean_data/clean_data_{KC_NAME}.csv", KC_NAME)