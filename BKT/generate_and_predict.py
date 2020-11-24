import os

def get_student_list(path_to_clean_data) -> list:
    with open(path_to_clean_data, 'r') as f:
        content = f.readlines()
    
    result = []
    for line in content:
        student_id = line.split("\t")[1]
        if student_id not in result:
            result.append(student_id)
    
    return result

def generate_models(student_list:list, KC_component:str):
    for student_id in student_list:
        os.system(f"./BKT/trainhmm -s 1.2 -d ~ -m 1 -p 1 ./BKT/inputs/train/clean_data\ KC\ \(Original\){student_id}_training.txt ./BKT/model/{student_id}_model.txt ./BKT/model/predict_result/{student_id}_predict.txt")

def predict_test_data(studnet_list: list, KC_component: str):
    for student_id in student_list:
        predicthmm_path = "./BKT/predicthmm"
        params = "-p 1"
        test_data_path = f"./BKT/inputs/predict/clean_data\ KC\ \(Original\){student_id}_predicting.txt"
        model_path = f"./BKT/model/{student_id}_model.txt"
        predict_result_path = f"./BKT/output/{student_id}_predict.txt"
        
        # the command should look like this
        # "./BKT/predicthmm -p 1 ./BKT/inputs/clean_data\ KC\ \(Original\)_predicting.txt ./BKT/model/model.txt ./BKT/model/predict_from_predict.txt"
        # os.system(f"touch {predict_result_path}")
        # os.system(f"{predict_result_path} {params} {test_data_path} {model_path} {predict_result_path}")
        os.system(f"./BKT/predicthmm -p 1 ./BKT/inputs/predict/clean_data\ KC\ \(Original\){student_id}_predicting.txt ./BKT/model/{student_id}_model.txt ./BKT/output/{student_id}_predict.txt")

if __name__ == "__main__":
    KNOWLEDGE_COMPONENT = "KC (Original)"
    path = "./BKT/inputs/clean_data KC (Original).txt"
    student_list = get_student_list(path)
    print(len(student_list))
    
    for i in ['./BKT/model/','./BKT/output/']:
        if not os.path.isdir(i):
            os.makedirs(i)

    generate_models(student_list, KNOWLEDGE_COMPONENT)
    predict_test_data(student_list, KNOWLEDGE_COMPONENT)
    