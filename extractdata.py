#Python
import csv
"""
This code is designed to extract data from raw data, and output several 
student interaction matrices into the specified directories
"""

path = "./data_sets/ds76_tx_2020_0918_151755/All_data.csv"

def get_col_indexes(header) -> tuple:
    '''
    this function will return the index of student id, KC, and outcome in a tuple
    '''
    for i, n in enumerate(header):
        if n == "Anon Student Id":
            student_ID_col_index = i
            continue
        if n.lower() == "outcome":
            outcome_idx = i
        if n == "KC (Original)":
            component_col_index = i
            break
    return (student_ID_col_index, outcome_idx, component_col_index)

def get_component_list(data, KC_indx:int) -> list:
    '''
    This funciton will return a list with unique component, 
    then we can make one_hot encoding from the list.
    '''
    result = []
    for row in data:
        KC_name = row[KC_indx]
        if KC_name not in result:
            result.append(KC_name)

    return result

def create_interaction_matrix_of_std(student_Id: str, data, KC_list: list, std_idx, outcome_indx, KC_indx) -> list:
    '''
    This function will return an interaction matrix of the specified student.
    '''
    # output_path = f"./interaction_matrices/{student_Id}.csv"
    # result = [0]*len(KC_list) + [0]
    result = []
    for row in data:
        
        interaction_row = [0]*len(KC_list) + [0]
        std_Id = row[std_idx]
        if std_Id != student_Id: 
            # print(f"Not the same! ({std_Id} and {student_Id} )")
            continue

        KC_component = row[KC_indx]
        outcome = row[outcome_indx]
        
        # set the corresponding KC bit to 1 in interaction row
        KC_bit_idx = KC_list.index(KC_component)
        interaction_row[KC_bit_idx] = 1

        # set the outcome bit if the student get it right
        if outcome.lower() == "correct":
            interaction_row[-1] = 1

        result.append(interaction_row)
    
    return result



if __name__ == "__main__":
    
    with open(path, 'r', newline='') as csv_file:
        span = csv.reader(csv_file) 
        header = next(span)
        
        st_ID_indx, outcome_indx, KC_indx = get_col_indexes(header)
        KC_list = get_component_list(span, KC_indx)
        
    print(st_ID_indx, KC_indx, outcome_indx)

    with open(path, 'r', newline='') as csv_file:
        span = csv.reader(csv_file)
        st_Id = "Stu_02ee1b3f31a6f6a7f4b8012298b2395e"
        
        args = (st_Id, span, KC_list, st_ID_indx, outcome_indx, KC_indx)
        interaction_matrix = create_interaction_matrix_of_std(*args)
        
        print(len(interaction_matrix))
    
    
