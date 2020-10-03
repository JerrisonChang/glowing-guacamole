#Python
import csv
"""
This code is designed to extract data from raw data, and output several 
student interaction matrices into the specified directories
"""

path = "./data_sets/ds76_tx_2020_0918_151755/All_data.csv"

if __name__ == "__main__":
    
    with open(path, 'r', newline='') as csv_file:
        span = csv.reader(csv_file) 
        header = next(span)
        
        student_ID_col_index: int
        component_col_index: int

        for i, n in enumerate(header):
            if n == "Anon Student Id":
                student_ID_col_index = i
                continue
            if n == "KC (Original)":
                component_col_index = i
                break
        print(student_ID_col_index, component_col_index)