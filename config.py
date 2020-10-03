import csv

with open('All_data.csv', 'r') as f:
    reader = csv.DictReader(f)
    KC_set = set()
    for row in reader:
        kc = row['KC (Original)']
        KC_set.add(kc)
    
    KC_list = list(KC_set)

    f.seek(0)

    tensor = list()
    matrix = list()
    y = list()
    current_std = ''
    first = True
    for row in reader:
        std = row['Anon Student Id']

        if first:
            current_std = std
            first = False

        elif current_std != std:
            tensor.append(matrix)
            matrix = list()
            current_std = std
        
        vector = [0 for i in range(len(KC_list))]
        for i in range(len(KC_list)):
            if row['KC (Original)'] == KC_list[i]:
                vector[i] = 1
                break

        matrix.append(vector)

        if row['outcome'] == 'CORRECT':
            y.append(1)
        else:
            y.append(0)

