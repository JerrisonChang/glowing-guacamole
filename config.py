import pandas as pd
import tensorflow as tf
import numpy as np


MASK_VALUE = -1.  # The masking value cannot be zero.


def load_dataset(fn):
    df = pd.read_csv(fn)

    # Step 1 - Remove questions without skill and Remove users with a single answer
    df.dropna(subset=['KC (Textbook)'], inplace=True)
    df = df.groupby('Anon Student Id').filter(lambda q: len(q) > 1).copy()

    # Step 2 - Enumerate skill id and change outcome to 0 or 1 values
    df['skill'], _ = pd.factorize(df['KC (Textbook)'], sort=True)

    correct = list()
    for entry in df['Outcome']:
        if entry == 'CORRECT':
            correct.append(1)
        else:
            correct.append(0)
    df['correct'] = correct
    

    # Step 3 - Cross skill id with answer to form a synthetic feature
    df['skill_with_answer'] = df['skill'] * 2 + df['correct']

    # Step 4 - Convert to a sequence per user id and shift features 1 timestep
    num_students = df['Anon Student Id'].nunique()

    max_question_count = 0
    for student in df['Anon Student Id'].unique():
        std_count = len(df[df['Anon Student Id'] == student])
        if max_question_count < std_count:
            max_question_count = std_count
    
    max_skill_count = df['skill'].max()+1

    x = np.zeros((num_students, max_question_count, max_skill_count))
    
    students = df['Anon Student Id'].unique()
    i = 0
    for student in students:
        questions = df[df['Anon Student Id'] == student]
        j = 0
        for skill in questions['skill']:
            x[i][j][skill] = 1
            j += 1

        i += 1
    
    return x, df['correct'].values

if __name__ == '__main__':
    x, y = load_dataset(open('All_data.csv'))
    print(x)
    print()
    print(y)
