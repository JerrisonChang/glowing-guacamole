import pandas as pd

df = pd.read_csv('All_data.csv')
df['skill'], _ = pd.factorize(df['KC (Textbook)'])
df['user'], _ = pd.factorize(df['Anon Student Id'])

correct = list()
for entry in df['Outcome']:
    if entry == 'CORRECT':
        correct.append(1)
    else:
        correct.append(0)
df['correct'] = correct

questions = list()
for i in range(len(df.index)):
    name = str(df['Problem Name'].values[i])
    view = str(df['Problem View'].values[i])
    step = str(df['Step Name'].values[i])

    question = '_'.join([name,view,step])
    questions.append(question)

df['question'] = questions
df['item'], _ = pd.factorize(df['question'])

wins = list()
fails = list()
user_skill_dict = dict()
for i in range(len(df.index)):
    user = df['user'].values[1]
    skill = df['skill'].values[i]
    correct = df['correct'].values[i]

    if user not in user_skill_dict:
        user_skill_dict[user] = dict()

    if skill not in user_skill_dict[user]:
        user_skill_dict[user][skill] = {'wins': 0, 'fails': 0}
        
    if correct == 1:
        user_skill_dict[user][skill]["wins"] += 1
    else:
        user_skill_dict[user][skill]["fails"] += 1

    wins.append(user_skill_dict[user][skill]["wins"])
    fails.append(user_skill_dict[user][skill]["fails"])

df['wins'] = wins
df['fails'] = fails

df['user_id'] = df['user']
df['item_id'] = df['item']

df[['user', 'item', 'skill', 'correct', 'wins', 'fails']].to_csv('data.csv', index=False)
df[['user_id', 'item_id', 'correct']].to_csv('needed.csv', index=False)