import argparse

import tensorflow as tf
import tensorflow_datasets as tfds

import deepkt, data_util, metrics

import csv

def run(args):
    all_lstm_history = dict()    
    all_rnn_history = dict()
    for i in range(1,16):
        dataset, length, nb_features, nb_skills = data_util.load_dataset(fn=args.f,
                                                                        batch_size=args.batch_size,
                                                                        shuffle=False, 
                                                                        num_kc=i)

        train_set, test_set = data_util.split_dataset(dataset=dataset,
                                                        total_size=length,
                                                        test_fraction=args.test_split)

        print("\n[----- COMPILING  ------]")
        lstm = deepkt.DKTModel(nb_features=nb_features,
                                nb_skills=nb_skills,
                                hidden_units=args.hidden_units, 
                                LSTM=True)
        lstm.compile(
            optimizer='adam',
            metrics=[
                metrics.BinaryAccuracy(),
                metrics.AUC(),
                metrics.Precision(),
                metrics.Recall()
            ])

        rnn = deepkt.DKTModel(nb_features=nb_features,
                                nb_skills=nb_skills,
                                hidden_units=args.hidden_units)
        rnn.compile(
            optimizer='adam',
            metrics=[
                metrics.BinaryAccuracy(),
                metrics.AUC(),
                metrics.Precision(),
                metrics.Recall()
            ])

        print(lstm.summary())
        print(rnn.summary())
        print("\n[-- COMPILING DONE  --]")

        print("\n[----- TRAINING ------]")
        lstm_history = lstm.fit(
            dataset=train_set,
            epochs=args.epochs,
            verbose=args.v)

        rnn_history = rnn.fit(
            dataset=train_set,
            epochs=args.epochs,
            verbose=args.v)
        print("\n[--- TRAINING DONE ---]")

        print("\n[----- TESTING  ------]")
        print("Number of KCs: ", i)
        lstm.evaluate(dataset=test_set, verbose=args.v)
        rnn.evaluate(dataset=test_set, verbose=args.v)
        print("\n[--- TESTING DONE  ---]")

        all_lstm_history[i] = lstm_history.history
        all_rnn_history[i] = rnn_history.history

        if i == 15:
            answers = data_util.get_answers(args.f)

            lstm_preds = lstm.get_predictions(test_set)
            rnn_preds = rnn.get_predictions(test_set)

            with open("lstm_roc.csv", 'w') as f:
                writer = csv.DictWriter(f, fieldnames=['y_actual', 'y_pred'])
                writer.writeheader()
                for i in range(len(answers)):
                    student_answers = answers[i]
                    student = lstm_preds[i][0]
                    for j in range(len(student)):
                        question = student_answers[j]
                        skill = question[0]
                        y = question[1]
                        y_pred = student[j][skill]

                        writer.writerow({'y_pred': y_pred, 'y_actual': y})

            with open("rnn_roc.csv", 'w') as f:
                writer = csv.DictWriter(f, fieldnames=['y_actual', 'y_pred'])
                writer.writeheader()
                for i in range(len(answers)):
                    student_answers = answers[i]
                    student = rnn_preds[i][0]
                    for j in range(len(student)):
                        question = student_answers[j]
                        skill = question[0]
                        y = question[1]
                        y_pred = student[j][skill]

                        writer.writerow({'y_pred': y_pred, 'y_actual': y})

    write_accuracy(all_lstm_history, all_rnn_history)


def parse_args():
    parser = argparse.ArgumentParser(prog="DeepKT Example")

    parser.add_argument("-f",
                        type=str,
                        default="All_data.csv",
                        help="the path to the data")

    parser.add_argument("-v",
                        type=int,
                        default=1,
                        help="verbosity mode [0, 1, 2].")

    model_group = parser.add_argument_group(title="Model arguments.")

    model_group.add_argument("--hidden_units",
                             type=int,
                             default=100,
                             help="number of units of the LSTM layer.")

    train_group = parser.add_argument_group(title="Training arguments.")
    train_group.add_argument("--batch_size",
                             type=int,
                             default=1,
                             help="number of elements to combine in a single batch.")

    train_group.add_argument("--epochs",
                             type=int,
                             default=1,
                             help="number of epochs to train.")

    train_group.add_argument("--test_split",
                             type=float,
                             default=.2,
                             help="fraction of data to be used for testing (0, 1).")

    return parser.parse_args()

def write_accuracy(all_lstm_history, all_rnn_history):
    with open("lstm_accuracy.csv", 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['num_kc', 'accuracy'])
        writer.writeheader()
        for kc in all_lstm_history:
            accuracy = all_lstm_history[kc]['binary_accuracy'][0]
            writer.writerow({'num_kc': kc, 'accuracy': accuracy})

    with open("rnn_accuracy.csv", 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['num_kc', 'accuracy'])
        writer.writeheader()
        for kc in all_rnn_history:
            accuracy = all_rnn_history[kc]['binary_accuracy'][0]
            writer.writerow({'num_kc': kc, 'accuracy': accuracy})

if __name__ == "__main__":
    run(parse_args())