import argparse

import tensorflow as tf

import config, dkt, metrics

def run(file):
    dataset, length, nb_features, nb_skills = config.load_dataset(fn=file, batch_size=1, shuffle=True)

    train_set, test_set, val_set = config.split_dataset(dataset=dataset, total_size=length, test_fraction=0.2, val_fraction=0.2)

    print("[----- COMPILING  ------]")
    model = dkt.DKTModel(nb_features=nb_features, nb_skills=nb_skills, hidden_units=100 ,dropout_rate=0.3)

    model.compile(
        optimizer='adam',
        metrics=[
            metrics.BinaryAccuracy(),
            metrics.AUC(),
            metrics.Precision(),
            metrics.Recall()
        ])

    print(model.summary())
    print("\n[-- COMPILING DONE  --]")

    print("\n[----- TRAINING ------]")
    model.fit(
        dataset=train_set,
        epochs=1,
        verbose=1,
        validation_data=val_set)
    print("\n[--- TRAINING DONE ---]")

    print("[----- TESTING  ------]")
    model.load_weights("weights/bestmodel")
    model.evaluate(dataset=test_set, verbose=1)
    print("\n[--- TESTING DONE  ---]")

if __name__ == "__main__":
    run('All_data.csv')
