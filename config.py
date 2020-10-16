import pandas as pd
import tensorflow as tf
import numpy as np


MASK_VALUE = -1.  # The masking value cannot be zero.


def load_dataset(fn, batch_size=32, shuffle=True):
    df = pd.read_csv(fn)

    if "KC (Textbook)" not in df.columns:
        raise KeyError(f"The column 'KC (Textbook)' was not found on {fn}")
    if "Outcome" not in df.columns:
        raise KeyError(f"The column 'Outcome' was not found on {fn}")
    if "Anon Student Id" not in df.columns:
        raise KeyError(f"The column 'Anon Student Id' was not found on {fn}")

    if not (df['Outcome'].isin(['CORRECT', 'INCORRECT'])).all():
        raise KeyError(f"The values of the column 'Outcome' must be 'CORRECT' or 'INCORRECT.")

    # Step 1.1 - Remove questions without skill
    df.dropna(subset=['KC (Textbook)'], inplace=True)

    # Step 1.2 - Remove users with a single answer
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
    seq = df.groupby('Anon Student Id').apply(
        lambda r: (
            r['skill_with_answer'].values[:-1],
            r['skill'].values[1:],
            r['correct'].values[1:],
        )
    )
    nb_users = len(seq)

    # Step 5 - Get Tensorflow Dataset
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: seq,
        output_types=(tf.int32, tf.int32, tf.float32)
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=nb_users)

    # Step 6 - Encode categorical features and merge skills with labels to compute target loss.
    # More info: https://github.com/tensorflow/tensorflow/issues/32142
    features_depth = df['skill_with_answer'].max() + 1
    skill_depth = df['skill'].max() + 1

    dataset = dataset.map(
        lambda feat, skill, label: (
            tf.one_hot(feat, depth=features_depth),
            tf.concat(
                values=[
                    tf.one_hot(skill, depth=skill_depth),
                    tf.expand_dims(label, -1)
                ],
                axis=-1
            )
        )
    )

    # Step 7 - Pad sequences per batch
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padding_values=(MASK_VALUE, MASK_VALUE),
        padded_shapes=([None, None], [None, None]),
        drop_remainder=True
    )

    length = nb_users // batch_size
    return dataset, nb_users, features_depth, skill_depth


def split_dataset(dataset, total_size, test_fraction, val_fraction=None):
    def split(dataset, split_size):
        split_set = dataset.take(split_size)
        dataset = dataset.skip(split_size)
        return dataset, split_set

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between (0, 1)")

    if val_fraction is not None and not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between (0, 1)")

    test_size = np.ceil(test_fraction * total_size)
    train_size = total_size - test_size

    if test_size == 0 or train_size == 0:
        raise ValueError(
            "The train and test datasets must have at least 1 element. Reduce the split fraction or get more data.")

    train_set, test_set = split(dataset, test_size)

    val_set = None
    if val_fraction:
        val_size = np.ceil(train_size * val_fraction)
        train_set, val_set = split(train_set, val_size)

    return train_set, test_set, val_set


def get_target(y_true, y_pred):
    # Get skills and labels from y_true
    mask = 1. - tf.cast(tf.equal(y_true, MASK_VALUE), y_true.dtype)
    y_true = y_true * mask

    skills, y_true = tf.split(y_true, num_or_size_splits=[-1, 1], axis=-1)

    # Get predictions for each skill
    y_pred = tf.reduce_sum(y_pred * skills, axis=-1, keepdims=True)

    return y_true, y_pred

if __name__ == '__main__':
    load_dataset(open('All_data.csv'))
