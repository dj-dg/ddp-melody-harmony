import pickle
import os
import numpy as np
import tensorflow as tf
from tqdm import trange
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, TimeDistributed, Bidirectional
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall, F1Score, TopKCategoricalAccuracy, AUC
from tensorflow.keras.utils import to_categorical
from config import *

def enhanced_harmonic_matrix():
    matrix = np.zeros((25, 25), dtype=np.float32)
    
    for i in range(25):
        for j in range(25):
            # Handle rest class (index 24)
            if i == 24 or j == 24:
                matrix[i,j] = 1.0 if i == j else 0.1
                continue
                
            # Extract musical properties
            root_i = (i//2) % 12
            qual_i = i % 2  # 0=major, 1=minor
            root_j = (j//2) % 12
            qual_j = j % 2
            
            # Interval calculations
            semitone_dist = abs(root_i - root_j) % 12
            circle_steps = abs((root_i * 7) % 12 - (root_j * 7) % 12)  # Circle of fifths
            
            # Base score based on interval relationships
            if i == j:
                score = 1.0
            elif semitone_dist in [0,7,5]:  # Unison, fifth, fourth
                score = 0.9 - (circle_steps * 0.05)
            elif semitone_dist in [3,4,8,9]:  # Thirds/sixths
                score = 0.7 - (circle_steps * 0.03)
            elif semitone_dist in [1,2,10,11]:  # Seconds/sevenths
                score = 0.2
            else:  # Tritone
                score = 0
                
            # Quality modifiers
            if qual_i == qual_j:
                score *= 1.1  # Reward same quality
            else:
                if (semitone_dist in [3,4] and qual_i != qual_j):
                    score *= 0.8  # Penalize major/minor swaps in thirds
                    
            # Functional harmony bonuses
            if (semitone_dist == 7 and qual_i == 0):  # Dominant relationship
                score += 0.15
            if (semitone_dist == 5 and qual_j == 0):  # Subdominant
                score += 0.1
                
            matrix[i,j] = np.clip(score, 0, 1.0)
    
    return tf.constant(matrix, dtype=tf.float32)


def enhanced_harmonic_metric(y_true, y_pred):
    y_pred_class = tf.argmax(y_pred, axis=-1)
    y_true_class = tf.argmax(y_true, axis=-1)
    matrix = enhanced_harmonic_matrix()
    scores = tf.gather_nd(matrix, tf.stack([y_true_class, y_pred_class], axis=-1))
    return tf.reduce_mean(scores)


def harmonic_distance_metric(y_true, y_pred):
    """Custom metric evaluating harmonic relationship between chords"""
    y_pred_class = tf.argmax(y_pred, axis=-1)
    y_true_class = tf.argmax(y_true, axis=-1)
    
    # Simplified harmonic relationship matrix (25x25)
    harmonic_matrix = tf.constant(
        [[1.0 if i == j else 
          0.5 if abs((i//2)%12 - (j//2)%12) in [0,7,5] else  # Same, fifth, fourth
          0.2 for j in range(25)] for i in range(25)],
        dtype=tf.float32
    )
    
    scores = tf.gather_nd(
        harmonic_matrix,
        tf.stack([y_true_class, y_pred_class], axis=-1)
    )
    return tf.reduce_mean(scores)

# from tensorflow.keras.metrics import FBetaScore

# def wrapped_f1_score(num_classes=25, beta=1.0, name='f1_score'):
#     f1 = FBetaScore(num_classes=num_classes, beta=beta, name=name)
    
#     def metric_fn(y_true, y_pred):
#         # Reshape from (batch_size, 8, 25) → (batch_size*8, 25)
#         y_true_reshaped = tf.reshape(y_true, [-1, num_classes])
#         y_pred_reshaped = tf.reshape(y_pred, [-1, num_classes])
#         return f1(y_true_reshaped, y_pred_reshaped)
    
#     metric_fn.__name__ = name
#     return metric_fn

def macro_f1_score(y_true, y_pred):
    # Reshape to 2D tensor (batch*seq, num_classes)
    y_true = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
    y_pred = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
    
    # Convert to class indices and ensure dtype consistency
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    dtype = y_true.dtype  # Preserve original dtype (int64)
    
    num_classes = tf.reduce_max(y_true) + 1

    def f1_for_class(i):
        # Cast index to match y_true/y_pred dtype
        i = tf.cast(i, dtype)
        true_pos = tf.reduce_sum(tf.cast((y_pred == i) & (y_true == i), tf.float32))
        false_pos = tf.reduce_sum(tf.cast((y_pred == i) & (y_true != i), tf.float32))
        false_neg = tf.reduce_sum(tf.cast((y_pred != i) & (y_true == i), tf.float32))
        precision = true_pos / (true_pos + false_pos + 1e-8)
        recall = true_pos / (true_pos + false_neg + 1e-8)
        return 2 * precision * recall / (precision + recall + 1e-8)

    # Generate class indices with matching dtype
    class_indices = tf.range(num_classes, dtype=dtype)
    f1s = tf.map_fn(f1_for_class, class_indices, dtype=tf.float32)
    
    return tf.reduce_mean(f1s)


def create_training_data(corpus_path=CORPUS_PATH, val_ratio=VAL_RATIO):
    with open(corpus_path, "rb") as filepath:
        data_corpus = pickle.load(filepath)
    input_melody = []
    output_chord = []
    input_melody_val = []
    output_chord_val = []
    cnt = 0
    np.random.seed(0)
    for songs_idx in trange(len(data_corpus)):
        song = data_corpus[songs_idx]
        if np.random.rand() > val_ratio:
            train_or_val = 'train'
        else:
            train_or_val = 'val'
        song_melody = song[0][0]
        song_chord = song[0][1]
        # Window size 8 for 2 chords per measure (half-measure resolution)
        for idx in range(len(song_melody)-7):
            melody = [np.array(v, dtype=np.float32) for v in song_melody[idx:idx+8]]
            chord = song_chord[idx:idx+8]
            if train_or_val == 'train':
                input_melody.append(melody)
                output_chord.append(chord)
            else:
                input_melody_val.append(melody)
                output_chord_val.append(chord)
            cnt += 1
    print("Successfully read %d pieces" %(cnt))
    onehot_chord = to_categorical(output_chord, num_classes=25)
    if len(input_melody_val)!=0:
        onehot_chord_val = to_categorical(output_chord_val, num_classes=25)
    else:
        onehot_chord_val = None
    return (input_melody, onehot_chord), (input_melody_val, onehot_chord_val)

def build_model(rnn_size=RNN_SIZE, num_layers=NUM_LAYERS, weights_path=None):
    input_melody = Input(shape=(8, 12), name='input_melody')
    melody = TimeDistributed(Dense(12))(input_melody)
    
    for idx in range(num_layers):
        melody = Bidirectional(LSTM(units=rnn_size, 
                                  return_sequences=True, 
                                  name=f'melody_{idx+1}'))(melody)
        melody = TimeDistributed(Dense(units=rnn_size, activation='tanh'))(melody)
        melody = Dropout(0.2)(melody)
    
    output_layer = TimeDistributed(Dense(25, activation='softmax'))(melody)
    
    model = Model(inputs=input_melody, outputs=output_layer)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',

            macro_f1_score,
            TopKCategoricalAccuracy(k=3, name='top3_acc'),
            AUC(name='auc', multi_label=False),
            harmonic_distance_metric,
            enhanced_harmonic_metric
        ]
    )
    
    if weights_path is None:
        model.summary()
    else:
        model.load_weights(weights_path)
    
    return model

def train_model(data, data_val, model=None, batch_size=BATCH_SIZE, epochs=EPOCHS, 
                verbose=1, weights_path=WEIGHTS_PATH):
    # Create model if not provided
    if model is None:
        model = build_model()

    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            print("Checkpoint loaded")
        except:
            os.remove(weights_path)
            print("Checkpoint deleted")
    if len(data_val[0])!=0:
        monitor = 'val_loss'
    else:
        monitor = 'loss'
    checkpoint = ModelCheckpoint(filepath=weights_path, monitor=monitor, verbose=0, save_best_only=True, mode='min')
    if len(data_val[0])!=0:
        history = model.fit(
            x={'input_melody': np.array(data[0], dtype=np.float32)},
            y=np.array(data[1], dtype=np.float32),
            validation_data=({'input_melody': np.array(data_val[0], dtype=np.float32)}, np.array(data_val[1], dtype=np.float32)),
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=[checkpoint]
        )
    else:
        history = model.fit(
            x={'input_melody': np.array(data[0], dtype=np.float32)},
            y=np.array(data[1], dtype=np.float32),
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=[checkpoint]
        )
    return model, history  # Return both model and history

if __name__ == "__main__":
    data, data_val = create_training_data()
    
    # Initialize model first
    model = build_model()
    
    # Pass model to train and receive updated model
    model, history = train_model(data, data_val, model=model)
    
    # Now evaluate using the returned model
    if len(data_val[0]) > 0:
        results = model.evaluate(
            x=np.array(data_val[0], dtype=np.float32),
            y=np.array(data_val[1], dtype=np.float32),
            verbose=0
        )
        print("\nFinal Validation Metrics:")
        print(f"• Accuracy: {results[1]:.4f}")
        # print(f"• Precision: {results[2]:.4f}")
        # print(f"• Recall: {results[3]:.4f}")
        print(f"• F1 Score: {results[2]:.4f}")
        print(f"• Top-3 Accuracy: {results[3]:.4f}")
        print(f"• AUC: {results[4]:.4f}")
        print(f"• Harmonic Score: {results[5]:.4f}")
        print(f"• Enhanced Harmonic Score: {results[6]:.4f}")


