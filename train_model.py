import pickle
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

NOTES_FILE = "notes_data.pkl"
MODEL_FILE = "music_model.keras"
MAPPING_FILE = "note_mapping.pkl"

SEQUENCE_LENGTH = 50
EPOCHS = 30
BATCH_SIZE = 64

with open(NOTES_FILE, "rb") as file:
    notes = pickle.load(file)

pitch_names = sorted(set(notes))

note_to_int = {
    note: number
    for number, note in enumerate(pitch_names)
}

network_input = []
network_output = []

for i in range(len(notes) - SEQUENCE_LENGTH):
    sequence_in = notes[i:i + SEQUENCE_LENGTH]
    sequence_out = notes[i + SEQUENCE_LENGTH]

    network_input.append(
        [note_to_int[n] for n in sequence_in]
    )

    network_output.append(
        note_to_int[sequence_out]
    )

n_vocab = len(pitch_names)

X = np.reshape(
    network_input,
    (len(network_input), SEQUENCE_LENGTH, 1)
)

X = X / float(n_vocab)

y = to_categorical(
    network_output,
    num_classes=n_vocab
)

model = Sequential()

model.add(
    LSTM(
        256,
        input_shape=(X.shape[1], X.shape[2]),
        return_sequences=True
    )
)

model.add(Dropout(0.3))

model.add(LSTM(256))

model.add(Dense(256))

model.add(Dropout(0.3))

model.add(Dense(n_vocab))

model.add(Activation("softmax"))

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam"
)

checkpoint = ModelCheckpoint(
    MODEL_FILE,
    monitor="loss",
    save_best_only=True,
    mode="min"
)

model.fit(
    X,
    y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint]
)

with open(MAPPING_FILE, "wb") as file:
    pickle.dump(
        {
            "note_to_int": note_to_int,
            "n_vocab": n_vocab,
            "sequence_length": SEQUENCE_LENGTH
        },
        file
    )

print("Model training complete.")
