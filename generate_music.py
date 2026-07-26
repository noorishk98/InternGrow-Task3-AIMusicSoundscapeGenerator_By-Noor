import pickle
import random
import numpy as np

from tensorflow.keras.models import load_model
from music21 import stream, note, chord, tempo, instrument

MODEL_FILE = "music_model.keras"
MAPPING_FILE = "note_mapping.pkl"

MOODS = {
    "1": {
        "name": "Happy",
        "tempo": 140,
        "instrument": instrument.Piano(),
        "temperature": 0.9
    },
    "2": {
        "name": "Sad",
        "tempo": 65,
        "instrument": instrument.Piano(),
        "temperature": 0.6
    },
    "3": {
        "name": "Emotional",
        "tempo": 90,
        "instrument": instrument.Violin(),
        "temperature": 1.1
    },
    "4": {
        "name": "Sleepy",
        "tempo": 50,
        "instrument": instrument.Flute(),
        "temperature": 0.4
    },
    "5": {
        "name": "Energetic",
        "tempo": 160,
        "instrument": instrument.ElectricGuitar(),
        "temperature": 1.3
    }
}

def sample_with_temperature(predictions, temperature):
    predictions = np.asarray(predictions).astype("float64")
    predictions = np.log(predictions + 1e-9) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, predictions, 1)

    return np.argmax(probas)

with open(MAPPING_FILE, "rb") as file:
    mapping = pickle.load(file)

note_to_int = mapping["note_to_int"]
n_vocab = mapping["n_vocab"]
sequence_length = mapping["sequence_length"]

int_to_note = {
    value: key
    for key, value in note_to_int.items()
}

with open("notes_data.pkl", "rb") as file:
    notes = pickle.load(file)

network_input = []

for i in range(len(notes) - sequence_length):
    sequence = notes[i:i + sequence_length]

    network_input.append(
        [note_to_int[n] for n in sequence]
    )

model = load_model(MODEL_FILE)

print("\nChoose Mood")
print("1. Happy")
print("2. Sad")
print("3. Emotional")
print("4. Sleepy")
print("5. Energetic")

choice = input("Enter choice: ")

mood = MOODS.get(choice, MOODS["1"])

num_notes = 200

start = random.randint(
    0,
    len(network_input) - 1
)

pattern = network_input[start]

prediction_output = []

for _ in range(num_notes):

    prediction_input = np.reshape(
        pattern,
        (1, len(pattern), 1)
    )

    prediction_input = prediction_input / float(n_vocab)

    prediction = model.predict(
        prediction_input,
        verbose=0
    )[0]

    index = sample_with_temperature(
        prediction,
        mood["temperature"]
    )

    result = int_to_note[index]

    prediction_output.append(result)

    pattern.append(index)
    pattern = pattern[1:]

offset = 0
output_notes = []

for pattern in prediction_output:

    if "." in pattern or pattern.isdigit():

        notes_in_chord = pattern.split(".")
        chord_notes = []

        for current_note in notes_in_chord:

            new_note = note.Note(
                int(current_note)
            )

            new_note.storedInstrument = mood["instrument"]

            chord_notes.append(new_note)

        new_chord = chord.Chord(chord_notes)
        new_chord.offset = offset

        output_notes.append(new_chord)

    else:

        new_note = note.Note(pattern)

        new_note.offset = offset
        new_note.storedInstrument = mood["instrument"]

        output_notes.append(new_note)

    offset += 0.5

midi_stream = stream.Stream()

midi_stream.append(
    tempo.MetronomeMark(
        number=mood["tempo"]
    )
)

midi_stream.append(
    mood["instrument"]
)

for item in output_notes:
    midi_stream.append(item)

output_file = (
    "generated_" +
    mood["name"].lower() +
    ".mid"
)

midi_stream.write(
    "midi",
    fp=output_file
)

print(f"\nMusic saved as {output_file}")
