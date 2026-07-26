import os
import pickle
from music21 import converter, instrument, note, chord

DATASET_DIR = "midi_dataset"
OUTPUT_FILE = "notes_data.pkl"

def get_midi_files(root_dir):
    midi_files = []

    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".mid") or file.endswith(".midi"):
                midi_files.append(os.path.join(dirpath, file))

    return midi_files

def extract_notes(midi_files):
    notes = []

    for file in midi_files:
        try:
            midi = converter.parse(file)

            parts = instrument.partitionByInstrument(midi)

            if parts:
                elements = parts.parts[0].recurse()
            else:
                elements = midi.flat.notes

            for element in elements:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))

                elif isinstance(element, chord.Chord):
                    notes.append(
                        ".".join(str(n) for n in element.normalOrder)
                    )

        except Exception:
            continue

    return notes

if __name__ == "__main__":

    midi_files = get_midi_files(DATASET_DIR)

    print(f"Found {len(midi_files)} MIDI files")

    notes = extract_notes(midi_files)

    print(f"Extracted {len(notes)} notes/chords")

    with open(OUTPUT_FILE, "wb") as file:
        pickle.dump(notes, file)

    print(f"Saved as {OUTPUT_FILE}")
