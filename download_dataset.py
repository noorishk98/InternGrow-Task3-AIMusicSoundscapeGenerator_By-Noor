import os
import zipfile
import requests

URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
ZIP_FILE = "maestro-v3.0.0-midi.zip"
OUTPUT_DIR = "midi_dataset"

def download_dataset():
    if os.path.exists(ZIP_FILE):
        print("Dataset already downloaded.")
        return

    response = requests.get(URL, stream=True)

    with open(ZIP_FILE, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)

    print("Download completed.")

def extract_dataset():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(OUTPUT_DIR)

    print("Extraction completed.")

if __name__ == "__main__":
    download_dataset()
    extract_dataset()
