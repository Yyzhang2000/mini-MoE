import os
import requests
import tiktoken
import numpy as np

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_file():
    input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
    if not os.path.exists(input_file_path):
        with open(input_file_path, "w") as f:
            f.write(requests.get(DATA_URL).text)

    return input_file_path


def load_and_tokenize(input_file_path):
    if not os.path.exists(input_file_path):
        input_file_path = download_file()

    with open(input_file_path, "r", encoding="utf-8") as f:
        data = f.read()

    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    # Encode Text
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)

    print(f"Train data: {len(train_ids)}")
    print(f"Val data: {len(val_ids)}")

    # Export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    train_ids.tofile(os.path.join(os.path.dirname(__file__), "shakespeare-train.bin"))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), "shakespeare-val.bin"))

    print("Data saved to bin files")


if __name__ == "__main__":
    input_file_path = download_file()
    load_and_tokenize(input_file_path)
