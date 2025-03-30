import os
import requests
import tiktoken
import numpy as np

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_and_tokenize_shakespeare():
    input_file_path = os.path.join(os.path.dirname(__file__), "shakespear.txt")

    if not os.path.exists(input_file_path):
        with open(input_file_path, "w") as f:
            f.write(requests.get(DATA_URL).text)
    print("Shakespeare dataset downloaded.")

    with open(input_file_path, "r") as f:
        data = f.read()

    print("Shakespeare dataset loaded.")

    n = len(data)

    train_data = data[: int(0.9 * n)]
    val_data = data[int(0.9 * n) :]

    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"Train data length: {len(train_data)}")
    print(f"Validation data length: {len(val_data)}")

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    train_ids.tofile(os.path.join(os.path.dirname(__file__), "shakespeare_train.bin"))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), "shakespeare_val.bin"))
    print("Shakespeare dataset tokenized and saved.")


def get_shakespeare_data():
    """
    Returns the Shakespeare dataset as a tuple of (train_ids, val_ids).
    """
    if not os.path.exists(
        os.path.join(os.path.dirname(__file__), "train.bin")
    ) or not os.path.exists(os.path.join(os.path.dirname(__file__), "val.bin")):
        download_and_tokenize_shakespeare()

    train_ids = np.fromfile(
        os.path.join(os.path.dirname(__file__), "train.bin"),
        dtype=np.uint16,
    )
    val_ids = np.fromfile(
        os.path.join(os.path.dirname(__file__), "val.bin"), dtype=np.uint16
    )
    return train_ids, val_ids


if __name__ == "__main__":
    download_and_tokenize_shakespeare()
