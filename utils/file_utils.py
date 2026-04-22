import os
from config import DATA_PATH

def save_uploaded_file(uploaded_file):
    os.makedirs(DATA_PATH, exist_ok=True)
    path = os.path.join(DATA_PATH, uploaded_file.name)

    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return path
