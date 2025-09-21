import csv
import os
from app.config import LOG_PATH

def save_to_csv(data_dict, file_path=LOG_PATH):
    """
    Appends a dictionary of evaluation metrics to a CSV file.
    If the file does not exist, it creates it and writes the header.
    """
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data_dict.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(data_dict)