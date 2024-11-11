import os
import pandas as pd

def load_files(folder_path):
    file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith((".xlsx", "xls"))]
    file_paths.sort(key=lambda x: os.path.basename(x).split('_')[0])
    return file_paths