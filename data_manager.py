import os
import re

from PIL import Image
import h5py
import pandas as pd
import numpy as np

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DATABASE_FOLDER_NAME = 'database'
CSV_FOLDER_NAME = 'dataframes'
DEFAULT_SEP = ";"
DEFAULT_DECIMAL = ","


def _add_folder(root_directory, folder_name):
    save_path = os.path.join(root_directory, folder_name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    return save_path


def save_plot(figure, folder_name, name):
    save_path = os.path.join(SCRIPT_PATH, 'plots')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, folder_name)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, name)

    if os.path.isfile(save_path):
        os.remove(save_path)
    figure.savefig(save_path)


def save_dataframe(df, file_name):
    """ сохраняет csv файл
    df : pandas.Dataframe
        таблица данных
    file_name : str
        Название csv или xlsx файла в который нужно сохранить df
    
    Returns
    -------
    """
    folder_path = _add_folder(SCRIPT_PATH, CSV_FOLDER_NAME)
    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path):
        os.remove(file_path)

    if file_name[-3:]=="csv":
        df.to_csv(file_path, index=False, sep=DEFAULT_SEP, decimal=DEFAULT_DECIMAL)
    elif file_name[-3:]=="lsx":
        df.to_excel(file_path)
    else:
        raise ValueError(f"file_name must consist .csv or .xlsx")


def load_datafrme(csv_file_name):
    """ загружает csv файл
    csv_file_name : str defaut "train_dataframe.csv"
        Название csv файла
    
    Returns
    -------
    out : pandas.Dataframe
        таблица данных
    """
    file_path = os.path.join(SCRIPT_PATH, CSV_FOLDER_NAME, csv_file_name)

    return pd.read_csv(file_path, sep=DEFAULT_SEP, decimal=DEFAULT_DECIMAL)


def get_img2d_from_database(file_name):
    path = os.path.join(SCRIPT_PATH, DATABASE_FOLDER_NAME, file_name)

    img2d_gray = np.array(Image.open(path))

    return img2d_gray