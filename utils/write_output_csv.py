import csv
import os
import numpy as np
import pandas as pd
import torch

from utils.cell2attr import cell2attr


def write_output_header(input_path: str, output_path: str):
    dat_files = [f for f in os.listdir(input_path) if f.endswith('.dat')]
    file_names = [os.path.splitext(f)[0] for f in dat_files]
    file_names.insert(0, 'cell_num')
    file_names_array = np.array([file_names])
    np.savetxt(output_path + '/output.csv', file_names_array, delimiter=',', fmt='%s', comments='')


def write_content(input_path: str, output_path: str):
    # 写入文件头
    write_output_header(input_path, output_path)
    origin_tensor = cell2attr('../datasets/input/property/')
    array = origin_tensor.numpy()
    with open(output_path + '/output.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        num = 1
        for row in array:
            writer.writerow(['cell_' + str(num)] + list(row))
            num += 1


def add_tensor_to_csv(tensor, column_name):
    df = pd.read_csv('../datasets/output/output.csv')
    tensor = tensor.flatten()
    tensor_series = pd.Series(tensor.numpy(), name=column_name)
    if len(tensor_series) != len(df):
        raise ValueError("Tensor length does not match the number of rows in the CSV file.")
    df[column_name] = tensor_series
    df.to_csv('../datasets/output/output.csv', index=False)


if __name__ == '__main__':
    write_content('../datasets/input/property', '../datasets/output')
