import numpy as np
import argparse

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Compare u.data files')
    parser.add_argument('file1', type=str, help='First u.data file')
    parser.add_argument('file2', type=str, help='Second u.data file')

    args = parser.parse_args()

    # Read the files
    u_data_1 = np.fromfile(args.file1, dtype=np.float32)
    u_data_2 = np.fromfile(args.file2, dtype=np.float32)

    # Check if the file is not all zeros
    if np.all(u_data_1 == 0):
        raise ValueError("The file {} is all zeros".format(args.file1))

    if np.all(u_data_2 == 0):
        raise ValueError("The file {} is all zeros".format(args.file2))

    # Compare the files
    difference = np.max(np.abs(u_data_1 - u_data_2))
    print("Diferença máxima entre os dados de u.data dos arquivos 1 e 2:", difference)
