"""
Helper functions to read and write large pickle files based on:
https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
"""

import pickle
import os
MAX_BYTES = 2 ** 31 - 1

def read_large_pickle(path):
    bytes_in = bytearray(0)
    input_size = os.path.getsize(path)
    with open(path, 'rb') as f_in:
        for _ in range(0, input_size, MAX_BYTES):
            bytes_in += f_in.read(MAX_BYTES)
    return pickle.loads(bytes_in)

def write_large_pickle(data, path):
    bytes_out = pickle.dumps(data)
    with open(path, 'wb') as f_out:
        for idx in range(0, len(bytes_out), MAX_BYTES):
            f_out.write(bytes_out[idx:idx + MAX_BYTES])