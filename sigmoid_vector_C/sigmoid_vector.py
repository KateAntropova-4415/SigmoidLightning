import ctypes
import numpy as np


class SigmoidVectorLibrary:
    def __init__(self, lib_path: str):
        # Загрузка библиотеки
        self.lib = ctypes.CDLL(lib_path)

        # Настройка типов аргументов и возвращаемого значения
        self.lib.sigmoid_vector.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # указатель на входной массив
            ctypes.POINTER(ctypes.c_double),  # указатель на выходной массив
            ctypes.c_int  # размер массива
        ]

    def compute(self, input_array: np.ndarray) -> np.ndarray:
        if not isinstance(input_array, np.ndarray) or input_array.dtype != np.float64:
            raise ValueError("Input must be a numpy array with dtype=np.float64")

        size = input_array.size
        output_array = np.empty(size, dtype=np.float64)

        # Передача данных в C++
        input_ptr = input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Вызов функции
        self.lib.sigmoid_vector(input_ptr, output_ptr, size)

        return output_array
