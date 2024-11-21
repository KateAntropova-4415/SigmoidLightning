import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from ctypes import cdll, POINTER, c_float, c_int

from sigmoid_vector_C.sigmoid_vector import SigmoidVectorLibrary

# Укажите путь к скомпилированной библиотеке
lib_path = "sigmoid_vector_C/sigmoid_vector.dll"  # или "sigmoid_vector.dll" на Windows

# Загрузим библиотеку
sigmoid_vector_lib = SigmoidVectorLibrary(lib_path)


class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.loadtxt(data_path, delimiter=',', skiprows=1)  # Пример: загрузка данных из CSV

    def __getitem__(self, index):
        row = self.data[index]

        # Преобразование строки в input_array для функции compute
        input_array = np.array(row, dtype=np.float64)  # Приводим к типу, ожидаемому C++ функцией

        # Получаем output_array с помощью C++ функции compute
        output_array = sigmoid_vector_lib.compute(input_array)  # Вызов функции compute из библиотеки

        # Преобразуем результат в тензор PyTorch
        output_tensor = torch.tensor(output_array, dtype=torch.float32)

        # Возвращаем вход и выход
        return torch.tensor(input_array, dtype=torch.float32), output_tensor

    def __len__(self):
        return len(self.data)


class CustomDataModule(LightningDataModule):
    def __init__(self, data_path, batch_size):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = CustomDataset(self.data_path)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
