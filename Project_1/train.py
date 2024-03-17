import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import CharRNN
import numpy as np
import os


# Функция для загрузки данных
def load_data(filepath):
  with open(filepath, 'r', encoding='utf-8') as f:
    data = f.read()
  return data


# Создание вокабуляра
def create_vocab(data):
  chars = sorted(list(set(data)))
  char_to_ix = {ch: i for i, ch in enumerate(chars)}
  ix_to_char = {i: ch for i, ch in enumerate(chars)}
  return char_to_ix, ix_to_char


# Кодирование данных
def encode_data(data, char_to_ix):
  encoded = [char_to_ix[ch] for ch in data]
  return encoded


# Определение класса датасета
class NameDataset(Dataset):

  def __init__(self, data, seq_length):
    self.data = data
    self.seq_length = seq_length

  def __len__(self):
    return len(self.data) - self.seq_length

  def __getitem__(self, index):
    return (torch.tensor(self.data[index:index + self.seq_length]),
            torch.tensor(self.data[index + 1:index + self.seq_length + 1]))


# Параметры модели
HIDDEN_SIZE = 100
NUM_LAYERS = 4
BATCH_SIZE = 3
SEQ_LENGTH = 10
NUM_EPOCHS = 3
LR = 0.005

# Загрузка и подготовка данных
data = load_data('train.txt')
char_to_ix, ix_to_char = create_vocab(data)
encoded_data = encode_data(data, char_to_ix)

dataset = NameDataset(encoded_data, SEQ_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Инициализация модели, функции потерь и оптимизатора
model = CharRNN(len(char_to_ix), HIDDEN_SIZE, len(char_to_ix), NUM_LAYERS)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

torch.autograd.set_detect_anomaly(True)

# Обучение
model.train()
for epoch in range(NUM_EPOCHS):
  hidden = model.init_hidden(BATCH_SIZE)
  for batch, (inp, target) in enumerate(dataloader):
    optimizer.zero_grad()
    output, hidden = model(inp, hidden)
    target = target[:, :SEQ_LENGTH].clone()

    loss = criterion(output.permute(0, 2, 1), target)

    loss.backward(
        retain_graph=True
    )  # Specify retain_graph=True to retain the computational graph

    optimizer.step()
    if batch % 100 == 0:
      print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')
# Сохранение модели
torch.save(model.state_dict(), 'model.torch')

# Передача char_to_ix в test.py
with open('char_to_ix.json', 'w') as f:
  json.dump(char_to_ix, f)
  
# Save the command to be executed to run `test.py` with `char_to_ix`
command = f'python test.py model.torch test.txt --char-to-ix char_to_ix.json'
# Execute the command to run `test.py` with `char_to_ix`
os.system(command)