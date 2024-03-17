import torch
import numpy as np
import sys
import json
import argparse
from model import CharRNN  # Импортируем модель
import torch.nn as nn


HIDDEN_SIZE = 100
NUM_LAYERS = 4

# Функция для загрузки данных
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    return data

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output.view(1, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

# Функция для тестирования модели на тестовых данных
def test_model(model, char_to_ix, ix_to_char, test_data):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        loss_list = []
        for _ in range(5):  # Количество имён, которые нужно сгенерировать
            hidden = model.init_hidden(1)  # Инициализация скрытого состояния для одного элемента
            start_input = torch.tensor([[char_to_ix[np.random.choice(list(char_to_ix.keys()))]]], dtype=torch.long)  # Случайный начальный символ имени
            predicted = ix_to_char[start_input.item()] if start_input.item() in ix_to_char else '?'
            loss = 0.0
            for _ in range(9):  # Каждый раз предсказывается имя из 10 букв
                output, hidden = model(start_input, hidden)  # Используем скрытое состояние из предыдущего шага
                probs = torch.softmax(output, dim=-1)
                _, next_char = torch.max(probs[0, -1], dim=-1)
                start_input = next_char.unsqueeze(0)
                predicted += ix_to_char[start_input.item()] if start_input.item() in ix_to_char else '?'
                loss += criterion(output.view(1, -1), start_input.view(-1))
            loss /= 10
            loss_list.append(loss.item())
            print(f"Предсказанное: {predicted}, Test Loss: {loss.item():.4f}")
        avg_loss = sum(loss_list) / len(loss_list)
        print(f"Average Test Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--char-to-ix', type=str, required=True)
    parser.add_argument('model_file', type=str)
    parser.add_argument('test_file', type=str)
    args = parser.parse_args()

    # Загрузка char_to_ix из файла
    with open(args.char_to_ix, 'r') as f:
        char_to_ix = json.load(f)

    # Создание вокабуляра
    chars = sorted(list(set(load_data(args.test_file))))
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    # Загрузка модели из файла
    model = CharRNN(len(char_to_ix), HIDDEN_SIZE, len(char_to_ix), NUM_LAYERS)
    model.load_state_dict(torch.load(args.model_file, map_location=torch.device('cpu')))

    # Загрузка тестовых данных
    test_data = load_data(args.test_file)

    # Тестирование модели
    test_model(model, char_to_ix, ix_to_char, test_data)
