import json
from pathlib import Path
import numpy as np
import torch
import torch.utils.data as utils_data
import torch.nn as nn


def load_data(data_path, label):
    data_list = []
    data_label = []
    for file_path in data_path.glob('*.json'):
        file = open(file_path, 'r')
        content = file.read()
        a = json.loads(content)
        file.close()
        b = a['people']
        c = np.array(b[0]['pose_keypoints_2d']).reshape(-1, 3)
        data = []
        for i in [1, 8, 9, 10, 11, 12, 13, 14]:
            data.append(c[i][0:2])
        data = np.array(data).reshape(1, -1)
        data_list.append(data[0])
        data_label.append(label)
    return data_list, data_label


class FCNNModel(nn.Module):
    def __init__(self, input_layer_size, hidden_layer_size, num_classes):
        super(FCNNModel, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_layer_size, hidden_layer_size),
            nn.BatchNorm1d(hidden_layer_size),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_layer_size, num_classes),
        )

    def forward(self, x):
        x = self.fc1(x)
        out = self.fc2(x)
        return out


def train():
    best_accuracy = 0
    for epoch in range(500):
        running_loss = 0.0
        train_acc = 0.0
        for step, (batch_image, batch_label) in enumerate(train_loader):
            model.train()
            batch_output = model(batch_image)
            batch_loss = loss_func(batch_output, batch_label)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()

            # train accuracy
            _, train_predicted = torch.max(batch_output.data, 1)
            train_acc += (train_predicted == batch_label).sum().item()

        # ----------test----------
        model.eval()
        test_acc = 0.0
        for test_image, test_label in test_loader:
            test_output = model(test_image)
            _, predicted = torch.max(test_output.data, 1)
            test_acc += (predicted == test_label).sum().item()
        test_acc /= test_size

        print('epoch={:d}\ttrain loss={:.6f}\ttrain accuracy={:.3f}\ttest accuracy={:.3f}'.format(
            epoch, running_loss, train_acc, test_acc))

        if test_acc >= best_accuracy:
            torch.save(model, './FCNN_model.pth')
            best_accuracy = test_acc


def test():
    output_dict = {0: 'sit', 1: 'stand'}
    model = torch.load('./FCNN_model.pth')
    input_path = input('please entry the picture path(enter 0 to exit):\n')
    while True:
        if input_path == '0':
            break
        input_path = Path(input_path)
        if input_path.is_file():
            file = open(input_path, 'r')
            content = file.read()
            a = json.loads(content)
            file.close()
            b = a['people']
            c = np.array(b[0]['pose_keypoints_2d']).reshape(-1, 3)
            data = []
            for i in [1, 8, 9, 10, 11, 12, 13, 14]:
                data.append(c[i][0:2])
            data = np.array(data).reshape(1, -1)
            output = model(torch.Tensor(data))
            _, predicted = torch.max(output.data, 1)
            print('it is {}\n'.format(output_dict[int(predicted)]))
            input_path = input('please entry the picture path(enter 0 to exit):\n')
        else:
            input_path = input('please entry the picture path(enter 0 to exit):\n')


if __name__ == '__main__':
    # load data
    json_path = Path.cwd() / 'json_folder'
    sit_path = json_path / 'sit'
    stand_path = json_path / 'stand'
    sit_data, sit_label = load_data(sit_path, 0)
    stand_data, stand_label = load_data(stand_path, 1)
    data_list = sit_data + stand_data
    label_list = sit_label + stand_label

    # load model
    model = FCNNModel(input_layer_size=16, hidden_layer_size=9, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_func = nn.CrossEntropyLoss()

    # train
    dataset_size = 6
    dataset = utils_data.TensorDataset(torch.Tensor(data_list), torch.LongTensor(label_list))
    split_ratio = 0.8
    train_size = int(split_ratio * dataset_size)
    test_size = dataset_size - train_size
    train_set, test_set = utils_data.random_split(dataset, [train_size, test_size])
    train_loader = utils_data.DataLoader(dataset=train_set, batch_size=8, shuffle=True)
    test_loader = utils_data.DataLoader(dataset=test_set, batch_size=8, shuffle=True)

    train()

    test()
