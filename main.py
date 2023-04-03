from torch_geometric.data import Dataset, download_url
from torch_geometric.data import Data
import os.path as osp
import glob
import torch
import torch.nn as nn
import numpy as np
from yaml import safe_load
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv,SAGEConv
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
import wandb




SIZE = 256
NEURAL_TYPE = 'linear'
files = glob.glob('/raid/data/cats_dogs_dataset/preprocessed/*/*.npy', recursive=True)
device = torch.device('cuda:0')

def create_circular(h, w, max_numb=None, center=None, tolerance=1, min_val=0):
    if center is None:  # use the middle of the image
        center = (w / 2 - 0.5, h / 2 - 0.5)
    if max_numb is None:
        max_numb = max(h//2, w//2)

    dist_from_center = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            dist_from_center[i][j] = int(max(abs(i - center[0]), abs(j-center[1]))//tolerance + min_val)
    return dist_from_center

def split_tensor(size, split_coefs=None):
    if split_coefs is None:
        split_coefs = [1, 2, 4]
    result = []

    last_val = 0
    for coef in split_coefs:
        step = size // coef
        # add fft
        row_data = []
        for y in range(0, size, step):
            step_result = []
            for x in range(0, size, step):
                #temperory unused
                mask = create_circular(step, step, min_val=last_val)
                last_val = mask[-1][-1] + 1
                step_result.append(mask)
            row_data.append(np.concatenate(step_result, axis=1))
        result.append(np.concatenate(row_data, axis=0))
    return result


SIZE = 256
global_map = split_tensor(SIZE)

#
class ConvAutoEncoder(torch.nn.Module):
    def __init__(self, band_length, out_features: int, in_channels=1, hidden=16, kernel=2):
        super().__init__()
        self.encoder = nn.Conv2d(in_channels, hidden, kernel_size=kernel)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(hidden * band_length, out_features)

    def forward(self, X):
        return self.decoder(self.relu(self.encoder(X).flatten()))


class LinearAutoEncoder(torch.nn.Module):
    def __init__(self, features: int, hidden: int, out_features: int):
        super().__init__()
        self.encoder = nn.Linear(features, hidden)
        self.relu = nn.ReLU()

        self.decoder = nn.Linear(hidden, out_features)

    def forward(self, X):
        return self.decoder(self.relu(self.encoder(X)))


def create_layers(img_shape, split_coefs=None, hidden=32, out=10, tp='linear'):
    layers = nn.ModuleList()

    if split_coefs is None:
        split_coefs = [1, 2, 4]

    for coef in split_coefs:
        # Add simular pieces
        for _ in range(coef ** 2):
            input_layers = 4 if tp == 'linear' else 1
            for _ in range(img_shape // (2 * coef)):
                # Linear
                if tp == 'linear':
                    layers.append(LinearAutoEncoder(input_layers, hidden, out))
                    input_layers += 8
                elif tp == 'conv':
                    layers.append(ConvAutoEncoder(input_layers, out, hidden=hidden).to(device))
                    input_layers += 4

    return layers

#
# def get_embedding(data, layers, border=None, neural_type='linear'):
#     min_value = 0
#     node_embeddings = []
#     for data_map, value_map in zip(data, global_map):
#         value_map = value_map.astype(int)
#         max_value = value_map[-1][-1].astype(int)
#
#         for rad in range(min_value, max_value + 1):
#             values = data_map[value_map == rad]
#             if neural_type == 'linear':
#                 values = torch.from_numpy(values).float().to(device)
#                 node_embeddings.append(layers[rad](values))
#             elif neural_type == 'conv':
#                 values2d = torch.from_numpy(reshape_2d(values, border[rad])).float().to(device)
#                 values2d = values2d.unsqueeze(0)
#                 node_embeddings.append(layers[rad](values2d))
#         min_value = max_value + 1
#     return torch.stack(node_embeddings)


def get_embedding(data, layers, border=None, neural_type='linear'):
    result = []
    for batch in data:
        min_value = 0
        node_embeddings = []
        for data_map, value_map in zip(batch, global_map):
            value_map = value_map.astype(int)
            max_value = value_map[-1][-1].astype(int)

            for rad in range(min_value, max_value + 1):
                values = torch.from_numpy(data_map[value_map == rad]).float().to(device)
                if neural_type == 'linear':
                    node_embeddings.append(layers[rad](values))
                elif neural_type == 'conv':
                    values2d = torch.from_numpy(reshape_2d(values, border[rad])).float()
                    values2d = values2d.unsqueeze(0)
                    node_embeddings.append(layers[rad](values2d))
            min_value = max_value + 1
        result.append(torch.stack(node_embeddings))
    return torch.stack(result)


def reshape_2d(array, border_size):
    ptr_st, ptr_end = border_size, len(array) - border_size
    data1 = list(array[:ptr_st])
    data2 = list(array[ptr_end:])
    itt_leng = 4
    while ptr_st != ptr_end:
        if itt_leng > 2:
            data1.append(array[ptr_st])
        else:
            data2.append(array[ptr_st])
        ptr_st += 1
        itt_leng -= 1
        if itt_leng == 0:
            itt_leng = 4
    return np.stack([data1, data2])


def create_border_leng(img_shape, split_coefs=None):
    border_leng = []

    if split_coefs is None:
        split_coefs = [1, 2, 4]
    for coef in split_coefs:
        # Add simular pieces
        for _ in range(coef ** 2):
            inp = 2
            for _ in range(img_shape // (2 * coef)):
                border_leng.append(inp)
                inp += 2

    return border_leng

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes, size=256, out=10, emb_type='linear', model_type='GCN'):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.neural_type = emb_type
        #
        self.layers = create_layers(size, tp=emb_type, out=out)

        if emb_type == 'conv':
            self.border = create_border_leng(size)
        else:
            self.border = None

        if model_type == 'GCN':
            self.conv1 = GCNConv(out, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
        elif model_type == 'SAGE':
            self.conv1 = SAGEConv(out, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        self.lin = nn.Linear(hidden_channels, num_classes)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x, edge_index, batch):
        x = get_embedding(x, self.layers, self.border, neural_type=self.neural_type)  # current linear
        # x.view(batch_size * num_nodes, -1
        x = x.view(batch_size * 896, -1)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)
        return self.soft(x)

class MyOwnDataset(Dataset):
    def __init__(self, root, files_list, is_train, size=256, allow_loops=False, transform=None, pre_transform=None,
                 pre_filter=None):
        self.data = files_list
        self.allow_loops = allow_loops
        self.is_train = is_train
        # TODO fix this shit
        self.gl_count = 1750 if self.is_train else 750
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return []

    @property
    def num_nodes(self):
        return SIZE // 2 + SIZE // 4 * 4 + SIZE // 8 * 16

    @property
    def processed_file_names(self):
        return [f'data_{idx}_is_train_{self.is_train}_loops={self.allow_loops}.pt' for idx in range(self.gl_count)]

    def _create_cco_matrix(self):
        result = split_tensor(SIZE)
        vert_amount = int(result[2][-1][-1] + 1)
        adj_matrix = np.zeros((vert_amount, vert_amount)).astype(int)

        gl_map, gen_map, loc_map = result[0].astype(int), result[1].astype(int), result[2].astype(int)
        for gl, gen, loc in np.nditer([gl_map, gen_map, loc_map]):
            adj_matrix[gl, gen] = 1
            adj_matrix[gl, loc] = 1
            adj_matrix[gen, loc] = 1
            if self.allow_loops:
                adj_matrix[gl, gl] = 1
                adj_matrix[loc, loc] = 1
                adj_matrix[gen, gen] = 1

        source_nodes = []
        target_nodes = []
        edge_list = []
        for iy, ix in np.ndindex(adj_matrix.shape):
            if adj_matrix[iy, ix] == 1:
                source_nodes.append(ix)
                target_nodes.append(iy)

                # unweighted solution
                edge_list.append(1)

        return source_nodes, target_nodes, edge_list

    def process(self):
        idx = 0
        source_vert, target_vert, edge_list = self._create_cco_matrix()
        edge_idx = torch.tensor([source_vert, target_vert])

        # DEBUG ROW
        _, small_data = train_test_split(self.data, test_size=0.1, random_state=42)
        train_data, test_data = train_test_split(small_data, test_size=0.3, random_state=42)
        data = train_data if self.is_train else test_data

        for file in data:
            # Read data from `raw_path`.
            amp, phase, target = np.load(file, allow_pickle=True)

            # temporary only amp|
            data = Data(x=amp,
                        # edge_index=torch.tensor(edge_idx).clone().detach().float().requires_grad_(True),
                        edge_index=edge_idx,
                        edge_attrs=edge_list,
                        y=torch.tensor([target]))

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}_is_train_{self.is_train}_loops={self.allow_loops}.pt'))
            idx += 1
        self.gl_count = idx

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}_is_train_{self.is_train}_loops={self.allow_loops}.pt'))
        return data

def get_config(path):
    with open(path, 'r') as stream:
        config = safe_load(stream)
    return config


if __name__ == '__main__':
    config = get_config('conf.yml')
    allow_loops = config['model']['allow_loops']
    model_type = config['model']['model_type']
    emb_type = config['model']['emb_type']
    save_root = '/raid/data/cats_dogs_dataset/'
    files = glob.glob('/raid/data/cats_dogs_dataset/preprocessed/*/*.npy', recursive=True)
    train_dataset = MyOwnDataset(save_root, files, is_train=True, allow_loops=allow_loops)
    test_dataset = MyOwnDataset(save_root, files, is_train=False, allow_loops=allow_loops)

    model = GCN(num_classes=2, hidden_channels=10, model_type=model_type, emb_type=emb_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    print(sum(p.numel() for p in model.parameters()), 'sum')
    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



    wandb.init(project="GNN")
    weights_stem = f'GNN_{model_type}_Conv_allow_loops={allow_loops}_bsize=1'
    wandb.run.name = weights_stem


    #                 experiment.log_metric("train_dice_loss", batch_loss.item(),
    #                                       epoch=epoch_idx, step=step_counter[action])
    def train():
        model.train()
        itt = 0
        for itt, data in enumerate(train_loader):  # Iterate in batches over the training dataset.
            #         out = model(data.x, data.edge_index, data.batch)
            optimizer.zero_grad()  # Clear gradients.

            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            print(out, loss)
            if itt % 100 == 0:
                wandb.log({"train_cross_entropy_loss": loss.item()})
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.


    def test(loader):
        model.eval()
        correct = 0
        for itt, data in enumerate(loader):  # %ђIterate in batches over the training/test dataset.
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        #         correct += int((pred == data.y>.squeeze().unsqueeze(0).float()).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.


    best_train, cur_epoch, best_val = -1, -1, -1
    for epoch in range(1, 50):
        train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        best_train, cur_epoch, best_val = (train_acc, epoch, test_acc) if test_acc > best_val \
            else (best_train, cur_epoch, best_val)
        wandb.log({'best train accuracy': best_train})
        wandb.log({'best test accuracy': best_val})

        wandb.log({'current train accuracy': train_acc})
        wandb.log({'current test accuracy': test_acc})

        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    print(f'Best test accuracy - {best_val} on epoch {cur_epoch} with train accuracy - {best_train}')
