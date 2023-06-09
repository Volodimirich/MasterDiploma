import io
import pickle

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import os.path as osp
from tqdm import tqdm
import wandb
from yaml import safe_load

import os
from pathlib import Path
import glob
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Dataset, download_url
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import glob
import networkx as nx

SIZE = 256
NEURAL_TYPE = 'linear'
device = torch.device('cuda:0')


def create_layers(img_shape, num_blocks=3, hidden=16, out=10, tp='linear', isoclines=16, div_val=4):
    layers = nn.ModuleList()

    split_coefs = [2 ** i for i in range(num_blocks)]
    for coef in split_coefs:
        # Add simular pieces, several times
        for _ in range(coef ** 2):
            input_layers = (img_shape ** 2) // (isoclines * coef ** 2)
            layers.append(LinearAutoEncoder(input_layers, hidden, out, div_val))
            if tp == 'linear':
                break
    return layers


class LinearAutoEncoder(torch.nn.Module):
    def __init__(self, features: int, hidden: int, out_features: int, div_val=4):
        super().__init__()
        self.layers = nn.Sequential()
        inp = features
        while inp // div_val > hidden:
            self.layers.append(nn.Linear(inp, inp // div_val))
            self.layers.append(nn.ReLU())
            inp = inp // div_val

        self.layers.append(nn.Linear(inp, hidden))
        self.layers.append(nn.ReLU())

        while inp * div_val < out_features:
            self.layers.append(nn.Linear(inp, inp * div_val))
            self.layers.append(nn.ReLU())
            inp *= div_val


    def forward(self, x):
        x = self.layers(x)
        return x


def get_long_embedding(data, layers, iso_amount=16, batch_size=2):
    node_embeddings = []
    for it_val, value in enumerate(data):
        value = value.reshape(batch_size, iso_amount, -1)
        for i in range(iso_amount):
            node_embeddings.append(layers[it_val](value[:, i, :]))
    return torch.stack(node_embeddings, dim=1)


def getPositionEncoding(seq_len, d, n=10000):
    P = torch.zeros((seq_len, d))
    for k in range(seq_len):
        for i in torch.arange(int(d / 2)):
            denominator = torch.pow(n, 2 * i / d)
            P[k, 2 * i] = torch.sin(k / denominator)
            P[k, 2 * i + 1] = torch.cos(k / denominator)
    return P


def get_embedding(data, layers, iso_amount=16, batch_size=2):
    node_embeddings = []
    for it_val, value in enumerate(data):
        value = value.reshape(batch_size, iso_amount, -1)
        if it_val == 0:
            layer_id = 0
        elif 1 <= it_val <= 4:
            layer_id = 1
        elif 5 <= it_val <= 21:
            layer_id = 2
        for i in range(iso_amount):
            node_embeddings.append(layers[layer_id](value[:, i, :]))

    return torch.stack(node_embeddings, dim=1)


def add_positional_encoding(embedding, iso_amount=16, emb_len=10):
    b, h, c = embedding.shape
    big_embed = torch.zeros(b, iso_amount, c)

    four_split_embed_func = getPositionEncoding(4, c)
    medium_embed = torch.cat([four_split_embed_func[i].unsqueeze(0).repeat(iso_amount, 1).unsqueeze(0).repeat(b, 1, 1)
                              for i in range(4)], dim=1)

    sixt_split_embed_func = getPositionEncoding(16, c)

    small_embed = torch.cat([sixt_split_embed_func[i].unsqueeze(0).repeat(iso_amount, 1).unsqueeze(0).repeat(b, 1, 1)
                             for i in range(16)], dim=1)
    positional_embed = torch.cat([big_embed, medium_embed, small_embed], dim=1).to(device)
    return embedding + positional_embed


class GNN(torch.nn.Module):

    def __init__(self, hidden_encoder, hidden_GCN, num_classes, edge_weight, size=256,
                 encoder_out=10, emb_type='linear', div_val=2, is_pos_embed='full', depth=1,
                 num_blocks=3, iso_amount=16, b_size=60, is_edges_trainable=True):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.is_pos_embed = is_pos_embed.lower()
        self.depth = depth
        self.enc_out = encoder_out
        self.emb_type = emb_type
        self.b_size = b_size
        self.iso_amount = iso_amount

        self.edge_weight_amp = nn.Parameter(torch.tensor([float(ed) for ed in edge_weight])) if is_edges_trainable else None
        self.edge_weight_ph = nn.Parameter(torch.tensor([float(ed) for ed in edge_weight])) if is_edges_trainable else None

        if depth not in [1, 2]:
            raise ValueError

        self.layers_amp = create_layers(size, num_blocks=num_blocks, hidden=hidden_encoder, tp=emb_type,
                                        out=encoder_out, div_val=div_val, isoclines=self.iso_amount)
        self.layers_ph = create_layers(size, num_blocks=num_blocks, hidden=hidden_encoder, tp=emb_type,
                                       out=encoder_out, div_val=div_val, isoclines=self.iso_amount)
        self.pos_embed = nn.Parameter(torch.zeros(b_size, 21 * self.iso_amount, hidden_encoder))
        # self.conv1 = SAGEConv(hidden_GCN, hidden_GCN)
        self.conv1_amp = GCNConv(hidden_GCN, hidden_GCN)
        self.conv1_ph = GCNConv(hidden_GCN, hidden_GCN)

        if depth == 2:
            self.conv2_amp = GCNConv(hidden_GCN, hidden_GCN)
            self.conv2_ph = GCNConv(hidden_GCN, hidden_GCN)

        # self.basic_conf_amp = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)
        # self.basic_conf_ph = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)

        self.conv1d = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)
        self.lin = nn.Linear(hidden_GCN, num_classes)
        self.soft = nn.Softmax(dim=1)

    def forward(self, amp, ph, edge_index, batch, batch_size):
        if self.emb_type == 'linear':
            amp = get_embedding(amp, self.layers_amp, iso_amount=self.iso_amount, batch_size=batch_size)  # current linear
            ph = get_embedding(ph, self.layers_ph, iso_amount=self.iso_amount, batch_size=batch_size)  # current linear
        elif self.emb_type == 'long':
            amp = get_long_embedding(amp, self.layers_amp, iso_amount=self.iso_amount, batch_size=batch_size)
            ph = get_long_embedding(ph, self.layers_ph, iso_amount=self.iso_amount, batch_size=batch_size)

        if self.is_pos_embed == 'only_train':
            amp = add_positional_encoding(amp, emb_len=self.enc_out)
            ph = add_positional_encoding(ph, emb_len=self.enc_out)
        elif self.is_pos_embed == 'only_param':
            amp += self.pos_embed
            ph += self.pos_embed
        elif self.is_pos_embed == 'full':
            amp = add_positional_encoding(amp, emb_len=self.enc_out) + self.pos_embed
            ph = add_positional_encoding(ph, emb_len=self.enc_out) + self.pos_embed
        # Maybe add this in get_embedding?
        amp = amp.view(batch_size * 21 * self.iso_amount, -1)
        ph = ph.view(batch_size * 21 * self.iso_amount, -1)

        if self.edge_weight_amp is not None:
            edges_amp = self.edge_weight_amp.repeat(self.b_size).sigmoid()
            edges_ph = self.edge_weight_ph.repeat(self.b_size).sigmoid()
        else:
            edges_amp, edges_ph = None, None
        amp = self.conv1_amp(amp, edge_index).relu()
        ph = self.conv1_ph(ph, edge_index).relu()
        if self.depth == 2:
            amp = self.conv2_amp(amp, edge_index).relu()
            ph = self.conv2_ph(ph, edge_index).relu()

        amp = global_mean_pool(amp, batch)  # [batch_size, hidden_channels]
        ph = global_mean_pool(ph, batch)


        # x = torch.cat([self.basic_conf_amp(amp.unsqueeze(1)), self.basic_conf_ph(ph.unsqueeze(1))], 1)
        x = torch.cat([amp.unsqueeze(1), ph.unsqueeze(1)], 1)
        x = self.conv1d(x).squeeze()
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)
        return self.soft(x)


class MyOwnDataset(Dataset):
    def __init__(self, root, files_list, is_train, size=256, allow_loops=False, transform=None, pre_transform=None,
                 pre_filter=None, iso_amount=16, split_amount=21):
        self.data = files_list
        self.allow_loops = allow_loops
        self.is_train = is_train
        root_path = Path(root)
        self._processed_dir = str(Path(root).parent) + '/processed_' + str(Path(root).name)
        # TODO fix this shit
        self.iso_amount = iso_amount
        self.split_amount = split_amount
        # self.gl_count = 17407 if self.is_train else 7461
        self.gl_count = 5270 if self.is_train else 586

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return []

    @property
    def num_nodes(self):
        return self.iso_amount * self.split_amount

    @property
    def processed_file_names(self):
        return [f'full_data_{idx}_is_train_{self.is_train}_loops={self.allow_loops}_isoc={self.iso_amount}' \
                f'_split={self.split_amount}.pt' for idx in range(self.gl_count)]

    def _create_cco_matrix(self):
        vert_amount = int(self.iso_amount * self.split_amount)

        adj_matrix = np.zeros((vert_amount, vert_amount))
        for i in range(vert_amount - 1):
            if self.allow_loops:
                adj_matrix[i][i] = nn.Parameter(1)
                # adj_matrix[i][i] = nn.Parameter(torch.tensor(1.0))
            if i % self.iso_amount != 1:
                # adj_matrix[i][i + 1] = nn.Parameter(torch.tensor(1.0))
                adj_matrix[i][i + 1] = 1
            for j in range(self.split_amount):
                if self.iso_amount * j + i < vert_amount:
                    # adj_matrix[i][self.iso_amount * j + i] = nn.Parameter(torch.tensor(1.0))
                    adj_matrix[i][self.iso_amount * j + i] = 1
                else:
                    break

        source_nodes = []
        target_nodes = []
        edge_list = []
        for iy, ix in np.ndindex(adj_matrix.shape):
            # small changes
            if adj_matrix[iy, ix] != 0:
                source_nodes.append(ix)
                target_nodes.append(iy)
                edge_list.append(1.0)
        return source_nodes, target_nodes, edge_list

    def process(self):
        idx = 0
        source_vert, target_vert, edge_list = self._create_cco_matrix()
        edge_idx = torch.tensor([source_vert, target_vert])
        small_data = self.data
        train_data, test_data = train_test_split(small_data, test_size=0.1, random_state=42)
        print(len(train_data), len(test_data))
        data = train_data if self.is_train else test_data
        for file in data:
            amp, phase, target = np.load(file, allow_pickle=True)
            try:
                amp = [torch.tensor(item).float() for item in amp]
                ph = [torch.tensor(item).float() for item in phase]

                data = Data(x=amp,
                            edge_index=edge_idx,
                            edge_attr=edge_list,
                            y=torch.tensor([target]),
                            ph=ph)
                torch.save(data, osp.join(self.processed_dir,
                                          f'full_data_{idx}_is_train_{self.is_train}_loops={self.allow_loops}'
                                          f'_isoc={self.iso_amount}_split={self.split_amount}.pt'))

                idx += 1
            except:
                continue
        self.gl_count = idx

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(
            osp.join(self.processed_dir,
                     f'full_data_{idx}_is_train_{self.is_train}_loops={self.allow_loops}'
                     f'_isoc={self.iso_amount}_split={self.split_amount}.pt'))

        return data


class GraphInformation:
    def __init__(self, graph_struct):
        self.max_val = 0
        self.graph_struct = graph_struct.detach().cpu().numpy()
        self.adj_matrix = self._get_adj()
        self.G = nx.from_numpy_array(self.adj_matrix)

    def _get_adj(self):
        _, edge_amount = self.graph_struct.shape
        self.max_val = np.amax(self.graph_struct)
        adj_matrix = np.zeros((self.max_val + 1, self.max_val + 1))
        for i in range(edge_amount):
            x, y = self.graph_struct[..., i]
            adj_matrix[x][y] = 1
            adj_matrix[y][x] = 1
        return adj_matrix

    def get_adj_view(self, connectivity_array):
        _, edge_amount = self.graph_struct.shape
        image = np.zeros((self.max_val + 1, self.max_val + 1))
        for i in range(edge_amount):
            x, y = self.graph_struct[..., i]
            image[x][y] = connectivity_array[i]
        return image

    def get_matrix_view(self, connectivity_array):
        colors = []
        for i, j in self.G.edges():
            colors.append(connectivity_array[i][j].item())

        im_io = io.BytesIO()
        fig, ax = plt.subplots()

        nx.draw_circular(self.G, ax=ax, edge_color=colors)
        fig.savefig(im_io, format='png')
        return Image.open(im_io)


def get_config(path):
    with open(path, 'r') as stream:
        config = safe_load(stream)
    return config


if __name__ == '__main__':
    config = get_config('conf.yml')
    train_params = config['train_params']
    model_params = config['model']
    files_params = config['files_params']

    device = torch.device(f'cuda:{train_params["device_num"]}')
    batch_size = config['train_params']['batch_size']

    save_root = files_params['save_root']
    files = glob.glob(files_params['path_to_files'], recursive=True)

    train_dataset = MyOwnDataset(save_root, files, is_train=True, iso_amount=model_params['iso_amount'],
                                 split_amount=21 if model_params['num_blocks'] == 3 else 85)
    test_dataset = MyOwnDataset(save_root, files, is_train=False, iso_amount=model_params['iso_amount'],
                                split_amount=21 if model_params['num_blocks'] == 3 else 85)
    graph = train_dataset[0].edge_attr

    model = GNN(num_classes=2, hidden_encoder=model_params['hidden_encoder_embed'], edge_weight=graph,
                hidden_GCN=model_params['hidden_GCN_embed'], encoder_out=model_params['encoder_out'],
                emb_type=model_params['emb_type'], div_val=model_params['div_val'],
                is_pos_embed=model_params['pos_embed'], depth=model_params['depth'],
                num_blocks=model_params['num_blocks'], iso_amount=model_params['iso_amount'],
                b_size=train_params['batch_size'], is_edges_trainable=model_params['is_edges_trainable']).to(device)
    # print(model.parameters)
    # for name, param in model.named_parameters():
    #     print(name, param.shape, param.numel())
    print(sum(p.numel() for p in model.parameters()), 'sum')
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    wandb_config = dict(
        batch_size=batch_size,
        emb_type=model_params['emb_type'],
        architecture=model_params['model_type'],
        number_of_layers=model_params['depth'],
        is_pos_embed=model_params['pos_embed'],
        hidden_GCN=model_params['hidden_GCN_embed'],
        encoder_out=model_params['encoder_out'],
        div_val=model_params['div_val'],
        hidden_encoder=model_params['hidden_encoder_embed'],
        num_blocks=model_params['num_blocks'],
        iso_amount=model_params['iso_amount'],
        is_edges_trainable=model_params['is_edges_trainable']
    )

    wandb.init(
        project="GNN",
        notes="Deeper network, with pos embedding with trainable",
        config=wandb_config,
        mode=train_params['wandb_mode']
    )
    weights_stem = f'{config["exp_name"]}'
    wandb.run.name = weights_stem
    print(train_dataset[0].edge_index.shape)
    GraphLogger = GraphInformation(train_dataset[0].edge_index) if model_params['is_edges_trainable'] else None
    adj_hist, adj_counter = [], 0

    def train():
        model.train()
        for itt, data in enumerate(train_loader):  # Iterate in batches over the training dataset.
            optimizer.zero_grad()  # Clear gradients.

            data = data.to(device)
            out = model(data.x, data.ph, data.edge_index, data.batch, batch_size)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            if itt % 100 == 0:
                wandb.log({"train_cross_entropy_loss": loss.item()})

            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients
        if GraphLogger and False:
            adj_view = GraphLogger.get_adj_view(model.edge_weight.detach())
            adj_hist.append(adj_view)
            img = GraphLogger.get_matrix_view(adj_view)
            wandb.log({'Train adj': wandb.Image(adj_view, caption='Train Adj view')})
            img.close()
            wandb.log(
                {'Train view': wandb.Image(GraphLogger.get_matrix_view(img), caption='Train graph view')})

    def test(loader):
        model.eval()
        correct = 0

        for itt, data in enumerate(loader):  # %Iterate in batches over the training/test dataset.
            data = data.to(device)
            out = model(data.x, data.ph, data.edge_index, data.batch, batch_size)  # Perform a single forward pass.
            data = data.to(device)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        if GraphLogger and False:
            adj_view = GraphLogger.get_adj_view(model.edge_weight.detach())
            #
            wandb.log(
                {'Test adj': wandb.Image(adj_view, caption='Test Adj view')})
            img = GraphLogger.get_matrix_view(adj_view)
            wandb.log(
                {'Test view': wandb.Image(img, caption='Test graph view')})
            img.close()
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.


    best_train, cur_epoch, best_val = -1, -1, -1
    for epoch in tqdm(range(0, 100)):
        train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        scheduler.step(test_acc)
        if train_acc > best_train:
            dict_vals_amp, dict_vals_ph = {}, {}
            for pos, (x, y) in enumerate(zip(train_dataset[0].edge_index[0][:4009], train_dataset[0].edge_index[1][:4009])):
                dict_vals_amp[(x.item(), y.item())] = model.edge_weight_amp[pos].item()
                dict_vals_ph[(x.item(), y.item())] = model.edge_weight_ph[pos].item()
            file_name = f"dicts/dct_double_isoc_amp_hid={model_params['hidden_GCN_embed']}_out={model_params['encoder_out']}_enc={model_params['hidden_encoder_embed']}.pkl"
            with open(file_name, 'wb') as f:
                dict_vals = dict(sorted(dict_vals_amp.items(), key=lambda item: item[1], reverse=True))
                pickle.dump(dict_vals, f)
            file_name = f"dicts/dct_double_isoc_ph_hid={model_params['hidden_GCN_embed']}_out={model_params['encoder_out']}_enc={model_params['hidden_encoder_embed']}.pkl"
            with open(file_name, 'wb') as f:
                dict_vals = dict(sorted(dict_vals_ph.items(), key=lambda item: item[1], reverse=True))
                pickle.dump(dict_vals, f)
            best_train = train_acc 
            best_val = test_acc 
            cur_epoch = epoch
        wandb.log({'best train accuracy': best_train})
        wandb.log({'best test accuracy': best_val})

        wandb.log({'current train accuracy': train_acc})
        wandb.log({'current test accuracy': test_acc})

        if epoch % 10:
            with open(f'pickles/adj_train{adj_counter}.pkl', 'wb') as f:
                pickle.dump(adj_hist, f)
                adj_counter += 1

        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


    print(f'Best test accuracy - {best_val} on epoch {cur_epoch} with train accuracy -> {best_train}')