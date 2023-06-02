# MasterDiploma
## Project structure:
``` bash
.
├── README.md 
├── circle.py - Rings architecture
├── conf.yml - Config file
├── dicts - directory with node importance dict 
├── full_isoclines.py - Two-tower architecture
├── isoclines.py - Isoclines architecture
├── notebooks - Notebooks with tests
│   ├── GNN_get_nodes.ipynb
│   ├── GNN_img.ipynb
│   ├── GNN_isoclines.ipynb
│   └── Positional embedding.ipynb
├── preprocessing_isoc.py - Isoclines (and Two-tower) data preprocessing script
├── preprocessing_rings.py - Rings preprocessings script
└── requirements.txt - 
```
## Conf.yml

model
- model_type - Type of model, **GCNConv** only avalible now
- hidden_encoder_embed - amount of min channels in encoder  
- hidden_GCN_embed - amount of GCN embed size
- encoder_out - Encoder-decoder result embed size
- allow_loops - Alowance of self loops of graph. Default - **True** 
- emb_type - Type of embedding strategy, **long** and **normal** avalible options
- div_val - Decoder step
- pos_embed - Positional embedding type, options: **full**, **only_param**, **only_train**, **none**
- depth - 2 - GNN depth
- num_blocks - Amount of blocks on preproc stage, **3** is recommended
- iso_amount - Amount of isoclines on preproc stage, **16** is recommended
- is_edges_trainable - Avalible parameters: **True** and **False**

train_params:
- batch_size - model batch size
- device_num - cuda device number
- wandb_mode - logging wandb mode ('online', 'offline', 'disabled')
files_params:
- save_root - Root for GNN data files saving 
- path_to_files - Path to preprocessed files

exp_name - Name of exp on wandb 
