
# PICE-GNN for Cascading Failure Prediction

This repository contains the implementation of the paper titled
"Physics-Informed Co-Embedding Graph Neural Network for Cascading Failure Prediction in Power Systems".

The proposed PICE-GNN framework is designed for fast cascading failure prediction in power systems. It jointly models bus-level and branch-level representations through a co-embedding graph neural network, and incorporates power-system physical knowledge through admittance-guided structural encoding and PTDF-guided node-edge feature fusion.

## Data

The simulated datasets used in this study include the IEEE 39-bus system, IEEE 118-bus system, and a practical 341-bus power system. The cascading failure samples were generated using an AC cascading failure simulation model (AC-CFM), which considers AC power flow, protection actions, island handling, and post-cascade stable states.

Each sample contains:

- bus features;
- branch features;
- original graph adjacency information;
- line graph adjacency information;
- node-edge incidence information;
- binary bus labels;
- binary branch labels.

The bus and branch labels indicate the final post-cascade operating states of power system components. A label of 1 denotes normal operation, while a label of 0 denotes failure or out-of-service status after the cascading process.

Due to data confidentiality and system security considerations, the datasets used in the paper are not publicly released in this repository. Users may generate their own cascading failure samples using an AC cascading failure simulation tool and convert them into the required PyTorch Geometric data format.

## Code Structure

The main training and testing procedures are implemented in `main.py`.

The repository contains the following main components:

| File / Folder      | Description                                           |
| ------------------ | ----------------------------------------------------- |
| `main.py`        | Main script for training and testing                  |
| `model.py`       | Implementation of PICE-GNN and baseline GNN models    |
| `data.py`        | Dataset loading functions                             |
| `revise_data.py` | Data preprocessing and index adjustment               |
| `utils.py`       | Evaluation metrics and utility functions              |
| `f_loss.py`         | Focal Loss implementation for imbalanced classification |
| `Early_stopping.py` | Early stopping and checkpoint saving utility          |
| `datasets/`         | Local dataset folder, not included in this repository |
## Run

Use the following command to train and test the model:

```bash
python main.py --dataset case39 --model pice_gnn --num_layers 4 --hidden 16 --lr 0.001 --batch_size 256
```
## License and Data Availability

The source code in this repository is released under the MIT License.

Due to data confidentiality, power system security considerations, and file size limitations, the full datasets used in the paper are not publicly uploaded to this repository. Researchers who are interested in the datasets may contact the corresponding author for reasonable academic use.

