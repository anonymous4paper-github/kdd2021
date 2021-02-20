### The code for KDD2021 Submision "A Self-Supervised Disentangled Network for Aspect-Aware Heterogeneous Graph Embedding"

## Requirements
Our proposed method relies on
- Python 3.7
- PyTorch 1.1.0
- DGL 0.4.1
- networkx 2.2
See `requirements.txt` for more details.

## Usage
1. Install the requirements in `requirements.txt`.
2. Data preprocessing: run `python preprocess.py`. After this, the preprocessed data will be stored at `$data_dir`.
3. Run the demo code: run `sh run.sh`.
