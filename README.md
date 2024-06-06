# SetToGraphPaper

This repository is a refactored version of the original [SetToGraphPaper](https://github.com/hadarser/SetToGraphPaper), focusing on training with CPU and utilizing only the Set2Graph (S2G) method. For more details, please refer to the `GNN_b_tag.pdf` included in this repository. This document was created to compare results with [this paper](https://arxiv.org/abs/2008.02831) and explore further questions.

## Data

Before running the code for the jets experiments, download the data using the following commands:

```bash
cd SetToGraphPaper
python download_jets_data.py
```

This script will download all the data from Zen

# SetToGraphPaper

This repository is a refactored version of the original [SetToGraphPaper](https://github.com/hadarser/SetToGraphPaper), focusing on training with CPU and utilizing only the Set2Graph (S2G) method. For more details, please refer to the `GNN_b_tag.pdf` included in this repository. This document was created to compare results with [this paper](https://arxiv.org/abs/2008.02831) and explore further questions.

## Data

Before running the code for the jets experiments, download the data using the following commands:

```bash
cd SetToGraphPaper
python download_jets_data.py
```

This script will download all the data from Zenodo links.

## Running the Tests

The `main_scripts` folder contains scripts that run different experiments. To run the main particle-physics jets experiment with our chosen hyper-parameters, use the following command:

```bash
python main_scripts/main_jets.py --method=lin2  # for S2G
```

If you wish to use the S2G+ method, run:

```bash
python main_scripts/main_jets.py --method=lin5  # for S2G+
```

## Additional Information

For more detailed information on the experimental setup and results, refer to the `GNN_b_tag.pdf` included in this repository. This document was prepared to compare results with those in [the referenced paper](https://arxiv.org/abs/2008.02831) and to explore further research questions.

## Note

This refactored version focuses solely on the S2G method and is optimized for CPU training. The original implementation and additional methods such as `siam` and `rnn` baselines are not included in this version.# S2G-GNN-for-b-tagging-in-LHC
