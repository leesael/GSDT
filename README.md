# GSDT

This project is a PyTorch implementation of Gaussian Soft Decision Trees for Interpretable Feature-Based Classification,
published as a conference proceeding at PAKDD 2021. This paper proposes Gaussian Soft Decision Trees (GSDT), a novel
tree-based classifier with multi-branched structures, Gaussian mixtured-based decisions, and a hinge loss with path regularization.

## Prerequisites

- Python 3.6+
- [PyTorch](https://pytorch.org/) 1.4.0
- [NumPy](https://numpy.org)
- [scikit-learn](https://scikit-learn.org/stable/)
- [pandas](https://pandas.pydata.org/)

## Datasets

The paper use six datasets for feature-based classification. The raw datasets are included in `data/raw` and also available at
the follwing websites:
- brain-tumor: https://www.kaggle.com/pranavraikokte/braintumorfeaturesextracted 
- breast-cancer: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer
- breast-cancer-wisconsin: https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)
- diabetes: https://www.kaggle.com/uciml/pima-indians-diabetes-database
- heart-disease: https://archive.ics.uci.edu/ml/datasets/Heart+Disease
- hepatitis: https://archive.ics.uci.edu/ml/datasets/Hepatitis

## Usage

You can reproduce the main experimental results of the paper by running the following command in the `bin` directory:
```
bash main.sh
```

It first preprocesses the datasets and saves the results at `data/preprocessed`. Then, it runs a training script for
each dataset using multiple GPUs at the same time. Each experiment is run eight times, and the average and standard
deviation of accuracy are reported. You may need to change `main.sh`, as it currently uses 4 GPUs (from 0 to 3) for 
parallel experiments with different random seeds. You can also change other hyperparameters by modifying the script.

For instance, you can run the following command in `src` to change the tree depth and the number of children at each
branch as command line arguments. The `--device` option is needed to use a GPU environment.
```
python main.py --device 0 --data brain-tumor --depth 8 --branch 2 
```

## Reference

Please cite our paper if you use the implementation. 

```
@inproceedings{YooS21,
  author    = {Jaemin Yoo and Lee Sael},
  title     = {Gaussian Soft Decision Trees for Interpretable Feature-Based Classification},
  booktitle = {Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD)},
  year      = {2021}
}
```

