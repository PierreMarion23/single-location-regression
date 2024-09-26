# Attention layers provably solve single-location regression

## Environment

### With conda

```
conda env create -f environment.yml
```
Then activate the environment and install the non-conda installable packages:
```
pip install jax==0.4.31 quadax==0.2.2
```

### With pip

Install Python 3.9.19 and pip 24.0, then

```
pip3 install -r requirements.txt
```

## Reproducing the experiments of the paper

For the experiments of Section 2 (linear probing in BERT), run the notebook ```linear_probing.ipynb```.

For the experiments of Section 5 (GD for single-attention regression), run

```
python main.py
```

The code takes in total of the order of one hour to run on a standard laptop CPU.