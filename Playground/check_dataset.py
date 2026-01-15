import pandas as pd
import numpy as np
from dgllife.data import Tox21
from dgllife.utils import SMILESToBigraph, CanonicalAtomFeaturizer

smiles_to_g = SMILESToBigraph(node_featurizer=CanonicalAtomFeaturizer())
dataset = Tox21(smiles_to_g)

dataset = np.array(dataset)
print(dataset.shape)
print(dataset[0])
print(dataset[1])
