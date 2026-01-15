from rdkit import Chem
from rdkit.Chem import Draw

smis = [
    'CC(=O)CC(C1=CC=CC=C1)C1=C(O)C2=C(OC1=O)C=CC=C2',
    'CC(=O)CC(C1=CC=C(C=C1)[N+]([O-])=O)C1=C(O)C2=CC=CC=C2OC1=O',
    'CCC(C1=CC=CC=C1)C1=C(O)C2=C(OC1=O)C=CC=C2'
]
mols = []
for smi in smis:
    m = Chem.MolFromSmiles(smi)
    mols.append(m)

img = Draw.MolsToGridImage(
    mols,
    molsPerRow=3,
    subImgSize=(300, 300),
    legends=['' for x in mols]
)
img.save('C:\\Users\\Kiana\\Desktop\\mol.jpg')