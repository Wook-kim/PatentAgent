# Derived from EdisonScientific/glyph, Apache-2.0.
# Pinned upstream: 0bf782f863d26b041ace157668928ef07c38b972. Modified for internal inference only.
# See LICENSE.txt and SOURCES.json for provenance and modifications.
from rdkit import Chem, RDLogger

def canonical_smiles(s: str | None) -> str | None:
    """Return RDKit canonical SMILES, or None if unparseable / empty."""

    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)
