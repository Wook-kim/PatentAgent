import pytest

from patentagent import postprocess as pp

pytest.importorskip("rdkit")


def _agree(smiles, cxsmiles, **extra):
    row = {"smiles_ocsr": smiles, "cxsmiles_markush": cxsmiles, **extra}
    return pp.enrich_with_rdkit([row])[0]


@pytest.mark.parametrize("smiles,cxsmiles", [
    ("CCC", "*CC |$_AP;;$|"),            # 물결선 부착점 vs 메틸
    ("NCO", "N*O |$;CH2?2;$|"),          # CH2 상위원자 vs 탄소 ("?" 꼬리 무시)
    ("CC(C)O", "C*(C)O |$;CH;;$|"),
])
def test_label_notation_is_normalized(smiles, cxsmiles):
    row = _agree(smiles, cxsmiles)
    assert row["agreement"] == "match_normalized"
    assert row["confidence"] == "high"


def test_variable_labels_stay_wildcards():
    assert _agree("*CC", "*CC |$R1;;$|")["agreement"] == "match"
    row = _agree("CCC", "*CC |$R1;;$|", compound_id_raw="1a")
    assert row["agreement"] == "mismatch"
    assert row["confidence"] == "low"


def test_isotope_and_map_wildcards_are_normalized():
    assert _agree("*CC", "[1*:2]CC")["agreement"] == "match_normalized"


def test_real_skeleton_difference_is_mismatch():
    assert _agree("CC=O", "CC=C")["agreement"] == "mismatch"
