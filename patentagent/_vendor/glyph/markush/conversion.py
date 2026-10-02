# Derived from EdisonScientific/glyph, Apache-2.0.
# Pinned upstream: 0bf782f863d26b041ace157668928ef07c38b972. Modified for internal inference only.
# See LICENSE.txt and SOURCES.json for provenance and modifications.
from __future__ import annotations
import re
from rdkit import Chem

CX_SECTION_STARTS = (
    "atomProp:",
    "SgD:",
    "SgH:",
    "SgN:",
    "ctu:",
    "wU:",
    "wD:",
    "Sg:",
    "LN:",
    "m:",
    "c:",
    "t:",
    "f:",
    "w:",
    "S:",
    "r:",
    "@@:",
    "@:",
    "$",
    "(",
)

def split_cxsmiles(cxsmiles: str) -> tuple[str, str]:
    """Split a standard or optimized CXSMILES into ``(core, extension)``."""

    value = cxsmiles.strip()
    if "|" not in value:
        return value, ""
    core, rest = value.split("|", 1)
    if "|" in rest:
        rest = rest.rsplit("|", 1)[0]
    return core.strip(), rest.strip()

def split_extension_sections(extension: str) -> list[str]:
    """Split CXSMILES extension sections without splitting Sg index lists."""

    if not extension:
        return []

    sections: list[str] = []
    current: list[str] = []
    for i, char in enumerate(extension):
        if char == ",":
            j = i + 1
            while j < len(extension) and extension[j] == ",":
                j += 1
            rest = extension[j:]
            if not rest or rest.startswith(CX_SECTION_STARTS):
                sections.append("".join(current))
                current = []
            else:
                current.append(char)
        else:
            current.append(char)
    if current:
        sections.append("".join(current))
    return [section for section in sections if section]

def opt_to_standard_cxsmiles(cxsmiles_opt: str) -> str:
    """Convert optimized ``<r>LABEL</r>`` CXSMILES to ``|$...$|`` form.

    The conversion preserves the input atom order.  Temporary atom-map numbers
    are used only to discover which parsed atom each ``<r>`` tag created, so
    labels are written into the standard ``$`` section by atom index instead of
    by detached occurrence order.
    """

    core, extension = split_cxsmiles(cxsmiles_opt)
    if any(s.startswith("$") for s in split_extension_sections(extension)):
        if _RGROUP_TAG_RE.search(core) or "[Ar]" in core:
            raise ValueError("Mixed standard and optimized atom labels")
        return cxsmiles_opt.strip()
    marker_base = 9000
    marker_to_label: dict[int, str] = {}

    def make_marker(label: str) -> str:
        marker = marker_base + len(marker_to_label)
        marker_to_label[marker] = label.strip()
        return f"[*:{marker}]"

    def mark_rgroup(match: re.Match[str]) -> str:
        return make_marker(match.group(1))

    marked_core = _RGROUP_TAG_RE.sub(mark_rgroup, core)
    marked_core = re.sub(r"\[Ar\]", lambda _match: make_marker("Ar"), marked_core)
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.strictCXSMILES = False
    mol = Chem.MolFromSmiles(marked_core, params)
    if mol is None:
        raise ValueError(f"SMILES core does not parse: {cxsmiles_opt}")

    labels = [""] * mol.GetNumAtoms()
    for atom in mol.GetAtoms():
        marker = atom.GetAtomMapNum()
        if marker in marker_to_label:
            labels[atom.GetIdx()] = marker_to_label[marker]

    standard_core = _RGROUP_TAG_RE.sub("*", core)
    standard_core = re.sub(r"\[Ar\]", "*", standard_core)
    sections = [
        section for section in split_extension_sections(extension) if not section.startswith("$")
    ]
    if any(labels):
        sections.insert(0, "$" + ";".join(labels) + "$")
    if sections:
        return f"{standard_core} |{','.join(sections)}|"
    return standard_core

_RGROUP_TAG_RE = re.compile(r"<r>([^<]+)</r>")
