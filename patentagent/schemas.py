"""Validated extraction contracts, independent of an engine's output files."""

from pydantic import BaseModel, ConfigDict, Field


class ExtractionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Identity(ExtractionModel):
    compound_id: str | None = None
    evidence_text: str | None = None


class AssayObservation(ExtractionModel):
    compound_id: str | None = None
    assay_name: str
    value_raw: str
    unit: str | None = None
    target: str | None = None
    cell_line: str | None = None
    organism: str | None = None
    condition: str | None = None
    evidence_text: str


class AssayPage(ExtractionModel):
    observations: list[AssayObservation] = Field(default_factory=list)


class PageClassification(ExtractionModel):
    has_structures: bool
    has_activity: bool


class StructureRegion(ExtractionModel):
    structure_id: str
    page: int
    bbox: tuple[int, int, int, int]
    image: str
    page_image: str
    compound_id: str | None = None
    identity_evidence: str | None = None
    smiles: str | None = None
    ocsr_raw: str | None = None
    ocsr_model: str | None = None
    markush_model: str | None = None
    markush_raw: str | None = None
    markush_stable_raw: str | None = None
    markush_conversion_status: str = "not_run"
    model_provenance: dict = Field(default_factory=dict)
    cxsmiles: str | None = None
    cxsmiles_opt: str | None = None
    substituents: dict = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class RunOptions(ExtractionModel):
    structure_pages: str | None = None
    assay_pages: str | None = None
    assay_names: list[str] = Field(default_factory=list)
    auto_pages: bool = True
    markush: bool = True
