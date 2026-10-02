"""PDF parsing/rendering and stable source coordinates owned by PatentAgent."""

from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw


def parse_pages(spec: str | None, page_count: int) -> list[int]:
    if not spec:
        return []
    pages = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            raise ValueError("빈 페이지 범위입니다.")
        bounds = part.split("-")
        if len(bounds) > 2 or not all(b.isdigit() for b in bounds):
            raise ValueError(f"잘못된 페이지 범위: {part}")
        start, end = int(bounds[0]), int(bounds[-1])
        if not 1 <= start <= end <= page_count:
            raise ValueError(f"페이지 범위는 1~{page_count} 안이어야 합니다: {part}")
        pages.update(range(start, end + 1))
    return sorted(pages)


class PatentDocument:
    def __init__(self, path: Path, work_dir: Path, dpi: int = 150):
        self.doc = pymupdf.open(path)
        if self.doc.needs_pass:
            self.doc.close()
            raise ValueError("암호화된 PDF는 먼저 잠금을 해제하세요.")
        self.work_dir = work_dir
        self.dpi = dpi
        (work_dir / "pages").mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.doc.close()

    @property
    def page_count(self):
        return len(self.doc)

    def text(self, page: int):
        return self.doc[page - 1].get_text(sort=True)

    def render(self, page: int) -> Path:
        path = self.work_dir / "pages" / f"page_{page}.png"
        if not path.exists():
            self.doc[page - 1].get_pixmap(dpi=self.dpi, alpha=False).save(path)
        return path

    def highlight(self, page: int, bbox, structure_id: str) -> Path:
        path = self.work_dir / "regions" / f"{structure_id}_highlight.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(self.render(page)) as image:
            image = image.convert("RGB")
            ImageDraw.Draw(image).rectangle(bbox, outline="red", width=4)
            image.save(path)
        return path
