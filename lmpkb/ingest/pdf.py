from pathlib import Path

import fitz
from PIL.Image import Image, frombytes


def count_pages(path: Path) -> int:
    doc = fitz.open(str(path))
    n = len(doc)
    doc.close()
    return n


def load_pdf(path: Path) -> list[tuple[int, Image]]:
    doc = fitz.open(str(path))
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        image = frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append((page_num, image))

    doc.close()
    return pages
