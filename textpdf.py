"""Write simple text-only PDFs that pypdf can extract by page.

The sample documents and the tests share this writer so a citation page is a
real PDF page, not a guessed line number.
"""

from pathlib import Path


def _escape(text):
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def _wrap(text, width):
    words = str(text).split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    lines.append(current)
    return lines


def write_text_pdf(path, pages, line_width=90):
    """Create a multi-page PDF.

    pages is a list of pages, and each page is a list of strings. Each string
    is wrapped but kept as one sentence for extraction.
    """
    if not pages:
        raise ValueError("A PDF needs at least one page.")

    page_lines = [
        [wrapped for paragraph in page for wrapped in _wrap(paragraph, line_width)]
        for page in pages
    ]

    page_ids = []
    content_ids = []
    next_id = 4
    for _page in page_lines:
        page_ids.append(next_id)
        content_ids.append(next_id + 1)
        next_id += 2

    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[2] = (
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_lines)} >>".encode("ascii")
    )

    for lines, page_id, content_id in zip(page_lines, page_ids, content_ids):
        commands = ["BT", "/F1 11 Tf", "54 740 Td", "14 TL"]
        for index, line in enumerate(lines):
            if index:
                commands.append("T*")
            commands.append(f"({_escape(line)}) Tj")
        commands.append("ET")
        stream = "\n".join(commands).encode("latin-1")
        objects[content_id] = (
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 3 0 R >> >> >>"
        ).encode("ascii")

    output = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for object_id in range(1, next_id):
        offsets[object_id] = len(output)
        output.extend(f"{object_id} 0 obj\n".encode("ascii"))
        output.extend(objects[object_id])
        output.extend(b"\nendobj\n")

    xref_at = len(output)
    output.extend(f"xref\n0 {next_id}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for object_id in range(1, next_id):
        output.extend(f"{offsets[object_id]:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer << /Size {next_id} /Root 1 0 R >>\n"
            f"startxref\n{xref_at}\n%%EOF\n"
        ).encode("ascii")
    )
    Path(path).write_bytes(output)
