"""Text-only PDFs for tests. pypdf can extract each page."""


def pdf_escape(text):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(pages):
    """Each page is a list of lines."""
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    kids = []
    next_id = 4
    for lines in pages:
        commands = ["BT", "/F1 11 Tf", "72 740 Td", "16 TL"]
        for line in lines:
            # A trailing space lets the chunker's ". " split see the period.
            commands.append(f"({pdf_escape(line)} ) Tj")
            commands.append("T*")
        commands.append("ET")
        stream = "\n".join(commands).encode("latin-1")
        content_id = next_id
        page_id = next_id + 1
        next_id += 2
        objects[content_id] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1")
            + stream
            + b"\nendstream"
        )
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 3 0 R >> >> >>"
        ).encode("latin-1")
        kids.append(page_id)

    objects[2] = (
        f"<< /Type /Pages /Count {len(pages)} /Kids "
        f"[{' '.join(f'{kid} 0 R' for kid in kids)}] >>"
    ).encode("latin-1")

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0] * (max(objects) + 1)
    for object_id in range(1, len(offsets)):
        offsets[object_id] = len(out)
        out += f"{object_id} 0 obj\n".encode("latin-1")
        out += objects[object_id]
        out += b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(offsets)}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for object_id in range(1, len(offsets)):
        out += f"{offsets[object_id]:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer << /Size {len(offsets)} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode("latin-1")
    return bytes(out)
