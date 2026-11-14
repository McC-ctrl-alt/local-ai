"""Write the fictional department PDFs used by the policy demo."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Each document is fictional. Page 2 holds the fact the demo asks about.
DOCUMENTS = [
    {
        "path": "docs/tax/estimated-tax-calendar.pdf",
        "pages": [
            [
                "ASTER MERIDIAN GROUP",
                "Tax Department",
                "FICTIONAL SAMPLE. Aster Meridian Group is not a real company.",
                "Estimated Tax Calendar for this calendar year.",
                "This calendar states when federal estimated income tax vouchers are due.",
                "It does not state any state tax date.",
            ],
            [
                "Federal estimated tax voucher due dates.",
                "The first quarterly voucher is due on April 15.",
                "The second quarterly voucher is due on June 15.",
                "The third quarterly voucher is due on September 15.",
                "The fourth quarterly voucher is due on January 15 of the following year.",
                "A high-income taxpayer uses a safe harbor of 110 percent of the prior-year tax.",
            ],
        ],
    },
    {
        "path": "docs/tax/meals-deduction-memo.pdf",
        "pages": [
            [
                "ASTER MERIDIAN GROUP",
                "Tax Department memorandum",
                "FICTIONAL SAMPLE. Aster Meridian Group is not a real company.",
                "This memo states how client meals and entertainment are deducted.",
                "It does not state estimated tax due dates.",
            ],
            [
                "Deduction rules.",
                "A client business meal is 50 percent deductible when the business purpose is written down within 15 days.",
                "An entertainment expense is 0 percent deductible.",
                "A team social event that is not a client meal is not deductible.",
            ],
        ],
    },
    {
        "path": "docs/legal/privilege-protocol.pdf",
        "pages": [
            [
                "ASTER MERIDIAN GROUP",
                "Legal Department",
                "FICTIONAL SAMPLE. Aster Meridian Group is not a real company.",
                "Privilege protocol for requests that seek legal advice.",
                "This protocol does not state contract review times.",
            ],
            [
                "Privilege rules.",
                "An email that seeks legal advice must include the marking Privileged and Confidential in the subject line.",
                "Legal advice must not be forwarded to outside vendors.",
                "A new outside-counsel matter needs written approval from the general counsel when expected fees exceed 25000 dollars.",
            ],
        ],
    },
    {
        "path": "docs/legal/contract-review-standard.pdf",
        "pages": [
            [
                "ASTER MERIDIAN GROUP",
                "Legal Department",
                "FICTIONAL SAMPLE. Aster Meridian Group is not a real company.",
                "Contract review standard for vendor paperwork.",
                "This standard does not state meal deduction rates.",
            ],
            [
                "Review times.",
                "A standard vendor contract is reviewed within 10 business days.",
                "A mutual nondisclosure agreement is reviewed within 3 business days.",
                "The firm standard governing law is the State of Delaware.",
                "Contracts are not reviewed on weekends.",
            ],
        ],
    },
    {
        "path": "docs/hr/parental-leave-policy.pdf",
        "pages": [
            [
                "ASTER MERIDIAN GROUP",
                "Human Resources",
                "FICTIONAL SAMPLE. Aster Meridian Group is not a real company.",
                "Parental leave policy for employees of the firm.",
                "This policy does not state remote work limits.",
            ],
            [
                "Leave length.",
                "An employee with 12 months of continuous service receives 16 weeks of paid parental leave.",
                "The employee must request leave at least 30 days before the expected start date.",
                "Unused parental leave is not paid out at separation.",
            ],
        ],
    },
    {
        "path": "docs/hr/remote-work-policy.pdf",
        "pages": [
            [
                "ASTER MERIDIAN GROUP",
                "Human Resources",
                "FICTIONAL SAMPLE. Aster Meridian Group is not a real company.",
                "Remote work policy for employees of the firm.",
                "This policy does not state parental leave length.",
            ],
            [
                "Remote work limits.",
                "An employee may work remotely for up to 3 days each week.",
                "Core hours are 10:00 to 15:00 in the employee local time zone.",
                "The annual home-office equipment stipend is 750 dollars.",
            ],
        ],
    },
]


def pdf_escape(text):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(pages):
    """Text-only PDF. Each page is a list of lines. pypdf can extract them."""
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    kids = []
    next_id = 4
    for lines in pages:
        commands = ["BT", "/F1 11 Tf", "72 740 Td", "16 TL"]
        for line in lines:
            commands.append(f"({pdf_escape(line)}) Tj")
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


def main():
    for document in DOCUMENTS:
        path = ROOT / document["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(build_pdf(document["pages"]))
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
