"""Regenerate the fictional Northwind Advisory PDFs.

Run from the repository root:

    python3 scripts/build_sample_documents.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from textpdf import write_text_pdf


OUTPUT_DIR = ROOT / "sample_documents"

# Every file is fiction. Names, dates, and dollar amounts were invented for
# this demo and are repeated here so the PDF text stays easy to audit.
DOCUMENTS = {
    "engagement-letter-harborline-markets.pdf": [
        [
            "SYNTHETIC DEMO DOCUMENT.",
            "This file is fictional and was written for a software demonstration.",
            "It is not legal advice, and it does not describe a real client or a real firm.",
            "Northwind Advisory LLC is a fictional professional-services firm.",
            "This engagement letter is between Northwind Advisory LLC and Harborline Markets, Inc., which is also fictional.",
            "The letter is dated March 3, 2026.",
            "Harborline Markets, Inc. asks Northwind Advisory LLC to review its monthly close procedures for the fictional fiscal year 2026.",
            "The engagement partner named in this letter is Priya Shah.",
        ],
        [
            "Fees for the Harborline Markets engagement are stated on this page.",
            "The fixed professional fee for this engagement is $18,500.",
            "Harborline Markets, Inc. will pay $9,250 when this letter is signed.",
            "Harborline Markets, Inc. will pay the remaining $9,250 when the final memo is delivered.",
            "Northwind Advisory LLC will not bill a success fee on this engagement.",
        ],
        [
            "The Harborline Markets engagement lasts 90 days from March 3, 2026.",
            "The engagement end date in this letter is June 1, 2026.",
            "This letter does not authorize Northwind Advisory LLC to prepare a tax return for Harborline Markets, Inc.",
            "Priya Shah will supervise the work and sign the final memo.",
            "Either party may end the engagement by giving written notice.",
        ],
    ],
    "engagement-letter-cedar-pine-retail.pdf": [
        [
            "SYNTHETIC DEMO DOCUMENT.",
            "This file is fictional and was written for a software demonstration.",
            "It is not legal advice, and it does not describe a real client or a real firm.",
            "This engagement letter is between Northwind Advisory LLC and Cedar & Pine Retail Cooperative, which is fictional.",
            "The letter is dated June 15, 2026.",
            "Cedar & Pine Retail Cooperative asks Northwind Advisory LLC to observe the annual inventory count at the fictional Cedar Warehouse.",
            "The fieldwork lead named in this letter is Andre Williams.",
        ],
        [
            "Fees and dates for the Cedar & Pine engagement are stated on this page.",
            "The professional fee for the inventory observation is $42,000.",
            "Out-of-pocket expenses for this engagement are billed at cost with no markup.",
            "Fieldwork at the Cedar Warehouse starts on July 8, 2026.",
            "The inventory observation report is due on August 12, 2026.",
        ],
        [
            "Andre Williams will attend the count and draft the report.",
            "The report will describe the count Northwind Advisory LLC observed.",
            "The report will not value the inventory.",
            "Cedar & Pine Retail Cooperative will give the team access to the warehouse on July 8, 2026.",
        ],
    ],
    "confidentiality-policy.pdf": [
        [
            "SYNTHETIC DEMO DOCUMENT.",
            "This file is fictional and was written for a software demonstration.",
            "It is not legal advice, and it does not describe a real client or a real firm.",
            "Northwind Advisory LLC confidentiality policy NW-POL-04 is effective January 1, 2026.",
            "This policy applies to every staff member and contractor of the fictional firm.",
            "Staff must not discuss client matters in elevators, rideshares, or other public places.",
            "Client files may not be stored on a personal cloud account.",
        ],
        [
            "Breach reporting for Northwind Advisory LLC is stated on this page.",
            "A suspected confidentiality breach must be reported to the Managing Principal within one business day.",
            "The report may be made by email or by phone.",
            "The Managing Principal records the report in the incident log.",
            "The duty of confidentiality lasts through the engagement and for two years after the engagement ends.",
        ],
    ],
    "document-retention-policy.pdf": [
        [
            "SYNTHETIC DEMO DOCUMENT.",
            "This file is fictional and was written for a software demonstration.",
            "It is not legal advice, and it does not describe a real client or a real firm.",
            "Northwind Advisory LLC document retention policy NW-POL-07 is effective January 1, 2026.",
            "This policy covers engagement records of the fictional firm.",
            "Email about a matter is kept with the engagement file.",
        ],
        [
            "Retention periods for Northwind Advisory LLC are stated on this page.",
            "Final engagement files are kept for seven years after the engagement ends.",
            "Draft workpapers are deleted 90 days after the final report is issued, unless a litigation hold applies.",
            "A litigation hold is issued only by the Managing Principal.",
            "Staff must not delete records that a litigation hold covers.",
        ],
    ],
    "billing-and-expenses-policy.pdf": [
        [
            "SYNTHETIC DEMO DOCUMENT.",
            "This file is fictional and was written for a software demonstration.",
            "It is not legal advice, and it does not describe a real client or a real firm.",
            "Northwind Advisory LLC billing and expenses policy NW-POL-02 is effective January 1, 2026.",
            "This policy tells staff how invoices and expenses work.",
            "A signed engagement letter controls the fee when it names one.",
        ],
        [
            "Invoice terms for Northwind Advisory LLC are stated on this page.",
            "Invoices are due within 15 days of the invoice date.",
            "Unpaid balances accrue a late charge of 1.5 percent per month.",
            "Travel, meals, or lodging over $75 require an itemized receipt.",
            "Out-of-pocket client expenses are billed at cost with no markup.",
            "A staff member may not offer a fee discount without written approval from the Managing Principal.",
        ],
    ],
    "conflicts-of-interest-policy.pdf": [
        [
            "SYNTHETIC DEMO DOCUMENT.",
            "This file is fictional and was written for a software demonstration.",
            "It is not legal advice, and it does not describe a real client or a real firm.",
            "Northwind Advisory LLC conflicts policy NW-POL-03 is effective January 1, 2026.",
            "The Conflicts Counsel screens every new engagement before the firm accepts it.",
            "The firm does not accept a new engagement when the Conflicts Counsel records an unresolved conflict.",
        ],
        [
            "Personal investment disclosures for Northwind Advisory LLC are stated on this page.",
            "A staff member must disclose a personal investment of more than $5,000 in a current client.",
            "A staff member must also disclose a personal investment of more than $5,000 in a prospective client.",
            "The disclosure goes to the Conflicts Counsel before work begins.",
            "The Managing Principal does not waive a conflict by silence.",
        ],
    ],
}


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    for filename, pages in DOCUMENTS.items():
        destination = OUTPUT_DIR / filename
        write_text_pdf(destination, pages)
        print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
