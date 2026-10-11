"""SEC EDGAR: fetch the latest 10-K, extract Risk Factors and MD&A, chunk them.

Some filers (JPMorgan, for example) put the MD&A in an Annual Report
exhibit (EX-13) instead of the 10-K body. When the 10-K body yields a
section that is too short to be real, we look for that exhibit.
"""

import re

import requests
from bs4 import BeautifulSoup

from src.config import PREDECESSOR_CIKS, secret
from src.tools.cache import memory

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_BASE = "https://data.sec.gov/submissions/"
SUBMISSIONS_URL = SUBMISSIONS_BASE + "CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{acc_dash}-index.htm"

# A section shorter than this is a table-of-contents hit, not the section.
MIN_SECTION_WORDS = 500

ITEM_END = r"[\.\:\-\s]"
MDNA_HEADING = r"management[’']s discussion and analysis"
MDNA_END = (
    r"item\s*7a" + ITEM_END + r"|item\s*8" + ITEM_END
    + r"|management[’']s report on internal control"
    + r"|report of independent registered public accounting firm"
)

# (section name, start heading, end heading). Headings vary by filer, so
# each pattern allows "Item 1A." / "ITEM 1A:" / "Item 1A -" and so on.
SECTIONS = [
    (
        "risk_factors",
        r"item\s*1a" + ITEM_END,
        r"item\s*1b" + ITEM_END + r"|item\s*1c" + ITEM_END + r"|item\s*2" + ITEM_END,
    ),
    ("mdna", r"item\s*7" + ITEM_END, MDNA_END),
]
# Fallback patterns for an Annual Report exhibit, which has no "Item 7".
EXHIBIT_SECTIONS = {"mdna": (MDNA_HEADING, MDNA_END)}


def _get(url: str) -> requests.Response:
    user_agent = secret("EDGAR_USER_AGENT")  # "Your Name your@email.com"
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=30)
    resp.raise_for_status()
    return resp


def _clean(text: str) -> str:
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def _html_to_text(html: str) -> str:
    return _clean(BeautifulSoup(html, "html.parser").get_text(" "))


@memory.cache
def get_cik(ticker: str) -> str:
    """Ticker to zero-padded 10-digit CIK."""
    for row in _get(TICKER_MAP_URL).json().values():
        if row["ticker"].upper() == ticker.upper():
            return str(row["cik_str"]).zfill(10)
    raise ValueError(f"No CIK found for {ticker}")


def _all_filings(cik: str):
    """Yield (form, accession, primary_doc, filed), newest first, every page.

    The submissions JSON holds only the latest ~1,000 filings in "recent";
    heavy filers (Exxon, for example) push an 8-month-old 10-K onto the
    paginated files listed under filings.files.
    """
    data = _get(SUBMISSIONS_URL.format(cik=cik)).json()
    pages = [data["filings"]["recent"]]
    for extra in data["filings"].get("files", []):
        pages.append(_get(SUBMISSIONS_BASE + extra["name"]).json())
    for page in pages:
        yield from zip(
            page["form"], page["accessionNumber"],
            page["primaryDocument"], page["filingDate"],
        )


def _find_10k(cik: str) -> tuple[str, str, str] | None:
    """(accession, primary_doc, filed) of the newest 10-K, or None."""
    for form, acc, doc, filed in _all_filings(cik):
        if form == "10-K":
            return acc, doc, filed
    return None


@memory.cache
def get_latest_10k(ticker: str) -> dict:
    """Metadata and plain text of the most recent 10-K, plus any EX-13 text.

    Tries the ticker's current registrant first, then its predecessor
    entity (config.PREDECESSOR_CIKS) for companies that reorganized.
    """
    ticker = ticker.upper()
    candidates = [get_cik(ticker)]
    if ticker in PREDECESSOR_CIKS:
        candidates.append(PREDECESSOR_CIKS[ticker])
    for cik in candidates:
        found = _find_10k(cik)
        if found is None:
            continue
        acc, doc, filed = found
        acc_nodash = acc.replace("-", "")
        url = ARCHIVE_URL.format(cik=int(cik), acc=acc_nodash, doc=doc)
        return {
            "ticker": ticker,
            "cik": cik,
            "filed": filed,
            "url": url,
            "text": _html_to_text(_get(url).text),
            "exhibit_text": _get_annual_report_exhibit(cik, acc, acc_nodash),
        }
    raise ValueError(
        f"No 10-K found for {ticker} (CIKs tried: {', '.join(candidates)}). "
        "If the company reorganized, add its prior CIK to PREDECESSOR_CIKS."
    )


def _get_annual_report_exhibit(cik: str, acc: str, acc_nodash: str) -> str:
    """Text of the EX-13 (Annual Report) document, or '' if there is none."""
    index_url = INDEX_URL.format(cik=int(cik), acc=acc_nodash, acc_dash=acc)
    soup = BeautifulSoup(_get(index_url).text, "html.parser")
    for row in soup.select("table.tableFile tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all("td")]
        link = row.find("a", href=True)
        if link and any(c.upper().startswith("EX-13") for c in cells):
            href = link["href"]
            if "/ix?doc=" in href:  # inline-XBRL viewer link
                href = href.split("/ix?doc=")[-1]
            return _html_to_text(_get("https://www.sec.gov" + href).text)
    return ""


def extract_section(text: str, start_pat: str, end_pat: str) -> str:
    """Return the longest span between a start and the next end heading.

    The table of contents also lists each heading, so the longest span is
    almost always the real section rather than the TOC entry.
    """
    best = ""
    starts = [m.start() for m in re.finditer(start_pat, text, flags=re.I)]
    for s in starts:
        end = re.search(end_pat, text[s + 20:], flags=re.I)
        if end:
            span = text[s:s + 20 + end.start()]
            if len(span) > len(best):
                best = span
    return best.strip()


def extract_sections(filing: dict) -> dict[str, str]:
    """Each section's text, with heading-based fallbacks for odd filers.

    When "Item 7" yields only a cross-reference (under MIN_SECTION_WORDS),
    try the plain section heading in the main document first (JPMorgan
    embeds its whole annual report there), then in the EX-13 exhibit.
    """
    out = {}
    for name, start, end in SECTIONS:
        section = extract_section(filing["text"], start, end)
        if len(section.split()) < MIN_SECTION_WORDS and name in EXHIBIT_SECTIONS:
            alt_start, alt_end = EXHIBIT_SECTIONS[name]
            for text in (filing["text"], filing.get("exhibit_text", "")):
                candidate = extract_section(text, alt_start, alt_end)
                if len(candidate.split()) >= MIN_SECTION_WORDS:
                    section = candidate
                    break
        out[name] = section
    return out


def chunk_text(text: str, size: int = 300, overlap: int = 50) -> list[str]:
    """Split into overlapping word windows (about 300 words each)."""
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


def build_filing_chunks(ticker: str) -> list[dict]:
    """Chunks with metadata, ready to embed into the vector store."""
    filing = get_latest_10k(ticker)
    chunks = []
    for section, text in extract_sections(filing).items():
        for i, chunk in enumerate(chunk_text(text)):
            chunks.append({
                "id": f"{ticker}-{section}-{i}",
                "text": chunk,
                "ticker": ticker,
                "section": section,
                "filed": filing["filed"],
                "url": filing["url"],
            })
    return chunks
