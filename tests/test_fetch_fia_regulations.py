"""FIA downloader tests with a fake HTTP session (no network): discovery, validation, atomic updates."""

import hashlib
import json

import pytest

from scripts.fetch_fia_regulations import discover_pdf_urls, fetch_regulations, validate_pdf_bytes
from tests.helpers import write_pdf

CATEGORY = "https://www.fia.com/regulation/category/2182"


def _link(section, issue, date):
    name = f"fia_2026_f1_regulations_-_section_{section.lower()}_example_-_iss_{issue:02d}_-_{date}.pdf"
    return f'<a href="/system/files/documents/{name}">Section {section}</a>', f"https://www.fia.com/system/files/documents/{name}"


class FakeResponse:
    def __init__(self, url, content=b"", text="", status=200, content_type="application/pdf"):
        self.url, self.content, self.text, self.status_code = url, content, text, status
        self.headers = {"content-type": content_type}

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, pages):
        self.pages = pages
        self.headers = {}
        self.requested = []

    def get(self, url, timeout=None, allow_redirects=True):
        self.requested.append(url)
        value = self.pages[url]
        if isinstance(value, FakeResponse):
            return value
        if isinstance(value, bytes):
            return FakeResponse(url, content=value)
        return FakeResponse(url, text=value, content_type="text/html")


@pytest.fixture
def pdf_bytes(tmp_path):
    def make(text):
        return write_pdf(tmp_path / f"{hashlib.md5(text.encode()).hexdigest()}.pdf", [text]).read_bytes()

    return make


def _site(pdf_bytes, issues):
    anchors, pages = [], {}
    for section, issue, date in issues:
        anchor, url = _link(section, issue, date)
        anchors.append(anchor)
        pages[url] = pdf_bytes(f"Section {section} issue {issue} regulation text for testing.")
    pages[CATEGORY] = "\n".join(anchors)
    return pages


def test_discovery_picks_latest_issue_regardless_of_page_order():
    old, old_url = _link("B", 5, "2026-02-27")
    new, new_url = _link("B", 8, "2026-08-05")
    for page in (old + new, new + old):
        assert discover_pdf_urls(page, CATEGORY, 2026) == {"B": new_url}


def test_discovery_ignores_other_years_and_non_section_pdfs():
    page = (
        '<a href="/docs/fia_2025_f1_regulations_-_section_b_sporting_-_iss_09_-_2025-12-01.pdf">x</a>'
        '<a href="/docs/fia_2026_formula_1_sporting_regulations_pu_-_issue_7_-_2024-10-17.pdf">y</a>'
    )
    assert discover_pdf_urls(page, CATEGORY, 2026) == {}


def test_discovery_uses_the_regulation_year_not_issue_dates():
    page = (
        '<a href="/docs/fia_2027_f1_regulations_-_section_c_technical_-_iss_2_-_2026-08-05_0.pdf">2027 C</a>'
        '<a href="/docs/fia_2026_f1_regulations_-_section_c_technical_-_iss_20_-_2026-08-05.pdf">2026 C</a>'
        '<a href="/download.php?file=fia_2026_f1_regulations_-_section_a_x.pdf">query-string link</a>'
    )
    found = discover_pdf_urls(page, CATEGORY, 2026)
    assert list(found) == ["C"] and "fia_2026_f1" in found["C"]
    assert discover_pdf_urls(page, CATEGORY, 2027)["C"].endswith("iss_2_-_2026-08-05_0.pdf")


def test_dry_run_downloads_nothing(tmp_path, pdf_bytes):
    session = FakeSession(_site(pdf_bytes, [("A", 3, "2026-06-25")]))
    manifest = fetch_regulations(tmp_path / "docs", 2026, ["A"], CATEGORY, dry_run=True, session=session)
    assert manifest["documents"][0]["issue"] == 3
    assert session.requested == [CATEGORY]
    assert not (tmp_path / "docs").exists()


def test_download_writes_validated_files_and_correct_manifest(tmp_path, pdf_bytes):
    session = FakeSession(_site(pdf_bytes, [("A", 3, "2026-06-25"), ("B", 8, "2026-08-05")]))
    out = tmp_path / "docs"
    fetch_regulations(out, 2026, ["A", "B"], CATEGORY, session=session)
    manifest = json.loads((out / "manifest.json").read_text())
    assert [d["section"] for d in manifest["documents"]] == ["A", "B"]
    for entry in manifest["documents"]:
        data = (out / entry["filename"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()
        assert entry["bytes"] == len(data) and entry["pages"] == 1
        assert entry["filename"] in entry["source_url"]
    assert not [p for p in out.iterdir() if p.name.startswith(".")]  # staging cleaned up


def test_new_issue_replaces_superseded_file_but_keeps_user_files(tmp_path, pdf_bytes):
    out = tmp_path / "docs"
    fetch_regulations(out, 2026, ["B"], CATEGORY, session=FakeSession(_site(pdf_bytes, [("B", 5, "2026-02-27")])))
    (out / "my_notes.pdf").write_bytes(b"%PDF-1.4 user file")
    fetch_regulations(
        out, 2026, ["B"], CATEGORY, session=FakeSession(_site(pdf_bytes, [("B", 5, "2026-02-27"), ("B", 8, "2026-08-05")]))
    )
    names = sorted(p.name for p in out.glob("*.pdf"))
    assert names == ["fia_2026_f1_regulations_-_section_b_example_-_iss_08_-_2026-08-05.pdf", "my_notes.pdf"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["removed_superseded_files"] == ["fia_2026_f1_regulations_-_section_b_example_-_iss_05_-_2026-02-27.pdf"]


def test_html_masquerading_as_pdf_is_rejected_and_nothing_is_replaced(tmp_path, pdf_bytes):
    out = tmp_path / "docs"
    fetch_regulations(out, 2026, ["A"], CATEGORY, session=FakeSession(_site(pdf_bytes, [("A", 3, "2026-06-25")])))
    before = json.loads((out / "manifest.json").read_text())

    pages = _site(pdf_bytes, [("A", 4, "2026-09-01"), ("B", 8, "2026-08-05")])
    _, b_url = _link("B", 8, "2026-08-05")
    pages[b_url] = FakeResponse(b_url, content=b"<!DOCTYPE html><html>Access denied</html>", content_type="text/html")
    with pytest.raises(RuntimeError, match="Expected a PDF"):
        fetch_regulations(out, 2026, ["A", "B"], CATEGORY, session=FakeSession(pages))

    assert json.loads((out / "manifest.json").read_text()) == before  # old set untouched
    assert sorted(p.name for p in out.glob("*.pdf")) == [before["documents"][0]["filename"]]
    assert not [p for p in out.iterdir() if p.name.startswith(".")]


def test_validation_rejects_truncated_and_corrupt_pdfs(pdf_bytes):
    good = pdf_bytes("A valid regulation page for validation testing.")
    assert validate_pdf_bytes(good, "u") == 1
    with pytest.raises(RuntimeError, match="truncated"):
        validate_pdf_bytes(good[: len(good) // 2], "u")
    with pytest.raises(RuntimeError, match="not a readable PDF"):
        validate_pdf_bytes(b"%PDF-1.4\n" + b"garbage" * 50 + b"\n%%EOF\n", "u")
    with pytest.raises(RuntimeError, match="Empty"):
        validate_pdf_bytes(b"", "u")


def test_missing_section_fails_clearly(tmp_path, pdf_bytes):
    session = FakeSession(_site(pdf_bytes, [("A", 3, "2026-06-25")]))
    with pytest.raises(RuntimeError, match="section\\(s\\): C"):
        fetch_regulations(tmp_path, 2026, ["A", "C"], CATEGORY, session=session)


def test_partial_section_run_keeps_the_other_sections(tmp_path, pdf_bytes):
    out = tmp_path / "docs"
    fetch_regulations(out, 2026, ["A", "B"], CATEGORY, session=FakeSession(_site(pdf_bytes, [("A", 3, "2026-06-25"), ("B", 5, "2026-02-27")])))
    manifest = fetch_regulations(
        out, 2026, ["A"], CATEGORY, session=FakeSession(_site(pdf_bytes, [("A", 4, "2026-09-01"), ("B", 5, "2026-02-27")]))
    )
    names = sorted(p.name for p in out.glob("*.pdf"))
    assert names == [
        "fia_2026_f1_regulations_-_section_a_example_-_iss_04_-_2026-09-01.pdf",
        "fia_2026_f1_regulations_-_section_b_example_-_iss_05_-_2026-02-27.pdf",
    ]
    written = json.loads((out / "manifest.json").read_text())
    assert [d["section"] for d in written["documents"]] == ["A", "B"]
    assert manifest["removed_superseded_files"] == ["fia_2026_f1_regulations_-_section_a_example_-_iss_03_-_2026-06-25.pdf"]


def test_unreadable_previous_manifest_still_prunes_superseded_official_files(tmp_path, pdf_bytes):
    out = tmp_path / "docs"
    fetch_regulations(out, 2026, ["B"], CATEGORY, session=FakeSession(_site(pdf_bytes, [("B", 5, "2026-02-27")])))
    (out / "manifest.json").write_text("{ not json")
    fetch_regulations(out, 2026, ["B"], CATEGORY, session=FakeSession(_site(pdf_bytes, [("B", 8, "2026-08-05")])))
    assert sorted(p.name for p in out.glob("*.pdf")) == ["fia_2026_f1_regulations_-_section_b_example_-_iss_08_-_2026-08-05.pdf"]
