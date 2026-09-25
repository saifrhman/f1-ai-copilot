"""Ingestion tests on real PDF files: validation, page extraction, chunking and metadata."""

import json

import pytest

from core_modules.rule_checker.fia_rag import ChunkingConfig, DocumentError, RAGConfigurationError
from core_modules.rule_checker.fia_rag.ingestion import (
    chunk_pages,
    discover_documents,
    load_and_chunk,
    load_pages,
    split_page_header,
)
from core_modules.rule_checker.fia_rag.rules import extract_headings, extract_rule_ids, is_supported
from tests.helpers import fia_page, write_pdf


def _one_doc(tmp_path, pages, name="regs.pdf"):
    write_pdf(tmp_path / name, pages)
    return discover_documents(tmp_path).documents


# ------------------------------------------------------------------ discovery / validation


def test_missing_directory_fails_with_download_instructions(tmp_path):
    with pytest.raises(RAGConfigurationError, match="fetch_fia_regulations.py"):
        discover_documents(tmp_path / "does-not-exist")


def test_directory_without_pdfs_fails(tmp_path):
    (tmp_path / "notes.txt").write_text("not a regulation")
    with pytest.raises(RAGConfigurationError, match="No FIA regulation PDFs"):
        discover_documents(tmp_path)


def test_html_saved_as_pdf_is_rejected(tmp_path):
    (tmp_path / "section_b.pdf").write_text("<!DOCTYPE html><html><body>Access denied</body></html>")
    with pytest.raises(DocumentError, match="not a PDF"):
        discover_documents(tmp_path)


def test_empty_file_is_rejected(tmp_path):
    (tmp_path / "section_b.pdf").write_bytes(b"")
    with pytest.raises(DocumentError, match="empty"):
        discover_documents(tmp_path)


def test_corrupt_pdf_fails_clearly_when_parsed(tmp_path):
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4\n" + b"\x00garbage" * 200)
    documents = discover_documents(tmp_path).documents
    with pytest.raises(DocumentError, match="broken.pdf"):
        load_pages(documents[0])


def test_truncated_pdf_fails_clearly(tmp_path):
    source = write_pdf(tmp_path / "full.pdf", ["B1.1 Some regulation text that is long enough to count."])
    data = source.read_bytes()
    source.unlink()
    (tmp_path / "truncated.pdf").write_bytes(data[: len(data) // 3])
    documents = discover_documents(tmp_path).documents
    with pytest.raises(DocumentError):
        load_pages(documents[0])


def test_pdf_without_extractable_text_is_rejected(tmp_path):
    documents = _one_doc(tmp_path, [None, ""])
    with pytest.raises(DocumentError, match="no.*extractable text|none of its 2 pages"):
        load_pages(documents[0])


def test_manifest_hash_mismatch_is_detected(tmp_path):
    documents = _one_doc(tmp_path, ["B1.1 Regulation text for the manifest test case."])
    (tmp_path / "manifest.json").write_text(
        json.dumps({"documents": [{"filename": documents[0].filename, "sha256": "0" * 64, "section": "B"}]})
    )
    with pytest.raises(DocumentError, match="SHA-256"):
        discover_documents(tmp_path)


def test_manifest_metadata_is_attached(tmp_path):
    documents = _one_doc(tmp_path, ["B1.1 Regulation text for the manifest test case."])
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "filename": documents[0].filename,
                        "sha256": documents[0].sha256,
                        "section": "B",
                        "source_url": "https://www.fia.com/example.pdf",
                    }
                ]
            }
        )
    )
    document = discover_documents(tmp_path).documents[0]
    assert (document.section, document.source_url) == ("B", "https://www.fia.com/example.pdf")


def test_byte_identical_duplicates_are_indexed_once(tmp_path):
    write_pdf(tmp_path / "a.pdf", ["B1.1 Identical regulation document content."])
    (tmp_path / "b_copy.pdf").write_bytes((tmp_path / "a.pdf").read_bytes())
    result = discover_documents(tmp_path)
    assert [d.filename for d in result.documents] == ["a.pdf"]
    assert "byte-identical" in result.warnings[0]


# ------------------------------------------------------------------ page extraction


def test_every_page_is_parsed_with_one_based_page_numbers_and_blank_pages_reported(tmp_path):
    documents = _one_doc(tmp_path, ["First page regulation text here.", None, "Third page regulation text here."])
    pages, report = load_pages(documents[0])
    assert [p.metadata["page"] for p in pages] == [1, 3]
    assert report.pages == 3 and report.pages_with_text == 2 and report.empty_pages == [2]
    assert all(p.metadata["source"] == "regs.pdf" for p in pages)
    assert "First page regulation text" in pages[0].page_content


def test_fia_page_header_is_removed_and_printed_page_label_kept(tmp_path):
    documents = _one_doc(tmp_path, [fia_page(7, "B1.6.3 Driving in the Pit Lane a. A speed limit of 80km/h applies.")])
    pages, _ = load_pages(documents[0])
    assert pages[0].page_content.startswith("B1.6.3 Driving in the Pit Lane")
    assert "Issue 08" not in pages[0].page_content and "©" not in pages[0].page_content
    assert pages[0].metadata["page_label"] == "B7"


def test_split_page_header_leaves_ordinary_text_untouched():
    text = "Article 12.4 The car must remain within the limit. Issue 3 of the bulletin."
    assert split_page_header(text) == (None, "", text)


def test_contents_pages_are_skipped_but_reported(tmp_path):
    contents = "\n".join(f"B{i}.{j} Section heading number {i} {j} {10 + i}" for i in range(1, 5) for j in range(1, 4))
    documents = _one_doc(tmp_path, [contents, "B1.1 The actual regulation text. It applies to every competitor."])
    pages, report = load_pages(documents[0])
    assert [p.metadata["page"] for p in pages] == [2]
    assert report.contents_pages == [1]


def test_broken_ligature_glyphs_are_repaired(tmp_path):
    documents = _one_doc(tmp_path, ["The o`icials must reduce speed su‘iciently near the ‘Pit Lane’ entry."])
    text = load_pages(documents[0])[0][0].page_content
    assert "officials" in text and "sufficiently" in text
    assert "‘Pit Lane’" in text  # genuine quotation marks are untouched


# ------------------------------------------------------------------ chunking


def _long_article(n_paragraphs=30):
    return "\n".join(
        f"B2.{i} Rule heading {i}\nParagraph {i} explains a distinct obligation number {i} for every competitor in detail."
        for i in range(1, n_paragraphs + 1)
    )


def test_chunk_size_and_overlap_are_honoured(tmp_path):
    documents = _one_doc(tmp_path, [_long_article()])
    pages, _ = load_pages(documents[0])
    small = chunk_pages(pages, ChunkingConfig(chunk_size=300, chunk_overlap=0))
    overlapping = chunk_pages(pages, ChunkingConfig(chunk_size=300, chunk_overlap=120))
    large = chunk_pages(pages, ChunkingConfig(chunk_size=2000, chunk_overlap=0))
    assert all(len(c.text) <= 300 for c in small + overlapping)
    assert len(large) < len(small) <= len(overlapping)
    assert any(len(c.text) > 300 for c in large)


@pytest.mark.parametrize("size,overlap", [(1000, 1000), (1000, 1500), (1000, -1), (10, 0)])
def test_invalid_chunk_configuration_is_rejected(size, overlap):
    with pytest.raises(RAGConfigurationError):
        ChunkingConfig(chunk_size=size, chunk_overlap=overlap)


def test_unusually_long_page_is_split_and_keeps_page_metadata(tmp_path):
    documents = _one_doc(tmp_path, ["Short page one with enough regulation words.", _long_article(80)])
    chunks, reports = load_and_chunk(documents, ChunkingConfig(chunk_size=500, chunk_overlap=50))
    page_two = [c for c in chunks if c.metadata["page"] == 2]
    assert len(page_two) > 5
    assert reports[0].chunks == len(chunks)
    assert all(c.metadata["source"] == "regs.pdf" for c in chunks)
    assert [c.metadata["chunk_index"] for c in page_two] == list(range(len(page_two)))


def test_chunks_are_meaningful_and_ids_deterministic_even_for_repeated_text(tmp_path):
    repeated = "B3.1 Identical wording appears on two different pages of the regulations."
    documents = _one_doc(tmp_path, [repeated, "7", repeated])
    first, _ = load_and_chunk(documents, ChunkingConfig(chunk_size=500, chunk_overlap=50))
    second, _ = load_and_chunk(documents, ChunkingConfig(chunk_size=500, chunk_overlap=50))
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first) == 2  # "7"-only page is layout debris
    assert all(c.text.strip() for c in first)


def test_article_context_is_carried_across_page_boundaries(tmp_path):
    documents = _one_doc(
        tmp_path,
        [
            "B4.2 Unsafe Release\nB4.2.1 A car must not be released from its pit stop position in an unsafe condition.",
            "continued text of the same article describing the responsibilities of competitors in the pit lane.",
        ],
    )
    chunks, _ = load_and_chunk(documents, ChunkingConfig(chunk_size=500, chunk_overlap=0))
    assert chunks[0].metadata["nearest_rule"] in {"B4.2", "B4.2.1"}
    assert chunks[1].metadata["page"] == 2
    assert chunks[1].metadata["nearest_rule"] == "B4.2.1"
    assert "B4.2.1" in chunks[0].metadata["rule_ids"]


def test_multiple_documents_keep_their_own_source(tmp_path):
    write_pdf(tmp_path / "section_b.pdf", ["B1.1 Sporting regulation text for document one."])
    write_pdf(tmp_path / "section_c.pdf", ["C1.1 Technical regulation text for document two."])
    chunks, reports = load_and_chunk(discover_documents(tmp_path).documents, ChunkingConfig())
    assert {c.metadata["source"] for c in chunks} == {"section_b.pdf", "section_c.pdf"}
    assert [r.filename for r in reports] == ["section_b.pdf", "section_c.pdf"]


# ------------------------------------------------------------------ rule identifiers


def test_rule_ids_follow_the_2026_fia_numbering():
    text = (
        "The Grid position referred to in Article B2.3.4c will remain vacant. B2.3.5 Sprint Session "
        "Classification. F1 Cars and the F1 Team, Issue 08 of 2026, pursuant to Article E4.1.1.a. "
        "ARTICLE D2: OBLIGATIONS and legacy Article 12.4.1."
    )
    assert extract_rule_ids(text) == ["B2.3.4c", "B2.3.5", "E4.1.1.a", "D2", "12.4.1"]
    assert [rule for _, rule in extract_headings(text)] == ["B2.3.5", "D2"]


def test_invented_identifiers_with_any_letter_are_extracted_when_introduced_as_articles():
    assert extract_rule_ids("According to Article Z1.1, teams may refuel.") == ["Z1.1"]
    assert extract_rule_ids("The F1 Team and F1 Cars") == []


def test_identifiers_split_by_pdf_extraction_are_rejoined():
    assert extract_rule_ids("as described in Article C5. 2.8 of these regulations") == ["C5.2.8"]


def test_rule_support_accepts_parents_but_not_invented_children():
    assert is_supported("B2.3", ["B2.3.5"])
    assert is_supported("B2.3.4", ["B2.3.4c"])
    assert not is_supported("B2.3.5.1", ["B2.3.5"])
    assert not is_supported("B2.35", ["B2.3.5"])
    assert is_supported("B1.6.3a", ["B1.6.3"]) and is_supported("B1.6.3.a", ["B1.6.3"])
    assert not is_supported("Z1.1", ["B1.6.3"]) and not is_supported("B1.6.4a", ["B1.6.3"])
