"""Unit tests for document extraction (loaders) and chunking."""

import io

import docx
import pytest

from ai_research_assistant.core.config import Settings
from ai_research_assistant.core.exceptions import IngestionError, UnsupportedFileTypeError
from ai_research_assistant.db.models.document import DocumentType
from ai_research_assistant.services.ingestion.chunking import DocumentChunker
from ai_research_assistant.services.ingestion.loaders import extract_document, resolve_document_type

_MINIMAL_PDF_BYTES = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 200 200] /Contents 5 0 R >>
endobj
4 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
5 0 obj
<< /Length 44 >>
stream
BT /F1 24 Tf 10 100 Td (Hello World) Tj ET
endstream
endobj
xref
0 6
0000000000 65535 f
trailer
<< /Size 6 /Root 1 0 R >>
startxref
0
%%EOF"""


def _build_docx_bytes(*, title: str | None = None, paragraphs: list[str] | None = None) -> bytes:
    document = docx.Document()
    if title is not None:
        document.core_properties.title = title
    for paragraph in paragraphs or ["Hello from a Word document."]:
        document.add_paragraph(paragraph)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_resolve_document_type_maps_known_content_types() -> None:
    assert resolve_document_type("application/pdf") is DocumentType.PDF
    assert resolve_document_type("text/plain") is DocumentType.TXT
    assert resolve_document_type("text/markdown") is DocumentType.MARKDOWN
    assert (
        resolve_document_type(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        is DocumentType.DOCX
    )


def test_resolve_document_type_rejects_unknown_content_type() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        resolve_document_type("image/png")


def test_extract_document_pdf() -> None:
    extracted = extract_document(raw=_MINIMAL_PDF_BYTES, content_type="application/pdf")

    assert extracted.document_type is DocumentType.PDF
    assert "Hello World" in extracted.text
    assert extracted.page_count == 1


def test_extract_document_pdf_with_metadata_title(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeMetadata:
        title = "PDF Title"

    class FakePage:
        def extract_text(self) -> str:
            return "Page text"

    class FakeReader:
        def __init__(self, _stream: object) -> None:
            self.pages = [FakePage()]
            self.metadata = FakeMetadata()

    monkeypatch.setattr("ai_research_assistant.services.ingestion.loaders.PdfReader", FakeReader)

    extracted = extract_document(raw=b"irrelevant", content_type="application/pdf")

    assert extracted.title == "PDF Title"


def test_extract_document_docx_with_title() -> None:
    raw = _build_docx_bytes(title="My Report", paragraphs=["First paragraph.", "Second paragraph."])

    extracted = extract_document(
        raw=raw,
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    assert extracted.document_type is DocumentType.DOCX
    assert extracted.title == "My Report"
    assert "First paragraph." in extracted.text
    assert "Second paragraph." in extracted.text


def test_extract_document_docx_without_title() -> None:
    raw = _build_docx_bytes(paragraphs=["Untitled content."])

    extracted = extract_document(
        raw=raw,
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    assert extracted.title is None


def test_extract_document_txt() -> None:
    extracted = extract_document(raw=b"Plain text content.", content_type="text/plain")

    assert extracted.document_type is DocumentType.TXT
    assert extracted.text == "Plain text content."
    assert extracted.page_count is None
    assert extracted.title is None


def test_extract_document_markdown() -> None:
    extracted = extract_document(raw=b"# Heading\n\nBody text.", content_type="text/markdown")

    assert extracted.document_type is DocumentType.MARKDOWN
    assert "Heading" in extracted.text


def test_extract_document_rejects_unsupported_content_type() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        extract_document(raw=b"whatever", content_type="image/png")


def test_extract_document_rejects_empty_text() -> None:
    with pytest.raises(IngestionError):
        extract_document(raw=b"   \n  ", content_type="text/plain")


def test_extract_document_wraps_parser_failures() -> None:
    with pytest.raises(IngestionError):
        extract_document(raw=b"not a real pdf", content_type="application/pdf")


def test_document_chunker_splits_text_into_overlapping_chunks() -> None:
    settings = Settings(chunk_size=20, chunk_overlap=5)
    chunker = DocumentChunker(settings)

    text = "This is a reasonably long piece of text used to validate chunking behavior end to end."
    chunks = chunker.split(text)

    assert len(chunks) > 1
    assert all(chunk.token_count >= 1 for chunk in chunks)
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_document_chunker_handles_empty_text() -> None:
    settings = Settings(chunk_size=100, chunk_overlap=10)
    chunker = DocumentChunker(settings)

    assert chunker.split("") == []
