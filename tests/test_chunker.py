from langchain_core.documents import Document

from src.ingestion.chunker import chunk_documents, clean_page_text, find_boilerplate_lines, make_chunk_id, normalize

FOOTER = "Andriy Burkov The Hundred-Page Machine Learning Book - Draft {}"
TOPICS = ["trees", "kernels", "boosting", "clustering", "ranking", "embeddings", "gradients"]
BODY = "Machine learning is a subfield of computer science that builds algorithms from examples. " * 6


def pages(n=10):
    return [
        Document(page_content=f"{BODY} This page covers {TOPICS[i % len(TOPICS)]} in {'depth' * i}.\n{FOOTER.format(i)}", metadata={"file_name": "book.pdf", "page": i})
        for i in range(1, n + 1)
    ]


def test_normalize_fixes_ligatures_and_lost_glyphs():
    assert normalize("deﬁned") == "defined"
    assert normalize("di\x00erent e\x00cient") == "different efficient"


def test_normalize_repairs_line_break_hyphenation():
    assert normalize("algo-\nrithm") == "algorithm"


def test_running_footer_with_changing_page_number_is_removed():
    docs = pages()
    boilerplate = find_boilerplate_lines(docs)
    cleaned = clean_page_text(docs[3].page_content, boilerplate)
    assert "Draft" not in cleaned
    assert cleaned.startswith("Machine learning")


def test_chunk_ids_are_deterministic_and_unique():
    first = [c.metadata["chunk_id"] for c in chunk_documents(pages())]
    second = [c.metadata["chunk_id"] for c in chunk_documents(pages())]
    assert first == second
    assert len(set(first)) == len(first)
    assert first[0].startswith("book.pdf#p1#c0#")


def test_chunk_id_changes_when_text_changes():
    assert make_chunk_id("a.pdf", 1, 0, "x") != make_chunk_id("a.pdf", 1, 0, "y")


def test_short_pages_are_dropped():
    docs = [Document(page_content="Cover page", metadata={"file_name": "b.pdf", "page": 1})]
    assert chunk_documents(docs) == []
