"""Tests for agents/common/trafilatura_utils.py's extract_text_and_title.

Real gap this closes: three call sites (page_fetcher.py's HTTP and
Playwright paths, action_executor.py's page-text extraction) read
document.text/.title as bare attribute access on trafilatura's
bare_extraction() result, typed Document | dict[str, Any] | None by
trafilatura's own stubs -- the dict case would have raised AttributeError
rather than degrading gracefully like every other extraction failure in
this codebase.
"""

from types import SimpleNamespace

from agents.common.trafilatura_utils import extract_text_and_title


def test_returns_empty_text_and_none_title_for_none_document():
    assert extract_text_and_title(None) == ("", None)


def test_extracts_text_and_title_from_a_document_like_object():
    document = SimpleNamespace(text="the real content", title="the real title")

    assert extract_text_and_title(document) == ("the real content", "the real title")


def test_extracts_text_and_title_from_a_plain_dict():
    """trafilatura's own type stubs say bare_extraction() can return a
    plain dict, not just a Document object -- this is the case bare
    attribute access (document.text) would have crashed on."""
    document = {"text": "dict-shaped content", "title": "dict-shaped title"}

    assert extract_text_and_title(document) == ("dict-shaped content", "dict-shaped title")


def test_missing_text_defaults_to_empty_string_object():
    document = SimpleNamespace(text=None, title="has a title")

    text, title = extract_text_and_title(document)

    assert text == ""
    assert title == "has a title"


def test_missing_title_defaults_to_none_dict():
    document = {"text": "has text", "title": ""}

    text, title = extract_text_and_title(document)

    assert text == "has text"
    assert title is None


def test_object_with_no_text_or_title_attributes_at_all():
    """A Document-like object missing the attributes entirely (not just
    None-valued) must still degrade gracefully, not raise AttributeError."""
    document = object()

    assert extract_text_and_title(document) == ("", None)
