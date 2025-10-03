import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vector_store import SearchResults


class TestSearchResultsIsEmpty:

    def test_is_empty_returns_true_when_no_documents(self):
        results = SearchResults(documents=[], metadata=[], distances=[])

        assert results.is_empty()

    def test_is_empty_returns_false_when_documents_exist(self):
        results = SearchResults(documents=["doc"], metadata=[{}], distances=[0.1])

        assert not results.is_empty()

    def test_is_empty_returns_true_when_documents_is_empty_list(self):
        results = SearchResults(documents=[], metadata=[], distances=[], error=None)

        assert results.is_empty()


class TestSearchResultsFromChroma:

    def test_from_chroma_creates_search_results_from_chroma_response(self):
        chroma_results = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"key": "value1"}, {"key": "value2"}]],
            "distances": [[0.1, 0.2]],
        }

        results = SearchResults.from_chroma(chroma_results)

        assert len(results.documents) == 2

    def test_from_chroma_extracts_first_element_from_nested_lists(self):
        chroma_results = {
            "documents": [["first_doc", "second_doc"]],
            "metadatas": [[{"id": 1}, {"id": 2}]],
            "distances": [[0.5, 0.6]],
        }

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents[0] == "first_doc"
        assert results.metadata[0]["id"] == 1
        assert results.distances[0] == 0.5

    def test_from_chroma_handles_empty_response(self):
        chroma_results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        results = SearchResults.from_chroma(chroma_results)

        assert results.is_empty()

    def test_from_chroma_sets_error_to_none_by_default(self):
        chroma_results = {
            "documents": [["doc"]],
            "metadatas": [[{}]],
            "distances": [[0.1]],
        }

        results = SearchResults.from_chroma(chroma_results)

        assert results.error is None


class TestSearchResultsEmpty:

    def test_empty_creates_search_results_with_error_message(self):
        results = SearchResults.empty("Test error message")

        assert results.error == "Test error message"

    def test_empty_creates_search_results_with_empty_lists(self):
        results = SearchResults.empty("Error")

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []

    def test_empty_creates_search_results_that_is_empty(self):
        results = SearchResults.empty("No results found")

        assert results.is_empty()


class TestSearchResultsDataclass:

    def test_search_results_can_be_created_with_all_fields(self):
        results = SearchResults(
            documents=["doc1", "doc2"],
            metadata=[{"id": 1}, {"id": 2}],
            distances=[0.1, 0.2],
            error="Test error",
        )

        assert len(results.documents) == 2
        assert len(results.metadata) == 2
        assert len(results.distances) == 2
        assert results.error == "Test error"

    def test_search_results_error_defaults_to_none(self):
        results = SearchResults(documents=["doc"], metadata=[{}], distances=[0.1])

        assert results.error is None
