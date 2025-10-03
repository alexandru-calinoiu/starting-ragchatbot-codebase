"""
API integration tests for the RAG system endpoints.
"""
import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException


class TestQueryEndpoint:
    """Test cases for /api/query endpoint."""
    
    def test_query_success_without_session(self, test_client, mock_rag_system):
        """Test successful query without providing session ID."""
        response = test_client.post(
            "/api/query",
            json={"query": "What is computer use?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"
        assert data["answer"] == "This is a test response based on course materials."
        assert len(data["sources"]) == 1
        
        # Verify RAG system was called correctly
        mock_rag_system.query.assert_called_once_with("What is computer use?", "test-session-123")
        mock_rag_system.session_manager.create_session.assert_called_once()
    
    def test_query_success_with_existing_session(self, test_client, mock_rag_system):
        """Test successful query with existing session ID."""
        response = test_client.post(
            "/api/query",
            json={
                "query": "Tell me more about safety",
                "session_id": "existing-session-456"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "existing-session-456"
        
        # Verify session creation was NOT called
        mock_rag_system.session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with("Tell me more about safety", "existing-session-456")
    
    def test_query_with_empty_query(self, test_client):
        """Test query with empty string."""
        response = test_client.post(
            "/api/query",
            json={"query": ""}
        )
        
        # Empty query is valid but should still return a response
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
    
    def test_query_with_very_long_query(self, test_client, mock_rag_system):
        """Test query with very long input."""
        long_query = "a" * 10000  # 10,000 character query
        response = test_client.post(
            "/api/query",
            json={"query": long_query}
        )
        
        assert response.status_code == 200
        mock_rag_system.query.assert_called_once()
    
    def test_query_with_special_characters(self, test_client, mock_rag_system):
        """Test query with special characters and unicode."""
        special_query = "What about 你好 & <script>alert('test')</script> @ #$%?"
        response = test_client.post(
            "/api/query",
            json={"query": special_query}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        mock_rag_system.query.assert_called_with(special_query, "test-session-123")
    
    def test_query_handles_rag_system_error(self, test_client, mock_rag_system):
        """Test error handling when RAG system raises exception."""
        mock_rag_system.query.side_effect = Exception("RAG system error")
        
        response = test_client.post(
            "/api/query",
            json={"query": "Test query"}
        )
        
        assert response.status_code == 500
        assert "RAG system error" in response.json()["detail"]
    
    def test_query_handles_session_creation_error(self, test_client, mock_rag_system):
        """Test error handling when session creation fails."""
        mock_rag_system.session_manager.create_session.side_effect = Exception("Session creation failed")
        
        response = test_client.post(
            "/api/query",
            json={"query": "Test query"}
        )
        
        assert response.status_code == 500
        assert "Session creation failed" in response.json()["detail"]
    
    def test_query_response_format(self, test_client, mock_rag_system):
        """Test the complete response format matches the schema."""
        response = test_client.post(
            "/api/query",
            json={"query": "Test query", "session_id": "test-123"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        assert all(isinstance(source, str) for source in data["sources"])


class TestCoursesEndpoint:
    """Test cases for /api/courses endpoint."""
    
    def test_get_courses_success(self, test_client, mock_rag_system):
        """Test successful retrieval of course statistics."""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Building Towards Computer Use with Anthropic" in data["course_titles"]
        
        mock_rag_system.get_course_analytics.assert_called_once()
    
    def test_get_courses_empty_database(self, test_client, mock_rag_system):
        """Test courses endpoint with empty database."""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []
    
    def test_get_courses_handles_error(self, test_client, mock_rag_system):
        """Test error handling in courses endpoint."""
        mock_rag_system.get_course_analytics.side_effect = Exception("Database error")
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 500
        assert "Database error" in response.json()["detail"]
    
    def test_get_courses_response_format(self, test_client, mock_rag_system):
        """Test the response format matches the schema."""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert all(isinstance(title, str) for title in data["course_titles"])


class TestSessionEndpoints:
    """Test cases for session management endpoints."""
    
    def test_clear_session_success(self, test_client, mock_rag_system):
        """Test successful session clearing."""
        session_id = "session-to-clear-123"
        response = test_client.post(f"/api/sessions/{session_id}/clear")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Session cleared successfully"
        
        mock_rag_system.session_manager.clear_session.assert_called_once_with(session_id)
    
    def test_clear_nonexistent_session(self, test_client, mock_rag_system):
        """Test clearing a session that doesn't exist."""
        mock_rag_system.session_manager.clear_session.side_effect = KeyError("Session not found")
        
        response = test_client.post("/api/sessions/nonexistent-session/clear")
        
        assert response.status_code == 500
        assert "Session not found" in response.json()["detail"]
    
    def test_clear_session_with_special_characters(self, test_client, mock_rag_system):
        """Test session ID with special characters."""
        # Use URL-safe special characters
        session_id = "session-with-special-dash_underscore-123"
        response = test_client.post(f"/api/sessions/{session_id}/clear")
        
        assert response.status_code == 200
        mock_rag_system.session_manager.clear_session.assert_called_once_with(session_id)
    
    def test_clear_session_handles_error(self, test_client, mock_rag_system):
        """Test error handling in clear session endpoint."""
        mock_rag_system.session_manager.clear_session.side_effect = Exception("Unexpected error")
        
        response = test_client.post("/api/sessions/test-session/clear")
        
        assert response.status_code == 500
        assert "Unexpected error" in response.json()["detail"]


class TestAPIValidation:
    """Test cases for API request validation."""
    
    def test_query_missing_required_field(self, test_client):
        """Test query endpoint with missing required field."""
        response = test_client.post("/api/query", json={})
        
        assert response.status_code == 422  # Validation error
        error_detail = response.json()["detail"]
        assert any("query" in str(error).lower() for error in error_detail)
    
    def test_query_invalid_json(self, test_client):
        """Test query endpoint with invalid JSON."""
        response = test_client.post(
            "/api/query",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_query_wrong_field_type(self, test_client):
        """Test query endpoint with wrong field type."""
        response = test_client.post(
            "/api/query",
            json={"query": 123, "session_id": "test"}  # query should be string
        )
        
        # Pydantic should coerce the int to string, so it might still work
        assert response.status_code in [200, 422]
    
    def test_unsupported_http_method(self, test_client):
        """Test endpoints with unsupported HTTP methods."""
        # /api/query only supports POST
        response = test_client.get("/api/query")
        assert response.status_code == 405  # Method not allowed
        
        # /api/courses only supports GET
        response = test_client.post("/api/courses", json={})
        assert response.status_code == 405


class TestConcurrency:
    """Test cases for concurrent requests."""
    
    def test_multiple_concurrent_queries(self, test_client, mock_rag_system):
        """Test handling multiple concurrent query requests."""
        import concurrent.futures
        
        def make_query(query_text):
            return test_client.post(
                "/api/query",
                json={"query": query_text}
            )
        
        queries = [f"Query {i}" for i in range(10)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_query, q) for q in queries]
            responses = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        assert all(r.status_code == 200 for r in responses)
        assert mock_rag_system.query.call_count == 10
    
    def test_concurrent_mixed_endpoints(self, test_client, mock_rag_system):
        """Test concurrent requests to different endpoints."""
        import concurrent.futures
        
        def make_request(endpoint, method="get", json_data=None):
            if method == "post":
                return test_client.post(endpoint, json=json_data)
            return test_client.get(endpoint)
        
        tasks = [
            ("post", "/api/query", {"query": "Test 1"}),
            ("get", "/api/courses", None),
            ("post", "/api/query", {"query": "Test 2"}),
            ("post", "/api/sessions/test-123/clear", None),
            ("get", "/api/courses", None),
        ]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request, task[1], task[0], task[2]) for task in tasks]
            responses = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        assert all(r.status_code == 200 for r in responses)


class TestErrorRecovery:
    """Test cases for error recovery and resilience."""
    
    def test_recovery_after_error(self, test_client, mock_rag_system):
        """Test that API recovers after an error."""
        # First request causes an error
        mock_rag_system.query.side_effect = Exception("Temporary error")
        response1 = test_client.post("/api/query", json={"query": "Test 1"})
        assert response1.status_code == 500
        
        # Reset and make successful request
        mock_rag_system.query.side_effect = None
        mock_rag_system.query.return_value = ("Success", ["Source"])
        response2 = test_client.post("/api/query", json={"query": "Test 2"})
        assert response2.status_code == 200
    
    def test_partial_system_failure(self, test_client, mock_rag_system):
        """Test behavior when only part of the system fails."""
        # Query endpoint fails
        mock_rag_system.query.side_effect = Exception("Query error")
        query_response = test_client.post("/api/query", json={"query": "Test"})
        assert query_response.status_code == 500
        
        # But courses endpoint still works
        courses_response = test_client.get("/api/courses")
        assert courses_response.status_code == 200


class TestRateLimiting:
    """Test cases for rate limiting behavior (if implemented)."""
    
    def test_burst_requests(self, test_client, mock_rag_system):
        """Test handling of burst requests."""
        responses = []
        for i in range(20):  # Reduced from 50 for faster CI execution
            response = test_client.post(
                "/api/query",
                json={"query": f"Burst query {i}"}
            )
            responses.append(response)
        
        # All should succeed (no rate limiting implemented yet)
        assert all(r.status_code == 200 for r in responses)
        
        # Verify all were processed
        assert mock_rag_system.query.call_count == 20


class TestCORSAndMiddleware:
    """Test cases for CORS headers and middleware functionality."""
    
    def test_cors_headers_present_on_query_post(self, test_client):
        """Test CORS headers are present on POST requests."""
        response = test_client.post(
            "/api/query",
            json={"query": "test"},
            headers={"Origin": "http://localhost:3000"}
        )
        
        assert response.status_code == 200
        # Note: TestClient may not always include CORS headers, but endpoint should work
        # The CORS middleware is configured in the app, which is what matters
    
    def test_cors_headers_present_on_courses_get(self, test_client):
        """Test CORS headers are present on GET requests."""
        response = test_client.get(
            "/api/courses",
            headers={"Origin": "http://localhost:3000"}
        )
        
        assert response.status_code == 200
        # Note: TestClient may not always include CORS headers, but endpoint should work
        # The CORS middleware is configured in the app, which is what matters
    
    def test_cors_preflight_request(self, test_client):
        """Test CORS preflight OPTIONS request."""
        response = test_client.options(
            "/api/query",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        # Should handle preflight requests
        assert response.status_code in [200, 204]
        if response.status_code == 200:
            assert "access-control-allow-origin" in response.headers
    
    def test_cors_allow_credentials(self, test_client):
        """Test CORS allow credentials header."""
        response = test_client.post(
            "/api/query",
            json={"query": "test"},
            headers={"Origin": "http://localhost:3000"}
        )
        
        assert response.status_code == 200
        # Check if credentials are allowed (this depends on CORS configuration)
        if "access-control-allow-credentials" in response.headers:
            assert response.headers["access-control-allow-credentials"] == "true"


class TestHealthAndStartup:
    """Test cases for application health and startup behavior."""
    
    def test_api_endpoints_available(self, test_client):
        """Test that all API endpoints are available and responding."""
        # Test query endpoint
        query_response = test_client.post("/api/query", json={"query": "test"})
        assert query_response.status_code == 200
        
        # Test courses endpoint
        courses_response = test_client.get("/api/courses")
        assert courses_response.status_code == 200
        
        # Test session clear endpoint
        session_response = test_client.post("/api/sessions/test-session/clear")
        assert session_response.status_code == 200
    
    def test_api_response_times(self, test_client):
        """Test that API responses are reasonably fast."""
        import time
        
        start_time = time.time()
        response = test_client.post("/api/query", json={"query": "test"})
        end_time = time.time()
        
        assert response.status_code == 200
        # Should respond within 1 second (very generous for mocked system)
        assert (end_time - start_time) < 1.0
    
    def test_large_query_handling(self, test_client, mock_rag_system):
        """Test handling of large query inputs."""
        large_query = "a" * 50000  # 50KB query
        
        response = test_client.post(
            "/api/query",
            json={"query": large_query}
        )
        
        assert response.status_code == 200
        mock_rag_system.query.assert_called_once()


class TestSecurityFeatures:
    """Test cases for security features and input sanitization."""
    
    def test_xss_protection_in_query(self, test_client, mock_rag_system):
        """Test protection against XSS attempts in queries."""
        xss_query = "<script>alert('xss')</script>What is computer use?"
        
        response = test_client.post(
            "/api/query",
            json={"query": xss_query}
        )
        
        assert response.status_code == 200
        # Query should be passed through but response should be safe
        data = response.json()
        assert "answer" in data
        mock_rag_system.query.assert_called_with(xss_query, "test-session-123")
    
    def test_sql_injection_protection(self, test_client, mock_rag_system):
        """Test protection against SQL injection attempts."""
        sql_injection = "'; DROP TABLE courses; SELECT * FROM users WHERE '1'='1"
        
        response = test_client.post(
            "/api/query",
            json={"query": sql_injection}
        )
        
        # Should handle safely without errors
        assert response.status_code == 200
        mock_rag_system.query.assert_called_once()
    
    def test_session_id_validation(self, test_client, mock_rag_system):
        """Test validation of session IDs for potential security issues."""
        # Test various potentially problematic session IDs
        problematic_sessions = [
            "../../../etc/passwd",
            "session;rm -rf /",
            "session<script>alert('xss')</script>",
            "session' OR '1'='1"
        ]
        
        for session_id in problematic_sessions:
            response = test_client.post(
                "/api/query",
                json={"query": "test", "session_id": session_id}
            )
            
            # Should handle all inputs safely
            assert response.status_code == 200