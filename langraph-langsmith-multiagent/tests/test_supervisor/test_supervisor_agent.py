import pytest
from unittest.mock import MagicMock
from src.core.supervisor.supervisor_agent import SupervisorAgent
from src.core.agents.music_catalog_agent import MusicCatalogAgent
from src.core.agents.invoice_info_agent import InvoiceInfoAgent
import json

def test_process_request_music_query():
    # Arrange
    mock_music_agent = MagicMock()
    mock_music_agent.process_request.return_value = json.dumps({
        "response": "Found some albums",
        "music_preferences": {"genres": ["rock"], "artists": ["Artist 1"]}
    })
    
    mock_invoice_agent = MagicMock()
    
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "music"
    
    supervisor = SupervisorAgent(
        music_agent=mock_music_agent,
        invoice_agent=mock_invoice_agent,
        llm=mock_llm
    )
    
    request = {
        "query": "What albums does Artist 1 have?",
        "customer_id": "123"
    }
    
    # Act
    result = supervisor.process_request(request)
    
    # Assert
    mock_llm.invoke.assert_called_once()
    mock_music_agent.process_request.assert_called_once()
    assert "response" in result
    assert "music_preferences" in result

def test_process_request_invoice_query():
    # Arrange
    mock_music_agent = MagicMock()
    
    mock_invoice_agent = MagicMock()
    mock_invoice_agent.process_request.return_value = json.dumps({
        "response": "Found invoice details",
        "sensitive": True
    })
    
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "invoice"
    
    supervisor = SupervisorAgent(
        music_agent=mock_music_agent,
        invoice_agent=mock_invoice_agent,
        llm=mock_llm
    )
    
    request = {
        "query": "What's my billing history?",
        "customer_id": "123"
    }
    
    # Act
    result = supervisor.process_request(request)
    
    # Assert
    mock_llm.invoke.assert_called_once()
    mock_invoice_agent.process_request.assert_called_once()
    assert "response" in result
    assert "sensitive" in result
    assert result["sensitive"] is True

def test_process_request_unknown_query_type():
    # Arrange
    mock_music_agent = MagicMock()
    mock_invoice_agent = MagicMock()
    
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "unknown"
    
    supervisor = SupervisorAgent(
        music_agent=mock_music_agent,
        invoice_agent=mock_invoice_agent,
        llm=mock_llm
    )
    
    request = {
        "query": "What's the weather like?",
        "customer_id": "123"
    }
    
    # Act
    result = supervisor.process_request(request)
    
    # Assert
    assert "error" in result
    assert "Unknown query type" in result["error"]
    assert "suggestion" in result

def test_handle_customer_verification_success():
    # Arrange
    mock_db_service = MagicMock()
    mock_db_service.get_customer_info.return_value = {
        "id": 1,
        "name": "John Doe",
        "email": "john@example.com"
    }
    
    mock_music_agent = MagicMock()
    mock_invoice_agent = MagicMock()
    
    supervisor = SupervisorAgent(
        music_agent=mock_music_agent,
        invoice_agent=mock_invoice_agent
    )
    supervisor.db_service = mock_db_service
    
    # Act
    result = supervisor.handle_customer_verification("1")
    
    # Assert
    mock_db_service.get_customer_info.assert_called_once_with("1")
    assert result["verified"] is True
    assert "customer_info" in result
    assert "message" in result

def test_handle_customer_verification_failure():
    # Arrange
    mock_db_service = MagicMock()
    mock_db_service.get_customer_info.return_value = None
    
    mock_music_agent = MagicMock()
    mock_invoice_agent = MagicMock()
    
    supervisor = SupervisorAgent(
        music_agent=mock_music_agent,
        invoice_agent=mock_invoice_agent
    )
    supervisor.db_service = mock_db_service
    
    # Act
    result = supervisor.handle_customer_verification("999")
    
    # Assert
    mock_db_service.get_customer_info.assert_called_once_with("999")
    assert result["verified"] is False
    assert "error" in result
    assert "CustomerNotFound" in result["error"]

def test_get_user_profile_success():
    # Arrange
    mock_store = MagicMock()
    mock_store.get.return_value = {
        "genres": ["rock"],
        "artists": ["Artist 1"]
    }
    
    mock_music_agent = MagicMock()
    mock_music_agent._get_user_profile.return_value = {
        "genres": ["rock"],
        "artists": ["Artist 1"]
    }
    
    mock_invoice_agent = MagicMock()
    
    supervisor = SupervisorAgent(
        music_agent=mock_music_agent,
        invoice_agent=mock_invoice_agent
    )
    
    # Act
    result = supervisor.get_user_profile("123")
    
    # Assert
    mock_music_agent._get_user_profile.assert_called_once_with("123")
    assert "genres" in result
    assert "artists" in result

def test_get_user_profile_error():
    # Arrange
    mock_store = MagicMock()
    mock_store.get.side_effect = Exception("Store error")
    
    mock_music_agent = MagicMock()
    mock_music_agent._get_user_profile.side_effect = Exception("Failed to get user profile")
    
    mock_invoice_agent = MagicMock()
    
    supervisor = SupervisorAgent(
        music_agent=mock_music_agent,
        invoice_agent=mock_invoice_agent
    )
    
    # Act
    result = supervisor.get_user_profile("123")
    
    # Assert
    assert "error" in result
    assert "Failed to get user profile" in result["error"]
