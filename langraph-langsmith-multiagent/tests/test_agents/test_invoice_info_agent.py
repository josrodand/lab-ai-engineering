import pytest
from unittest.mock import MagicMock
from src.core.agents.invoice_info_agent import InvoiceInfoAgent
from src.core.services.database_service import DatabaseService
from src.core.services.llm_service import LLMService
import json

def test_process_request():
    # Arrange
    mock_db_service = MagicMock()
    mock_db_service.get_customer_info.return_value = {
        "id": 1,
        "name": "John Doe",
        "email": "john@example.com"
    }
    
    mock_llm_service = MagicMock()
    mock_llm_service.process_invoice_query.return_value = json.dumps({
        "response": "Found invoice details",
        "sensitive": True
    })
    
    agent = InvoiceInfoAgent(
        llm=MagicMock(),
        tools=[],
        in_memory_store=MagicMock()
    )
    agent.db_service = mock_db_service
    agent.llm_service = mock_llm_service
    
    request = {
        "query": "What's my billing history?",
        "customer_id": "1"
    }
    
    # Act
    result = agent.process_request(request)
    
    # Assert
    mock_db_service.get_customer_info.assert_called_once()
    mock_llm_service.process_invoice_query.assert_called_once()
    assert "response" in result
    assert "sensitive" in result

def test_customer_verification():
    # Arrange
    mock_db_service = MagicMock()
    mock_db_service.get_customer_info.return_value = {
        "id": 1,
        "name": "John Doe",
        "email": "john@example.com"
    }
    
    agent = InvoiceInfoAgent(
        llm=MagicMock(),
        tools=[],
        in_memory_store=MagicMock()
    )
    agent.db_service = mock_db_service
    
    # Act
    result = agent._get_customer_info("1")
    
    # Assert
    mock_db_service.get_customer_info.assert_called_once()
    assert "id" in result
    assert result["name"] == "John Doe"

def test_customer_not_found():
    # Arrange
    mock_db_service = MagicMock()
    mock_db_service.get_customer_info.return_value = None
    
    agent = InvoiceInfoAgent(
        llm=MagicMock(),
        tools=[],
        in_memory_store=MagicMock()
    )
    agent.db_service = mock_db_service
    
    # Act
    result = agent._get_customer_info("999")
    
    # Assert
    assert result is None
