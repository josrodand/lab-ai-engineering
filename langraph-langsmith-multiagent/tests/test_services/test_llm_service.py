import pytest
from unittest.mock import MagicMock
from src.core.services.llm_service import LLMService
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
import json

def test_process_music_query():
    # Arrange
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = json.dumps({"response": "Music response", "music_preferences": {"genres": ["rock"], "artists": ["The Beatles"]}})
    
    llm_service = LLMService()
    llm_service.llm = mock_llm
    
    tools = [Tool(name="get_albums_by_artist", description="Get albums by artist", func=lambda x: x)]
    user_profile = {"genres": ["pop"], "artists": ["Coldplay"]}
    query = "What are some good rock albums?"
    
    # Act
    result = llm_service.process_music_query(query, user_profile, tools)
    
    # Assert
    mock_llm.invoke.assert_called_once()
    assert "response" in result
    assert "music_preferences" in result
    assert "rock" in result["music_preferences"]["genres"]

def test_process_invoice_query():
    # Arrange
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = json.dumps({"response": "Invoice response", "sensitive": True})
    
    llm_service = LLMService()
    llm_service.llm = mock_llm
    
    tools = [Tool(name="get_invoice_details", description="Get invoice details", func=lambda x: x)]
    customer_info = {"id": 1, "name": "John Doe"}
    query = "What's my billing history?"
    
    # Act
    result = llm_service.process_invoice_query(query, customer_info, tools)
    
    # Assert
    mock_llm.invoke.assert_called_once()
    assert "response" in result
    assert "sensitive" in result
    assert result["sensitive"] is True

def test_process_music_query_error_handling():
    # Arrange
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = 'Invalid JSON'
    
    llm_service = LLMService()
    llm_service.llm = mock_llm
    
    tools = [Tool(name="get_albums_by_artist", description="Get albums by artist", func=lambda x: x)]
    user_profile = {}
    query = "What are some good rock albums?"
    
    # Act
    result = llm_service.process_music_query(query, user_profile, tools)
    
    # Assert
    assert "error" in result
    assert "Error processing query" in result["error"]
