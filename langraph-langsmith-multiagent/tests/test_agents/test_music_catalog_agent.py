import pytest
from unittest.mock import MagicMock
from src.core.agents.music_catalog_agent import MusicCatalogAgent
from src.core.services.database_service import DatabaseService
from src.core.services.llm_service import LLMService
import json

def test_process_request():
    # Arrange
    mock_db_service = MagicMock()
    mock_db_service.get_albums_by_artist.return_value = {
        "albums": [
            {"id": 1, "title": "Album 1", "artist": "Artist 1"}
        ]
    }
    
    mock_llm_service = MagicMock()
    mock_llm_service.process_music_query.return_value = json.dumps({
        "response": "Found some albums",
        "music_preferences": {"genres": ["rock"], "artists": ["Artist 1"]}
    })
    
    agent = MusicCatalogAgent(
        llm=MagicMock(),
        tools=[],
        in_memory_store=MagicMock()
    )
    agent.db_service = mock_db_service
    agent.llm_service = mock_llm_service
    
    request = {
        "query": "What albums does Artist 1 have?",
        "customer_id": "123"
    }
    
    # Act
    result = agent.process_request(request)
    
    # Assert
    mock_db_service.get_albums_by_artist.assert_called_once()
    mock_llm_service.process_music_query.assert_called_once()
    assert "response" in result
    assert "music_preferences" in result

def test_update_user_profile():
    # Arrange
    mock_store = MagicMock()
    agent = MusicCatalogAgent(
        llm=MagicMock(),
        tools=[],
        in_memory_store=mock_store
    )
    
    customer_id = "123"
    profile_data = {"genres": ["rock"], "artists": ["Artist 1"]}
    
    # Act
    agent._update_user_profile(customer_id, profile_data)
    
    # Assert
    mock_store.put.assert_called_once_with("user_profiles", customer_id, profile_data)

def test_get_user_profile():
    # Arrange
    mock_store = MagicMock()
    mock_store.get.return_value = {"genres": ["rock"], "artists": ["Artist 1"]}
    
    agent = MusicCatalogAgent(
        llm=MagicMock(),
        tools=[],
        in_memory_store=mock_store
    )
    
    customer_id = "123"
    
    # Act
    result = agent._get_user_profile(customer_id)
    
    # Assert
    mock_store.get.assert_called_once_with("user_profiles", customer_id)
    assert "genres" in result
    assert "artists" in result
