import pytest
from unittest.mock import MagicMock
from sqlalchemy import text
from src.core.services.database_service import DatabaseService
from src.config.settings import settings

def test_get_albums_by_artist():
    # Arrange
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [
        (1, "Album 1", "Artist 1"),
        (2, "Album 2", "Artist 1")
    ]
    mock_session.execute.return_value = mock_result
    
    db_service = DatabaseService()
    db_service.Session = MagicMock(return_value=mock_session)
    
    # Act
    result = db_service.get_albums_by_artist("Artist 1")
    
    # Assert
    mock_session.execute.assert_called_once()
    assert "albums" in result
    assert len(result["albums"]) == 2
    assert result["albums"][0]["title"] == "Album 1"

def test_get_customer_info():
    # Arrange
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchone.return_value = (
        1, "John", "Doe", "john@example.com", "123-456-7890", "Company Inc."
    )
    mock_session.execute.return_value = mock_result
    
    db_service = DatabaseService()
    db_service.Session = MagicMock(return_value=mock_session)
    
    # Act
    result = db_service.get_customer_info("1")
    
    # Assert
    mock_session.execute.assert_called_once()
    assert "id" in result
    assert result["name"] == "John Doe"
    assert result["email"] == "john@example.com"

def test_get_customer_info_not_found():
    # Arrange
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchone.return_value = None
    mock_session.execute.return_value = mock_result
    
    db_service = DatabaseService()
    db_service.Session = MagicMock(return_value=mock_session)
    
    # Act
    result = db_service.get_customer_info("999")
    
    # Assert
    assert result is None

def test_get_purchase_history():
    # Arrange
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [
        (1, "2024-01-01", 100.0),
        (2, "2024-01-02", 50.0)
    ]
    mock_session.execute.return_value = mock_result
    
    db_service = DatabaseService()
    db_service.Session = MagicMock(return_value=mock_session)
    
    # Act
    result = db_service.get_purchase_history("1")
    
    # Assert
    mock_session.execute.assert_called_once()
    assert "purchases" in result
    assert len(result["purchases"]) == 2
    assert result["purchases"][0]["total"] == 100.0
