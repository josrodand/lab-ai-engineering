from typing import Dict, Any, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from src.config.settings import settings

class DatabaseService:
    def __init__(self):
        """Initialize database connection."""
        self.engine = create_engine(settings.DB_URL)
        self.Session = sessionmaker(bind=self.engine)

    def get_albums_by_artist(self, artist: str) -> Dict[str, Any]:
        """
        Get albums by artist from the database.
        
        Args:
            artist: Name of the artist
            
        Returns:
            Dictionary with album information
        """
        with self.Session() as session:
            query = text("""
                SELECT Album.Title as album_title, Album.AlbumId, Artist.Name as artist_name
                FROM Album
                JOIN Artist ON Album.ArtistId = Artist.ArtistId
                WHERE LOWER(Artist.Name) LIKE LOWER(:artist)
            """)
            result = session.execute(query, {"artist": f"%{artist}%"}).fetchall()
            return {
                "albums": [
                    {
                        "id": row.AlbumId,
                        "title": row.album_title,
                        "artist": row.artist_name
                    }
                    for row in result
                ]
            }

    def get_artist_by_genre(self, genre: str) -> Dict[str, Any]:
        """
        Get artists by genre from the database.
        
        Args:
            genre: Music genre
            
        Returns:
            Dictionary with artist information
        """
        with self.Session() as session:
            query = text("""
                SELECT DISTINCT Artist.Name as artist_name, Artist.ArtistId
                FROM Artist
                JOIN Album ON Artist.ArtistId = Album.ArtistId
                JOIN Track ON Album.AlbumId = Track.AlbumId
                JOIN Genre ON Track.GenreId = Genre.GenreId
                WHERE LOWER(Genre.Name) LIKE LOWER(:genre)
            """)
            result = session.execute(query, {"genre": f"%{genre}%"}).fetchall()
            return {
                "artists": [
                    {
                        "id": row.ArtistId,
                        "name": row.artist_name
                    }
                    for row in result
                ]
            }

    def get_top_tracks(self, artist: str) -> Dict[str, Any]:
        """
        Get top tracks for an artist.
        
        Args:
            artist: Name of the artist
            
        Returns:
            Dictionary with track information
        """
        with self.Session() as session:
            query = text("""
                SELECT Track.Name as track_name, Track.TrackId, Album.Title as album_title
                FROM Track
                JOIN Album ON Track.AlbumId = Album.AlbumId
                JOIN Artist ON Album.ArtistId = Artist.ArtistId
                WHERE LOWER(Artist.Name) LIKE LOWER(:artist)
                ORDER BY Track.PlayCount DESC
                LIMIT 10
            """)
            result = session.execute(query, {"artist": f"%{artist}%"}).fetchall()
            return {
                "tracks": [
                    {
                        "id": row.TrackId,
                        "name": row.track_name,
                        "album": row.album_title
                    }
                    for row in result
                ]
            }

    def get_customer_info(self, customer_id: str) -> Dict[str, Any]:
        """
        Get customer information from the database.
        
        Args:
            customer_id: ID of the customer
            
        Returns:
            Dictionary with customer information
        """
        with self.Session() as session:
            query = text("""
                SELECT CustomerId, FirstName, LastName, Email, Phone, Company
                FROM Customer
                WHERE CustomerId = :customer_id
            """)
            result = session.execute(query, {"customer_id": customer_id}).fetchone()
            if result:
                return {
                    "id": result.CustomerId,
                    "name": f"{result.FirstName} {result.LastName}",
                    "email": result.Email,
                    "phone": result.Phone,
                    "company": result.Company
                }
            return None

    def get_invoice_details(self, invoice_id: str) -> Dict[str, Any]:
        """
        Get invoice details from the database.
        
        Args:
            invoice_id: ID of the invoice
            
        Returns:
            Dictionary with invoice information
        """
        with self.Session() as session:
            query = text("""
                SELECT InvoiceId, InvoiceDate, BillingAddress, Total
                FROM Invoice
                WHERE InvoiceId = :invoice_id
            """)
            result = session.execute(query, {"invoice_id": invoice_id}).fetchone()
            if result:
                return {
                    "id": result.InvoiceId,
                    "date": result.InvoiceDate,
                    "address": result.BillingAddress,
                    "total": float(result.Total)
                }
            return None

    def get_purchase_history(self, customer_id: str) -> Dict[str, Any]:
        """
        Get purchase history for a customer.
        
        Args:
            customer_id: ID of the customer
            
        Returns:
            Dictionary with purchase history
        """
        with self.Session() as session:
            query = text("""
                SELECT Invoice.InvoiceId, Invoice.InvoiceDate, SUM(InvoiceLine.UnitPrice * InvoiceLine.Quantity) as Total
                FROM Invoice
                JOIN InvoiceLine ON Invoice.InvoiceId = InvoiceLine.InvoiceId
                WHERE Invoice.CustomerId = :customer_id
                GROUP BY Invoice.InvoiceId, Invoice.InvoiceDate
                ORDER BY InvoiceDate DESC
                LIMIT 10
            """)
            result = session.execute(query, {"customer_id": customer_id}).fetchall()
            return {
                "purchases": [
                    {
                        "id": row.InvoiceId,
                        "date": row.InvoiceDate,
                        "total": float(row.Total)
                    }
                    for row in result
                ]
            }
