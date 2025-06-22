from typing import Any, Dict, List, Optional
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool
from src.core.agents.base_agent import BaseAgent
from src.core.services.database_service import DatabaseService
from src.core.services.llm_service import LLMService

class MusicCatalogAgent(BaseAgent):
    def __init__(
        self,
        llm: Any,
        tools: List[Any] = None,
        memory_saver: Optional[MemorySaver] = None,
        in_memory_store: Optional[InMemoryStore] = None,
    ):
        super().__init__(llm, tools, memory_saver, in_memory_store)
        self.db_service = DatabaseService()
        self.llm_service = LLMService()
        
        # Initialize tools if not provided
        if not tools:
            self._initialize_tools()

    def _initialize_tools(self):
        """Initialize the tools for the music catalog agent."""
        self.tools = [
            self.get_albums_by_artist,
            self.get_artist_by_genre,
            self.get_top_tracks,
        ]
        self.llm = self.llm.bind_tools(self.tools)

    @tool
    def get_albums_by_artist(self, artist: str) -> Dict[str, Any]:
        """
        Get albums by artist from the database.
        
        Args:
            artist: Name of the artist
            
        Returns:
            Dictionary with album information
        """
        return self.db_service.get_albums_by_artist(artist)

    @tool
    def get_artist_by_genre(self, genre: str) -> Dict[str, Any]:
        """
        Get artists by genre from the database.
        
        Args:
            genre: Music genre
            
        Returns:
            Dictionary with artist information
        """
        return self.db_service.get_artist_by_genre(genre)

    @tool
    def get_top_tracks(self, artist: str) -> Dict[str, Any]:
        """
        Get top tracks for an artist.
        
        Args:
            artist: Name of the artist
            
        Returns:
            Dictionary with track information
        """
        return self.db_service.get_top_tracks(artist)

    def get_prompt_template(self) -> ChatPromptTemplate:
        """
        Get the prompt template for the music catalog agent.
        """
        return ChatPromptTemplate.from_messages([
            HumanMessage(
                content="""You are a music catalog assistant specialized in helping customers find music information.
                Use the available tools to answer questions about artists, albums, and tracks.
                If the customer mentions music preferences, update their profile accordingly.
                """
            )
        ])

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a music-related request.
        
        Args:
            request: Dictionary containing customer query and optional customer_id
            
        Returns:
            Response with music information
        """
        # Get customer information
        customer_id = request.get("customer_id")
        query = request.get("query")
        
        # Get user profile
        user_profile = self._get_user_profile(customer_id) if customer_id else {}
        
        # Process query using LLM
        response = self.llm_service.process_music_query(
            query=query,
            user_profile=user_profile,
            tools=self.tools
        )
        
        # Update user profile if new preferences were mentioned
        if "music_preferences" in response:
            self._update_user_profile(customer_id, response["music_preferences"])
        
        return response
