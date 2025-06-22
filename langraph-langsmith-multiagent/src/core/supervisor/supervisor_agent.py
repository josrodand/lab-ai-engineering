from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from src.config.settings import settings
from src.core.agents.base_agent import BaseAgent
from src.core.services.database_service import DatabaseService

class SupervisorAgent:
    def __init__(
        self,
        music_agent: BaseAgent,
        invoice_agent: BaseAgent,
        llm: Optional[Any] = None,
    ):
        """
        Initialize the supervisor agent.
        
        Args:
            music_agent: Music catalog agent
            invoice_agent: Invoice information agent
            llm: Language model instance (optional)
        """
        self.music_agent = music_agent
        self.invoice_agent = invoice_agent
        self.llm = llm or AzureChatOpenAI(
            openai_api_key=settings.UOC_API_KEY,
            azure_endpoint=settings.UOC_ENDPOINT,
            deployment_name=settings.UOC_MODEL_NAME,
            api_version=settings.UOC_API_VERSION
        )

    def _get_query_type(self, query: str) -> str:
        """
        Determine the type of query (music or invoice).
        
        Args:
            query: User's query
            
        Returns:
            Type of query ('music' or 'invoice')
        """
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
                You are a query classifier for a customer support system.
                Classify each query as either 'music' or 'invoice' based on its content.
                Music queries are about artists, albums, songs, or music preferences.
                Invoice queries are about billing, purchases, or account information.
            """),
            HumanMessage(content="Query: {query}")
        ])

        response = self.llm.invoke(prompt.format_messages(query=query))
        return response.content.lower()

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a customer support request.
        
        Args:
            request: Dictionary containing customer query and optional customer_id
            
        Returns:
            Response from the appropriate agent
        """
        # Extract request information
        query = request.get("query", "")
        customer_id = request.get("customer_id")
        
        # Get query type
        query_type = self._get_query_type(query)
        
        # Process request with appropriate agent
        if query_type == "music":
            return self.music_agent.process_request(request)
        elif query_type == "invoice":
            return self.invoice_agent.process_request(request)
        else:
            return {
                "error": f"Unknown query type: {query_type}",
                "suggestion": "Please rephrase your query to be more specific about music or billing information."
            }

    def get_prompt_template(self) -> ChatPromptTemplate:
        """
        Get the prompt template for the supervisor agent.
        """
        return ChatPromptTemplate.from_messages([
            SystemMessage(content="""
                You are the supervisor agent for a customer support system.
                Your role is to:
                1. Classify incoming queries as either 'music' or 'invoice'
                2. Route queries to the appropriate specialized agent
                3. Ensure proper handling of customer context and preferences
                
                Remember to:
                - Always verify customer information before sharing sensitive data
                - Update user profiles with new preferences when relevant
                - Maintain conversation context using memory management
            """),
            MessagesPlaceholder(variable_name="chat_history")
        ])

    def handle_customer_verification(self, customer_id: str) -> Dict[str, Any]:
        """
        Handle customer verification process.
        
        Args:
            customer_id: ID of the customer to verify
            
        Returns:
            Verification status and customer information
        """
        # Initialize database service if not exists
        if not hasattr(self, 'db_service'):
            self.db_service = DatabaseService()
        
        # Get customer info from database
        customer_info = self.db_service.get_customer_info(customer_id)
        
        if not customer_info:
            return {
                "verified": False,
                "message": "Customer not found. Please provide valid customer ID.",
                "error": "CustomerNotFound"
            }
        
        return {
            "verified": True,
            "customer_info": customer_info,
            "message": "Customer verified successfully.",
            "customer_id": customer_id
        }

    def update_user_profile(self, customer_id: str, profile_data: Dict[str, Any]) -> None:
        """
        Update user profile across all agents.
        
        Args:
            customer_id: ID of the customer
            profile_data: New profile data to update
        """
        self.music_agent._update_user_profile(customer_id, profile_data)
        # No need to update invoice agent as it doesn't use user profiles

    def get_user_profile(self, customer_id: str) -> Dict[str, Any]:
        """
        Get user profile from the music agent.
        
        Args:
            customer_id: ID of the customer
            
        Returns:
            User profile data
        """
        try:
            return self.music_agent._get_user_profile(customer_id)
        except Exception as e:
            return {
                "error": f"Failed to get user profile: {str(e)}",
                "customer_id": customer_id
            }
