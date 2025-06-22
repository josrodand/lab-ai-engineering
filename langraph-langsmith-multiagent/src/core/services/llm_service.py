from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import AzureChatOpenAI
from src.config.settings import settings

class LLMService:
    def __init__(self):
        """Initialize LLM service with Azure OpenAI."""
        self.llm = AzureChatOpenAI(
            openai_api_key=settings.UOC_API_KEY,
            azure_endpoint=settings.UOC_ENDPOINT,
            deployment_name=settings.UOC_MODEL_NAME,
            api_version=settings.UOC_API_VERSION
        )

    def process_music_query(self, query: str, user_profile: Dict[str, Any], tools: List[Tool]) -> Dict[str, Any]:
        """
        Process a music-related query using the LLM.
        
        Args:
            query: User's query
            user_profile: User's music preferences
            tools: Available tools for the LLM
            
        Returns:
            Dictionary with response and any updated preferences
        """
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
                You are a music catalog assistant. Use the available tools to answer questions about music.
                If the user mentions music preferences, update their profile accordingly.
                Format the response as JSON with the following structure:
                {
                    "response": "Your response here",
                    "music_preferences": {
                        "genres": ["genre1", "genre2"],
                        "artists": ["artist1", "artist2"]
                    }
                }
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="""
                User query: {query}
                User preferences: {user_profile}
            """)
        ])

        # Generate response
        response = self.llm.invoke(prompt.format_messages(
            query=query,
            user_profile=user_profile
        ))

        # Parse response
        try:
            result = response.content
            return result
        except Exception as e:
            return {"error": f"Error processing query: {str(e)}"}

    def process_invoice_query(self, query: str, customer_info: Dict[str, Any], tools: List[Tool]) -> Dict[str, Any]:
        """
        Process an invoice-related query using the LLM.
        
        Args:
            query: User's query
            customer_info: Customer information
            tools: Available tools for the LLM
            
        Returns:
            Dictionary with response
        """
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
                You are an invoice information assistant. Use the available tools to answer questions about invoices.
                Verify customer information before providing sensitive data.
                Format the response as JSON with the following structure:
                {
                    "response": "Your response here",
                    "sensitive": boolean
                }
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="""
                User query: {query}
                Customer info: {customer_info}
            """)
        ])

        # Generate response
        response = self.llm.invoke(prompt.format_messages(
            query=query,
            customer_info=customer_info
        ))

        # Parse response
        try:
            result = response.content
            return result
        except Exception as e:
            return {"error": f"Error processing query: {str(e)}"}
