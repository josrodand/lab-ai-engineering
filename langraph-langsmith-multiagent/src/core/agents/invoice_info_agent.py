from typing import Any, Dict, List, Optional
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from src.core.agents.base_agent import BaseAgent
from src.core.services.database_service import DatabaseService
from src.core.services.llm_service import LLMService

class InvoiceInfoAgent(BaseAgent):
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
        """Initialize the tools for the invoice information agent."""
        self.tools = [
            self.get_customer_info,
            self.get_invoice_details,
            self.get_purchase_history,
        ]
        self.llm = self.llm.bind_tools(self.tools)

    @tool
    def get_customer_info(self, customer_id: str) -> Dict[str, Any]:
        """
        Get customer information from the database.
        
        Args:
            customer_id: ID of the customer
            
        Returns:
            Dictionary with customer information
        """
        return self.db_service.get_customer_info(customer_id)

    @tool
    def get_invoice_details(self, invoice_id: str) -> Dict[str, Any]:
        """
        Get invoice details from the database.
        
        Args:
            invoice_id: ID of the invoice
            
        Returns:
            Dictionary with invoice information
        """
        return self.db_service.get_invoice_details(invoice_id)

    @tool
    def get_purchase_history(self, customer_id: str) -> Dict[str, Any]:
        """
        Get purchase history for a customer.
        
        Args:
            customer_id: ID of the customer
            
        Returns:
            Dictionary with purchase history
        """
        return self.db_service.get_purchase_history(customer_id)

    def get_prompt_template(self) -> ChatPromptTemplate:
        """
        Get the prompt template for the invoice information agent.
        """
        return ChatPromptTemplate.from_messages([
            HumanMessage(
                content="""You are an invoice information assistant specialized in helping customers with billing and purchase history.
                Use the available tools to answer questions about invoices and purchases.
                Verify customer information before providing sensitive data.
                """
            )
        ])

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an invoice-related request.
        
        Args:
            request: Dictionary containing customer query and optional customer_id
            
        Returns:
            Response with invoice information
        """
        # Get customer information
        customer_id = request.get("customer_id")
        query = request.get("query")
        
        # Verify customer information
        if not customer_id:
            return {"error": "Customer ID is required for invoice information"}
            
        # Get customer info
        customer_info = self.get_customer_info(customer_id)
        if not customer_info:
            return {"error": "Customer not found"}
        
        # Process query using LLM
        response = self.llm_service.process_invoice_query(
            query=query,
            customer_info=customer_info,
            tools=self.tools
        )
        
        return response
