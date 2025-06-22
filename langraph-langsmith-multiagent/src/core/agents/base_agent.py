from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from langgraph import Agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

class BaseAgent(ABC):
    def __init__(
        self,
        llm: Any,
        tools: List[Any] = None,
        memory_saver: MemorySaver = None,
        in_memory_store: InMemoryStore = None,
    ):
        """
        Base class for all agents in the system.
        
        Args:
            llm: Language model instance
            tools: List of tools available to the agent
            memory_saver: Short-term memory checkpointer
            in_memory_store: Long-term memory store
        """
        self.llm = llm
        self.tools = tools or []
        self.memory_saver = memory_saver or MemorySaver()
        self.in_memory_store = in_memory_store or InMemoryStore()
        
        # Initialize the agent with tools
        if self.tools:
            self.llm = self.llm.bind_tools(self.tools)

    @abstractmethod
    def get_prompt_template(self) -> ChatPromptTemplate:
        """Get the prompt template for this agent."""
        pass

    @abstractmethod
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request and return a response."""
        pass

    def _get_user_profile(self, customer_id: str) -> Dict[str, Any]:
        """Retrieve user profile from long-term memory."""
        try:
            return self.in_memory_store.get("user_profiles", customer_id)
        except KeyError:
            return {}

    def _update_user_profile(self, customer_id: str, profile_data: Dict[str, Any]) -> None:
        """Update user profile in long-term memory."""
        self.in_memory_store.put("user_profiles", customer_id, profile_data)

    def _save_checkpoint(self, state: Dict[str, Any]) -> None:
        """Save current state to short-term memory."""
        self.memory_saver.save(state)

    def _load_checkpoint(self, thread_id: str) -> Dict[str, Any]:
        """Load state from short-term memory."""
        return self.memory_saver.load(thread_id)
