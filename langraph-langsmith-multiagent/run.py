import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.config.settings import settings
from src.core.agents.base_agent import BaseAgent
from src.core.agents.music_catalog_agent import MusicCatalogAgent
from src.core.agents.invoice_info_agent import InvoiceInfoAgent
from src.core.supervisor.supervisor_agent import SupervisorAgent

app = FastAPI(title="Multi-Agent Customer Support System")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents
music_agent = MusicCatalogAgent()
invoice_agent = InvoiceInfoAgent()
supervisor = SupervisorAgent(music_agent, invoice_agent)

@app.post("/api/v1/support")
async def handle_customer_support(request: dict):
    """
    Handle customer support requests.
    
    Args:
        request: Dictionary containing customer query and optional customer_id
    
    Returns:
        Response from the appropriate agent
    """
    try:
        # Get customer_id from request or create new one
        customer_id = request.get("customer_id", None)
        
        # Process request through supervisor
        response = supervisor.process_request(request)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "run:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug"
    )
