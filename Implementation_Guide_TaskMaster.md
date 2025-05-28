# Implementation Guide for TaskMaster: Dynamic MCP Agent API

## Overview
This guide provides specific implementation instructions for TaskMaster to build the Dynamic MCP Agent API backend based on the PRD.

## Phase 1: Initial Setup and Database

### 1.1 Project Initialization
```bash
# Create project structure
mkdir dynamic-mcp-agent
cd dynamic-mcp-agent
uv init
uv add fastapi uvicorn fast-agent-mcp supabase python-dotenv pydantic aiofiles python-multipart
uv add openai anthropic numpy scikit-learn  # For embeddings
```

### 1.2 Supabase Setup
1. Create a new Supabase project
2. Enable pgvector extension for semantic search:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

3. Create the database schema (execute in Supabase SQL editor):
```sql
-- Sessions table
CREATE TABLE sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),
  status VARCHAR(50) NOT NULL DEFAULT 'active',
  context JSONB DEFAULT '{}',
  required_credentials JSONB DEFAULT '[]',
  collected_credentials JSONB DEFAULT '{}'
);

-- Agents table
CREATE TABLE agents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
  created_at TIMESTAMP DEFAULT NOW(),
  mcp_config JSONB NOT NULL,
  workflow TEXT NOT NULL,
  status VARCHAR(50) NOT NULL DEFAULT 'active',
  metadata JSONB DEFAULT '{}'
);

-- Conversations table
CREATE TABLE conversations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
  agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,
  message TEXT NOT NULL,
  role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
  timestamp TIMESTAMP DEFAULT NOW(),
  metadata JSONB DEFAULT '{}'
);

-- MCP Server Cache table with vector support
CREATE TABLE mcp_server_cache (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  server_name VARCHAR(255) UNIQUE NOT NULL,
  description TEXT,
  tools JSONB,
  credential_requirements JSONB,
  embedding VECTOR(1536),
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_sessions_status ON sessions(status);
CREATE INDEX idx_agents_session ON agents(session_id);
CREATE INDEX idx_conversations_session ON conversations(session_id);
CREATE INDEX idx_mcp_embedding ON mcp_server_cache USING ivfflat (embedding vector_cosine_ops);
```

## Phase 2: Core Implementation

### 2.1 Environment Configuration
Create `.env` file:
```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
ENCRYPTION_KEY=your_32_char_encryption_key
```

### 2.2 Master Agent Implementation
Create `app/core/master_agent.py`:
```python
import asyncio
from mcp_agent.core.fastagent import FastAgent
from typing import Dict, List, Optional
import json

# Initialize FastAgent
fast = FastAgent("Dynamic MCP Agent System")

@fast.agent(
    name="workflow_analyzer",
    instruction="""You are an expert at analyzing user requests and determining:
    1. Which MCP servers are needed for the workflow
    2. What credentials are required
    3. The optimal workflow steps
    
    You have access to Supabase MCP to query server information.
    When analyzing requests:
    - Identify the main tasks and subtasks
    - Map each task to appropriate MCP servers
    - Determine the sequence of operations
    - Identify all required credentials
    
    Use the search_mcp_servers tool to find relevant servers.
    Use the get_credential_requirements tool to get credential info.
    """,
    servers=["supabase"],
    model="sonnet",
    human_input=True
)
async def workflow_analyzer():
    pass

# Tool for creating new agents
async def create_agent_tool(
    mcp_server_info: Dict,
    mcp_config: Dict,
    user_prompt: str,
    workflow: str
) -> Dict:
    """Tool that creates a new agent with specified MCP configuration"""
    # For now, return OK as requested
    return {
        "status": "ok",
        "agent_id": "mock_agent_id",
        "message": "Agent created successfully"
    }

# Register the tool with the master agent
fast.register_tool(create_agent_tool)
```

### 2.3 MCP Server Data Loader
Create `scripts/load_mcp_data.py`:
```python
import json
import asyncio
from supabase import create_client
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize clients
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def generate_embedding(text: str) -> List[float]:
    """Generate embedding for text using OpenAI"""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

async def load_mcp_servers():
    """Load MCP server data into Supabase with embeddings"""
    # Load server data
    with open('config/servers.json', 'r') as f:
        servers_data = json.load(f)
    
    with open('config/credinfo.json', 'r') as f:
        cred_data = json.load(f)
    
    for server in servers_data['servers']:
        # Create searchable text
        search_text = f"{server['name']} {server['description']} "
        search_text += " ".join([tool['name'] for tool in server.get('tools', [])])
        
        # Generate embedding
        embedding = await generate_embedding(search_text)
        
        # Get credential requirements
        cred_info = cred_data['credential_instructions'].get(
            server['name'].lower().replace(' ', '-'), 
            {}
        )
        
        # Insert into database
        data = {
            'server_name': server['name'],
            'description': server['description'],
            'tools': server.get('tools', []),
            'credential_requirements': cred_info,
            'embedding': embedding
        }
        
        supabase.table('mcp_server_cache').upsert(data).execute()
        print(f"Loaded: {server['name']}")

if __name__ == "__main__":
    asyncio.run(load_mcp_servers())
```

### 2.4 API Implementation
Create `app/main.py`:
```python
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid
from datetime import datetime

from app.api import chat, agents, sessions
from app.core.master_agent import workflow_analyzer
from app.services.supabase_service import SupabaseService

app = FastAPI(title="Dynamic MCP Agent API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api")
app.include_router(agents.router, prefix="/api")
app.include_router(sessions.router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Dynamic MCP Agent API", "version": "1.0.0"}

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    # Initialize master agent
    await workflow_analyzer.initialize()
```

Create `app/api/chat.py`:
```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid
from app.services.agent_service import AgentService
from app.services.session_service import SessionService

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    credentials: Optional[Dict[str, Dict[str, str]]] = None
    files: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    reply: str
    session_id: str
    required_credentials: Optional[List[Dict]] = None
    required_files: Optional[List[Dict]] = None
    status: str
    agent_created: bool
    agent_id: Optional[str] = None

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint for interacting with the master agent"""
    
    # Get or create session
    session_service = SessionService()
    if request.session_id:
        session = await session_service.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        session = await session_service.create_session()
    
    # Process with master agent
    agent_service = AgentService()
    response = await agent_service.process_message(
        session_id=session['id'],
        message=request.message,
        credentials=request.credentials,
        files=request.files
    )
    
    return ChatResponse(**response)
```

### 2.5 Service Layer Implementation
Create `app/services/agent_service.py`:
```python
from typing import Dict, List, Optional
import json
from app.core.master_agent import fast, workflow_analyzer
from app.services.supabase_service import SupabaseService
from app.services.mcp_selector import MCPSelector

class AgentService:
    def __init__(self):
        self.supabase = SupabaseService()
        self.mcp_selector = MCPSelector()
    
    async def process_message(
        self,
        session_id: str,
        message: str,
        credentials: Optional[Dict] = None,
        files: Optional[List[Dict]] = None
    ) -> Dict:
        """Process a message through the master agent"""
        
        # Get session context
        session = await self.supabase.get_session(session_id)
        
        # Update session with any new credentials
        if credentials:
            collected_creds = session.get('collected_credentials', {})
            collected_creds.update(credentials)
            await self.supabase.update_session(
                session_id, 
                {'collected_credentials': collected_creds}
            )
        
        # Analyze the message with master agent
        async with fast.run() as agent:
            # Send message to workflow analyzer
            analysis = await agent.workflow_analyzer(message)
            
            # Extract required MCP servers from analysis
            required_servers = await self.mcp_selector.extract_servers(analysis)
            
            # Check for missing credentials
            missing_creds = await self.check_missing_credentials(
                required_servers, 
                session.get('collected_credentials', {})
            )
            
            if missing_creds:
                # Return required credentials
                return {
                    "reply": analysis,
                    "session_id": session_id,
                    "required_credentials": missing_creds,
                    "status": "pending_credentials",
                    "agent_created": False,
                    "agent_id": None
                }
            
            # All credentials available, create agent
            agent_config = await self.build_agent_config(
                required_servers,
                session.get('collected_credentials', {}),
                analysis
            )
            
            # Create the agent
            agent_result = await agent.create_agent_tool(**agent_config)
            
            return {
                "reply": f"Agent created successfully! {analysis}",
                "session_id": session_id,
                "required_credentials": None,
                "status": "success",
                "agent_created": True,
                "agent_id": agent_result['agent_id']
            }
    
    async def check_missing_credentials(
        self, 
        required_servers: List[str], 
        collected_credentials: Dict
    ) -> List[Dict]:
        """Check which credentials are missing"""
        missing = []
        
        for server in required_servers:
            server_creds = await self.supabase.get_server_credentials(server)
            for cred in server_creds:
                if server not in collected_credentials or \
                   cred['key'] not in collected_credentials[server]:
                    missing.append({
                        "mcp_server": server,
                        "credential_key": cred['key'],
                        "description": cred['description'],
                        "instructions": cred.get('instructions', [])
                    })
        
        return missing
```

### 2.6 MCP Selector Service
Create `app/services/mcp_selector.py`:
```python
from typing import List, Dict
import numpy as np
from app.services.supabase_service import SupabaseService
from openai import OpenAI
import os

class MCPSelector:
    def __init__(self):
        self.supabase = SupabaseService()
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    async def search_servers(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for MCP servers using semantic search"""
        # Generate embedding for query
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = response.data[0].embedding
        
        # Search in Supabase using vector similarity
        results = self.supabase.client.rpc(
            'match_mcp_servers',
            {
                'query_embedding': query_embedding,
                'match_threshold': 0.7,
                'match_count': limit
            }
        ).execute()
        
        return results.data
    
    async def extract_servers(self, analysis: str) -> List[str]:
        """Extract required MCP servers from analysis"""
        # This would parse the analysis to extract server names
        # For now, return a mock implementation
        # In production, use NLP or structured output from the agent
        servers = []
        
        # Search for each potential task in the analysis
        keywords = self.extract_keywords(analysis)
        for keyword in keywords:
            matches = await self.search_servers(keyword, limit=3)
            servers.extend([m['server_name'] for m in matches])
        
        # Remove duplicates
        return list(set(servers))
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for searching"""
        # Simple implementation - in production use NLP
        common_tasks = [
            'email', 'gmail', 'calendar', 'drive', 'slack', 
            'notion', 'github', 'file', 'web', 'search'
        ]
        
        keywords = []
        text_lower = text.lower()
        for task in common_tasks:
            if task in text_lower:
                keywords.append(task)
        
        return keywords
```

## Phase 3: Testing and Deployment

### 3.1 Create Test Suite
Create `tests/test_api.py`:
```python
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Dynamic MCP Agent API"

def test_chat_endpoint():
    response = client.post("/api/chat", json={
        "message": "I want to monitor Gmail and send Slack notifications"
    })
    assert response.status_code == 200
    assert "session_id" in response.json()

def test_session_creation():
    response = client.post("/api/sessions/create")
    assert response.status_code == 200
    assert "id" in response.json()
```

### 3.2 Docker Configuration
Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY app/ app/
COPY config/ config/

# Install dependencies
RUN uv sync

# Expose port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3.3 Deployment Script
Create `deploy.sh`:
```bash
#!/bin/bash

# Build Docker image
docker build -t dynamic-mcp-agent .

# Run with environment variables
docker run -d \
  --name mcp-agent-api \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/config:/app/config \
  dynamic-mcp-agent
```

## Phase 4: Usage Examples

### 4.1 Example API Calls

**Initial Request:**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I want to create a workflow that monitors my Gmail for invoices, extracts the data, and saves it to a Google Sheet"
  }'
```

**Response with Required Credentials:**
```json
{
  "reply": "I'll help you create a workflow for monitoring Gmail invoices and saving to Google Sheets. I've identified that we'll need Gmail and Google Sheets access.",
  "session_id": "123e4567-e89b-12d3-a456-426614174000",
  "required_credentials": [
    {
      "mcp_server": "Gmail",
      "credential_key": "GMAIL_CREDENTIALS_PATH",
      "description": "Path to Gmail OAuth2 credentials file"
    },
    {
      "mcp_server": "Google Sheets",
      "credential_key": "SHEETS_CREDENTIALS_PATH",
      "description": "Path to Google Sheets OAuth2 credentials file"
    }
  ],
  "status": "pending_credentials",
  "agent_created": false
}
```

**Providing Credentials:**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Here are my credentials",
    "session_id": "123e4567-e89b-12d3-a456-426614174000",
    "credentials": {
      "Gmail": {
        "GMAIL_CREDENTIALS_PATH": "/path/to/gmail-creds.json"
      },
      "Google Sheets": {
        "SHEETS_CREDENTIALS_PATH": "/path/to/sheets-creds.json"
      }
    }
  }'
```

## Important Notes for TaskMaster

1. **Supabase MCP Integration**: Use the Supabase MCP server for all database operations within the master agent
2. **Error Handling**: Implement comprehensive error handling for credential validation and agent creation
3. **Security**: Never log or expose credentials in responses
4. **Scalability**: The vector search implementation should handle 4000+ servers efficiently
5. **Testing**: Test with various workflow scenarios to ensure proper MCP server selection

## Next Steps

1. Implement the Supabase vector search function
2. Create comprehensive MCP server selection logic
3. Build the actual agent creation mechanism
4. Add monitoring and logging
5. Implement rate limiting and authentication
6. Create documentation and examples 