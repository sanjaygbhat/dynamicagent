# MVP Dynamic MCP Agent API Specification

## Overview
Build a Python FastAPI backend that analyzes user workflow requests, identifies required MCP servers, collects credentials, and creates configured agents using fast-agent framework.

## Core Requirements

### 1. Database Schema (Supabase)
```sql
-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Users table (managed by Supabase Auth)
-- Supabase automatically creates auth.users table

-- Sessions table with user association
CREATE TABLE sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  created_at TIMESTAMP DEFAULT NOW(),
  status VARCHAR(50) DEFAULT 'active',
  context JSONB DEFAULT '{}',
  collected_credentials JSONB DEFAULT '{}'
);

-- Chat messages table for history
CREATE TABLE chat_messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  message TEXT NOT NULL,
  role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE mcp_servers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name VARCHAR(255) UNIQUE NOT NULL,
  description TEXT,
  tools JSONB,
  config_template JSONB,
  credential_info JSONB,
  embedding VECTOR(1536)
);

-- Indexes
CREATE INDEX idx_sessions_user ON sessions(user_id);
CREATE INDEX idx_messages_session ON chat_messages(session_id);
CREATE INDEX idx_messages_user ON chat_messages(user_id);
CREATE INDEX idx_mcp_embedding ON mcp_servers USING ivfflat (embedding vector_cosine_ops);

-- Vector search function
CREATE OR REPLACE FUNCTION search_mcp_servers(
  query_embedding vector(1536),
  threshold float DEFAULT 0.7,
  max_results int DEFAULT 10
)
RETURNS TABLE (
  name varchar,
  description text,
  tools jsonb,
  config_template jsonb,
  credential_info jsonb,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    m.name,
    m.description,
    m.tools,
    m.config_template,
    m.credential_info,
    1 - (m.embedding <=> query_embedding) as similarity
  FROM mcp_servers m
  WHERE 1 - (m.embedding <=> query_embedding) > threshold
  ORDER BY m.embedding <=> query_embedding
  LIMIT max_results;
END;
$$;

-- RLS Policies
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;

-- Users can only see their own sessions
CREATE POLICY "Users can view own sessions" ON sessions
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create own sessions" ON sessions
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own sessions" ON sessions
  FOR UPDATE USING (auth.uid() = user_id);

-- Users can only see their own messages
CREATE POLICY "Users can view own messages" ON chat_messages
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create own messages" ON chat_messages
  FOR INSERT WITH CHECK (auth.uid() = user_id);
```

### 2. Single File Implementation
```python
# main.py - Complete MVP Implementation with Auth
import os
import json
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from supabase import create_client, Client
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# Initialize clients
app = FastAPI(title="Dynamic MCP Agent API")
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Security
security = HTTPBearer()

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class AuthRequest(BaseModel):
    access_token: str

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    credentials: Optional[Dict[str, Dict[str, str]]] = None

class ChatResponse(BaseModel):
    reply: str
    session_id: str
    required_credentials: Optional[List[Dict]] = None
    status: str
    agent_created: bool
    agent_id: Optional[str] = None

class ChatHistoryResponse(BaseModel):
    sessions: List[Dict]
    total_count: int

# Auth dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return user"""
    try:
        # Verify token with Supabase
        user = supabase.auth.get_user(credentials.credentials)
        if not user or not user.user:
            raise HTTPException(status_code=401, detail="Invalid authentication")
        return user.user
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication")

# Core Functions
async def save_message(session_id: str, user_id: str, message: str, role: str, metadata: Dict = {}):
    """Save chat message to database"""
    data = {
        'session_id': session_id,
        'user_id': user_id,
        'message': message,
        'role': role,
        'metadata': metadata
    }
    supabase.table('chat_messages').insert(data).execute()

async def analyze_workflow(message: str, session_context: Dict) -> Dict:
    """Use Claude to analyze workflow and identify MCP servers"""
    
    # Get MCP server list for context
    servers = supabase.table('mcp_servers').select('name, description').execute()
    server_list = "\n".join([f"- {s['name']}: {s['description']}" for s in servers.data[:50]])
    
    prompt = f"""Analyze this workflow request and identify required MCP servers.

Available MCP Servers:
{server_list}

User Request: {message}

Previous Context: {json.dumps(session_context.get('context', {}))}

Respond in JSON format:
{{
    "analysis": "Brief analysis of the workflow",
    "required_servers": ["server1", "server2"],
    "workflow_steps": ["step1", "step2"],
    "response_to_user": "Natural language response to the user"
}}"""

    response = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    try:
        return json.loads(response.content[0].text)
    except:
        return {
            "analysis": response.content[0].text,
            "required_servers": [],
            "workflow_steps": [],
            "response_to_user": response.content[0].text
        }

async def get_missing_credentials(required_servers: List[str], collected_creds: Dict) -> List[Dict]:
    """Check which credentials are missing"""
    missing = []
    
    for server_name in required_servers:
        # Get server info from database
        result = supabase.table('mcp_servers').select('credential_info').eq('name', server_name).execute()
        if not result.data:
            continue
            
        cred_info = result.data[0].get('credential_info', {})
        
        # Check each required credential
        for cred_key, cred_desc in cred_info.get('required_tokens', {}).items():
            if server_name not in collected_creds or cred_key not in collected_creds[server_name]:
                missing.append({
                    "mcp_server": server_name,
                    "credential_key": cred_key,
                    "description": cred_desc,
                    "instructions": cred_info.get('steps', [])[:3]  # First 3 steps
                })
    
    return missing

async def create_agent_config(servers: List[str], credentials: Dict, workflow: Dict) -> Dict:
    """Create agent configuration"""
    mcp_config = {"mcpServers": {}}
    
    for server_name in servers:
        # Get server config template
        result = supabase.table('mcp_servers').select('config_template').eq('name', server_name).execute()
        if not result.data:
            continue
            
        config = result.data[0]['config_template']
        
        # Add credentials if available
        if server_name in credentials:
            if 'env' not in config:
                config['env'] = {}
            config['env'].update(credentials[server_name])
        
        # Clean server name for config key
        config_key = server_name.lower().replace(' ', '-')
        mcp_config['mcpServers'][config_key] = config
    
    return {
        "mcp_server_info": {
            "servers": [{"name": s} for s in servers]
        },
        "mcp_config": mcp_config,
        "user_prompt": workflow.get('analysis', ''),
        "workflow": "\n".join(workflow.get('workflow_steps', []))
    }

# API Endpoints
@app.post("/api/auth/google")
async def google_auth(auth_request: AuthRequest):
    """Exchange Google OAuth token for Supabase session"""
    try:
        # Sign in with Google OAuth token
        response = supabase.auth.sign_in_with_id_token({
            'provider': 'google',
            'token': auth_request.access_token
        })
        
        return {
            "access_token": response.session.access_token,
            "refresh_token": response.session.refresh_token,
            "user": {
                "id": response.user.id,
                "email": response.user.email
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, current_user = Depends(get_current_user)):
    """Main chat endpoint"""
    
    # Get or create session
    if request.session_id:
        session_result = supabase.table('sessions').select('*').eq('id', request.session_id).eq('user_id', current_user.id).execute()
        if not session_result.data:
            raise HTTPException(status_code=404, detail="Session not found")
        session = session_result.data[0]
    else:
        session_data = {
            'id': str(uuid.uuid4()),
            'user_id': current_user.id,
            'status': 'active',
            'context': {},
            'collected_credentials': {}
        }
        session_result = supabase.table('sessions').insert(session_data).execute()
        session = session_result.data[0]
    
    # Save user message
    await save_message(session['id'], current_user.id, request.message, 'user')
    
    # Update credentials if provided
    if request.credentials:
        collected_creds = session.get('collected_credentials', {})
        collected_creds.update(request.credentials)
        supabase.table('sessions').update({
            'collected_credentials': collected_creds
        }).eq('id', session['id']).execute()
        session['collected_credentials'] = collected_creds
    
    # Analyze workflow
    analysis = await analyze_workflow(request.message, session)
    
    # Update session context
    context = session.get('context', {})
    context['last_analysis'] = analysis
    supabase.table('sessions').update({'context': context}).eq('id', session['id']).execute()
    
    # Check for missing credentials
    required_servers = analysis.get('required_servers', [])
    missing_creds = await get_missing_credentials(
        required_servers, 
        session.get('collected_credentials', {})
    )
    
    response_text = analysis['response_to_user']
    
    if missing_creds:
        response_text += "\n\nI need some credentials to proceed."
        status = "pending_credentials"
        agent_created = False
        agent_id = None
    elif required_servers:
        agent_config = await create_agent_config(
            required_servers,
            session.get('collected_credentials', {}),
            analysis
        )
        response_text = f"{response_text}\n\nAgent created successfully with {len(required_servers)} MCP servers!"
        status = "success"
        agent_created = True
        agent_id = str(uuid.uuid4())
    else:
        status = "analyzing"
        agent_created = False
        agent_id = None
    
    # Save assistant response
    await save_message(session['id'], current_user.id, response_text, 'assistant', {
        'status': status,
        'agent_created': agent_created,
        'agent_id': agent_id
    })
    
    return ChatResponse(
        reply=response_text,
        session_id=session['id'],
        required_credentials=missing_creds if missing_creds else None,
        status=status,
        agent_created=agent_created,
        agent_id=agent_id
    )

@app.get("/api/chat/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    limit: int = 20,
    offset: int = 0,
    session_id: Optional[str] = None,
    current_user = Depends(get_current_user)
):
    """Get user's chat history"""
    
    # Build query
    query = supabase.table('chat_messages').select('*, sessions(*)').eq('user_id', current_user.id)
    
    if session_id:
        query = query.eq('session_id', session_id)
    
    # Get total count
    count_result = query.count().execute()
    total_count = count_result.count
    
    # Get paginated results
    result = query.order('created_at', desc=True).range(offset, offset + limit - 1).execute()
    
    # Group by session
    sessions_dict = {}
    for msg in result.data:
        sid = msg['session_id']
        if sid not in sessions_dict:
            sessions_dict[sid] = {
                'session_id': sid,
                'created_at': msg['sessions']['created_at'],
                'messages': []
            }
        sessions_dict[sid]['messages'].append({
            'id': msg['id'],
            'message': msg['message'],
            'role': msg['role'],
            'created_at': msg['created_at'],
            'metadata': msg.get('metadata', {})
        })
    
    # Convert to list and sort by most recent
    sessions = list(sessions_dict.values())
    sessions.sort(key=lambda x: x['created_at'], reverse=True)
    
    return ChatHistoryResponse(
        sessions=sessions,
        total_count=total_count
    )

@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, current_user = Depends(get_current_user)):
    """Get all messages for a specific session"""
    
    # Verify session belongs to user
    session_result = supabase.table('sessions').select('*').eq('id', session_id).eq('user_id', current_user.id).execute()
    if not session_result.data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get messages
    messages = supabase.table('chat_messages').select('*').eq('session_id', session_id).order('created_at').execute()
    
    return {
        'session': session_result.data[0],
        'messages': messages.data
    }

@app.get("/")
async def root():
    return {"message": "Dynamic MCP Agent API", "version": "MVP with Auth"}

# Data loader script (run separately)
async def load_mcp_data():
    """Load MCP server data from JSON files"""
    with open('config/servers.json', 'r') as f:
        servers_data = json.load(f)
    
    with open('config/config.json', 'r') as f:
        config_data = json.load(f)
    
    with open('config/credinfo.json', 'r') as f:
        cred_data = json.load(f)
    
    for server in servers_data['servers']:
        # Generate embedding
        search_text = f"{server['name']} {server['description']}"
        embedding_response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=search_text
        )
        
        # Get config template
        config_key = server['name'].lower().replace(' ', '-')
        config_template = config_data.get('mcpServers', {}).get(config_key, {})
        
        # Get credential info
        cred_info = cred_data.get('credential_instructions', {}).get(config_key, {})
        
        # Upsert to database
        data = {
            'name': server['name'],
            'description': server['description'],
            'tools': server.get('tools', []),
            'config_template': config_template,
            'credential_info': cred_info,
            'embedding': embedding_response.data[0].embedding
        }
        
        supabase.table('mcp_servers').upsert(data).execute()
        print(f"Loaded: {server['name']}")

if __name__ == "__main__":
    import uvicorn
    # Uncomment to load data
    # asyncio.run(load_mcp_data())
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 3. Supabase Setup

#### 3.1 Enable Google OAuth in Supabase Dashboard
1. Go to Authentication > Providers in Supabase Dashboard
2. Enable Google provider
3. Add your Google OAuth credentials:
   - Client ID
   - Client Secret
4. Add redirect URLs for your frontend

#### 3.2 Frontend Integration Example
```javascript
// Example using Supabase JS client
import { createClient } from '@supabase/supabase-js'

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY)

// Google Sign In
async function signInWithGoogle() {
  const { data, error } = await supabase.auth.signInWithOAuth({
    provider: 'google',
    options: {
      redirectTo: 'http://localhost:3000/auth/callback'
    }
  })
}

// Use the access token with API
async function callAPI(message) {
  const { data: { session } } = await supabase.auth.getSession()
  
  const response = await fetch('http://localhost:8000/api/chat', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${session.access_token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ message })
  })
  
  return response.json()
}
```

### 4. Environment Variables
```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
```

### 5. API Usage

**Google Authentication:**
```bash
# Exchange Google token for Supabase session
curl -X POST http://localhost:8000/api/auth/google \
  -H "Content-Type: application/json" \
  -d '{"access_token": "google-oauth-token"}'
```

**Authenticated Chat Request:**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Authorization: Bearer your-supabase-access-token" \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to monitor Gmail for invoices and save to Notion"}'
```

**Get Chat History:**
```bash
curl -X GET "http://localhost:8000/api/chat/history?limit=20&offset=0" \
  -H "Authorization: Bearer your-supabase-access-token"
```

**Get Session Messages:**
```bash
curl -X GET "http://localhost:8000/api/sessions/{session_id}/messages" \
  -H "Authorization: Bearer your-supabase-access-token"
```

## Key Updates for Auth & History

1. **Google OAuth Integration** - Uses Supabase Auth with Google provider
2. **User Association** - All sessions and messages linked to authenticated users
3. **Row Level Security** - Users can only access their own data
4. **Chat History API** - Retrieve previous conversations with pagination
5. **Session Messages** - Get all messages for a specific session
6. **Message Storage** - All chat messages saved for history

## Notes for TaskMaster

- Enable Google OAuth in Supabase Dashboard before testing
- RLS policies ensure data isolation between users
- Frontend needs to handle OAuth flow and token management
- Consider adding refresh token handling for production
- Add rate limiting per user for production deployment 