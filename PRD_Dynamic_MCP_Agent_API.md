# Product Requirements Document: Dynamic MCP Agent API Backend

## 1. Executive Summary

### 1.1 Product Overview
The Dynamic MCP Agent API is a Python-based backend system that enables intelligent agent creation and management through natural language conversations. The system analyzes user requests, determines required MCP (Model Context Protocol) servers, collects necessary credentials, and dynamically creates specialized agents with appropriate workflows.

### 1.2 Key Features
- Natural language workflow description and agent creation
- Intelligent MCP server selection from 4000+ available servers
- Dynamic credential collection and management
- File upload support for credentials and configurations
- Automated workflow generation
- RESTful API for agent interaction

### 1.3 Technology Stack
- **Backend Framework**: Python (FastAPI recommended)
- **Database**: Supabase
- **Agent Framework**: fast-agent-mcp
- **MCP Integration**: Supabase MCP for database operations

## 2. Functional Requirements

### 2.1 Core Components

#### 2.1.1 Master Agent (Workflow Analyzer)
- **Purpose**: Analyze user requests and determine required MCP servers and workflows
- **Capabilities**:
  - Parse natural language workflow descriptions
  - Identify required MCP servers from 4000+ available options
  - Generate workflow steps
  - Determine credential requirements
  - Create agent configuration specifications

#### 2.1.2 Dynamic Agent Creator
- **Purpose**: Create new fast-agent instances with specific MCP configurations
- **Capabilities**:
  - Configure MCP servers with provided credentials
  - Set up agent workflows
  - Initialize agents with appropriate instructions

### 2.2 API Endpoints

#### 2.2.1 Chat Endpoint
```
POST /api/chat
```
**Request Body**:
```json
{
  "message": "string",
  "session_id": "string",
  "credentials": {
    "mcp_server_name": {
      "credential_key": "credential_value"
    }
  },
  "files": [
    {
      "filename": "string",
      "content": "base64_encoded_string",
      "type": "credential|config|other"
    }
  ]
}
```

**Response**:
```json
{
  "reply": "string",
  "session_id": "string",
  "required_credentials": [
    {
      "mcp_server": "string",
      "credential_key": "string",
      "description": "string",
      "instructions": ["string"]
    }
  ],
  "required_files": [
    {
      "key": "string",
      "description": "string",
      "file_type": "string"
    }
  ],
  "status": "success|pending_credentials|error",
  "agent_created": boolean,
  "agent_id": "string|null"
}
```

#### 2.2.2 Agent Creation Endpoint (Internal Tool Call)
```
POST /api/agents/create
```
**Request Body**:
```json
{
  "mcp_server_info": {
    "servers": [
      {
        "name": "string",
        "description": "string",
        "tools": [...]
      }
    ]
  },
  "mcp_config": {
    "mcpServers": {
      "server_name": {
        "command": "string",
        "args": ["string"],
        "env": {
          "KEY": "value"
        }
      }
    }
  },
  "user_prompt": "string",
  "workflow": "string"
}
```

**Response**:
```json
{
  "status": "ok",
  "agent_id": "string",
  "message": "Agent created successfully"
}
```

#### 2.2.3 Session Management
```
GET /api/sessions/{session_id}
POST /api/sessions/create
DELETE /api/sessions/{session_id}
```

### 2.3 Data Models

#### 2.3.1 Database Schema (Supabase)

**sessions**
```sql
CREATE TABLE sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),
  status VARCHAR(50) NOT NULL,
  context JSONB,
  required_credentials JSONB,
  collected_credentials JSONB
);
```

**agents**
```sql
CREATE TABLE agents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID REFERENCES sessions(id),
  created_at TIMESTAMP DEFAULT NOW(),
  mcp_config JSONB NOT NULL,
  workflow TEXT NOT NULL,
  status VARCHAR(50) NOT NULL,
  metadata JSONB
);
```

**conversations**
```sql
CREATE TABLE conversations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID REFERENCES sessions(id),
  agent_id UUID REFERENCES agents(id),
  message TEXT NOT NULL,
  role VARCHAR(20) NOT NULL,
  timestamp TIMESTAMP DEFAULT NOW(),
  metadata JSONB
);
```

**mcp_server_cache**
```sql
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
```

### 2.4 Workflow Processing

#### 2.4.1 User Interaction Flow
1. User sends initial request describing desired workflow
2. Master agent analyzes request and identifies required MCP servers
3. System checks for missing credentials
4. If credentials missing, return required credentials list
5. User provides credentials (and files if needed)
6. System validates credentials
7. Master agent creates workflow specification
8. Dynamic agent is created with configuration
9. Confirmation sent to user

#### 2.4.2 MCP Server Selection Algorithm
1. Parse user request for keywords and intent
2. Use semantic search on MCP server descriptions (vector embeddings)
3. Rank servers by relevance score
4. Cross-reference with workflow requirements
5. Validate server compatibility
6. Return selected servers with confidence scores

### 2.5 Security Requirements

#### 2.5.1 Credential Handling
- Encrypt credentials at rest using Supabase encryption
- Use secure transmission (HTTPS only)
- Implement credential validation before storage
- Support temporary credential storage with TTL
- Audit trail for credential access

#### 2.5.2 File Upload Security
- Validate file types and sizes
- Scan for malicious content
- Store files in secure Supabase storage
- Implement access controls

## 3. Non-Functional Requirements

### 3.1 Performance
- API response time < 2 seconds for standard requests
- Support 100 concurrent sessions
- MCP server search < 500ms for 4000+ servers
- Agent creation < 5 seconds

### 3.2 Scalability
- Horizontal scaling support
- Database connection pooling
- Caching for MCP server information
- Queue-based agent creation for heavy loads

### 3.3 Reliability
- 99.9% uptime target
- Graceful error handling
- Automatic retry for failed operations
- Session persistence across restarts

### 3.4 Monitoring
- API request logging
- Performance metrics
- Error tracking
- Agent creation success rates

## 4. Implementation Guidelines

### 4.1 Project Structure
```
dynamic-mcp-agent/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── chat.py
│   │   ├── agents.py
│   │   └── sessions.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── master_agent.py
│   │   ├── agent_creator.py
│   │   ├── mcp_selector.py
│   │   └── workflow_generator.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database.py
│   │   └── schemas.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── supabase_service.py
│   │   ├── credential_service.py
│   │   └── file_service.py
│   └── utils/
│       ├── __init__.py
│       ├── embeddings.py
│       └── security.py
├── config/
│   ├── config.json
│   ├── servers.json
│   └── credinfo.json
├── tests/
├── requirements.txt
└── README.md
```

### 4.2 Key Implementation Details

#### 4.2.1 Master Agent Configuration
```python
@fast.agent(
    name="workflow_analyzer",
    instruction="""You are an expert at analyzing user requests and determining:
    1. Which MCP servers are needed for the workflow
    2. What credentials are required
    3. The optimal workflow steps
    
    You have access to information about 4000+ MCP servers and their capabilities.
    Use the Supabase MCP to query and retrieve relevant server information.""",
    servers=["supabase"],
    model="sonnet",
    human_input=True
)
```

#### 4.2.2 MCP Server Information Management
- Pre-process and store all MCP server information in Supabase
- Create vector embeddings for server descriptions
- Implement semantic search using pgvector
- Cache frequently used servers

#### 4.2.3 Credential Collection Strategy
- Progressive credential collection (only ask for what's needed)
- Clear instructions from credinfo.json
- Support for OAuth flows where applicable
- Temporary storage with encryption

### 4.3 API Implementation Example

```python
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Optional

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str]
    credentials: Optional[Dict[str, Dict[str, str]]]
    files: Optional[List[Dict[str, str]]]

class ChatResponse(BaseModel):
    reply: str
    session_id: str
    required_credentials: Optional[List[Dict]]
    required_files: Optional[List[Dict]]
    status: str
    agent_created: bool
    agent_id: Optional[str]

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # Implementation here
    pass
```

## 5. Testing Requirements

### 5.1 Unit Tests
- Test MCP server selection algorithm
- Test credential validation
- Test workflow generation
- Test agent creation

### 5.2 Integration Tests
- End-to-end workflow creation
- Credential collection flow
- File upload handling
- Supabase operations

### 5.3 Performance Tests
- Load testing with concurrent users
- MCP server search performance
- Agent creation under load

## 6. Deployment Considerations

### 6.1 Environment Variables
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
ANTHROPIC_API_KEY=your_api_key
OPENAI_API_KEY=your_api_key
ENCRYPTION_KEY=your_encryption_key
```

### 6.2 Docker Configuration
- Containerize the application
- Include all MCP server configurations
- Volume mounts for credential storage

### 6.3 CI/CD Pipeline
- Automated testing
- Security scanning
- Deployment to cloud platform
- Database migrations

## 7. Future Enhancements

### 7.1 Phase 2 Features
- Multi-agent collaboration
- Workflow templates
- Visual workflow builder
- Real-time agent monitoring

### 7.2 Phase 3 Features
- Agent marketplace
- Custom MCP server integration
- Advanced analytics
- Enterprise features

## 8. Success Metrics

### 8.1 Key Performance Indicators
- Average time to create agent < 30 seconds
- Credential collection success rate > 95%
- User satisfaction score > 4.5/5
- API uptime > 99.9%

### 8.2 User Metrics
- Number of agents created per day
- Average workflow complexity
- Most used MCP servers
- Error rates by workflow type

## 9. Risk Mitigation

### 9.1 Technical Risks
- **Risk**: MCP server compatibility issues
  - **Mitigation**: Comprehensive testing suite, fallback mechanisms

- **Risk**: Credential security breach
  - **Mitigation**: Encryption, access controls, audit logging

- **Risk**: Performance degradation with scale
  - **Mitigation**: Caching, database optimization, horizontal scaling

### 9.2 Business Risks
- **Risk**: User adoption challenges
  - **Mitigation**: Clear documentation, examples, tutorials

- **Risk**: MCP server changes/deprecation
  - **Mitigation**: Version tracking, update notifications

## 10. Appendix

### 10.1 Example Workflow Creation

**User Request**: "I want to create a workflow that monitors my Gmail for specific emails, extracts data, stores it in Notion, and sends a Slack notification"

**System Response**:
1. Identifies required MCP servers: Gmail, Notion, Slack
2. Requests credentials for each service
3. Creates workflow with steps:
   - Monitor Gmail inbox
   - Extract email data
   - Format for Notion
   - Create Notion page
   - Send Slack notification
4. Creates configured agent with workflow

### 10.2 Glossary
- **MCP**: Model Context Protocol
- **fast-agent**: Python framework for creating AI agents
- **Supabase**: Open-source Firebase alternative
- **Workflow**: Sequence of steps executed by an agent 