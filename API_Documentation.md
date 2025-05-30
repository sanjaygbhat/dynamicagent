# Dynamic MCP Agent API Documentation

## Overview

The Dynamic MCP Agent API is a FastAPI-based backend system that enables intelligent creation and management of MCP (Model Context Protocol) agents through natural language conversations. The system analyzes user workflow requests, identifies required MCP servers, collects necessary credentials, and dynamically executes workflows using specialized agents.

**Base URL**: `http://localhost:8000` (development)  
**API Version**: `2.0.0`  
**Documentation**: Auto-generated at `/docs` (Swagger UI) and `/redoc`

## Features

- 🤖 **Agentic Conversation Flow**: Natural language workflow analysis and execution
- 🔌 **MCP Server Integration**: Support for 12+ MCP servers including Google Drive, GitHub, Slack, etc.
- 🔐 **Credential Management**: Secure collection and storage of API keys and OAuth tokens
- ⚡ **Fast Agent Execution**: Dynamic agent configuration and workflow execution
- 💾 **Session Management**: Persistent conversation sessions with context
- 📊 **Chat History**: Complete conversation tracking and retrieval

## Authentication

The API uses JWT-based authentication with Bearer tokens.

### Headers Required
```http
Authorization: Bearer <your-jwt-token>
Content-Type: application/json
```

### Authentication Flow
1. Obtain JWT token from your authentication provider (Google OAuth)
2. Include token in Authorization header for all API requests
3. For testing, the API accepts a test user when no valid token is provided

---

## Endpoints

### 1. Health Check

**GET /** 

Basic health check endpoint to verify API status.

#### Response
```json
{
  "status": "healthy",
  "message": "Dynamic MCP Agent API is running",
  "version": "2.0.0",
  "features": [
    "Agentic conversation flow",
    "MCP server integration", 
    "Credential management",
    "Fast-agent execution"
  ]
}
```

---

### 2. Chat Endpoint

**POST /api/chat**

Main endpoint for conversational workflow building and execution. Implements a multi-stage flow:
1. Greet user and understand requirements
2. Analyze workflow and identify required MCP servers
3. Collect necessary credentials
4. Execute workflow using configured agents

#### Request Body
```json
{
  "message": "string",
  "session_id": "string", 
  "credentials": {
    "credential_key": "credential_value"
  }
}
```

#### Request Parameters
- **message** (string, required): User's natural language input
- **session_id** (string, required): Unique session identifier (UUID)
- **credentials** (object, optional): Key-value pairs of credentials for MCP servers

#### Response
```json
{
  "response": "string",
  "session_id": "string",
  "status": "string",
  "workflow_identified": boolean,
  "required_servers": [
    {
      "id": "string",
      "name": "string", 
      "description": "string",
      "tools": ["string"]
    }
  ],
  "missing_credentials": [
    {
      "server_id": "string",
      "server_name": "string",
      "credentials": [
        {
          "key": "string",
          "description": "string",
          "type": "string"
        }
      ]
    }
  ],
  "agent_execution_result": {
    "status": "success|error",
    "message": "string",
    "details": {}
  }
}
```

#### Response Fields
- **response** (string): AI assistant's response message
- **session_id** (string): Session identifier
- **status** (string): Current conversation status
  - `greeting`: Initial greeting phase
  - `conversation`: General conversation, gathering requirements
  - `awaiting_credentials`: Waiting for user to provide credentials
  - `completed`: Workflow executed successfully  
  - `error`: Execution failed
- **workflow_identified** (boolean): Whether a valid workflow was identified
- **required_servers** (array): List of MCP servers needed for the workflow
- **missing_credentials** (array): Credentials still needed from user
- **agent_execution_result** (object): Results of workflow execution

#### Status Codes
- **200**: Success
- **400**: Invalid request data
- **401**: Unauthorized (invalid/missing JWT token)
- **500**: Internal server error

#### Example: Initial Greeting
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "",
    "session_id": "550e8400-e29b-41d4-a716-446655440000"
  }'
```

```json
{
  "response": "Hi, what can I help you build today?",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "greeting",
  "workflow_identified": false
}
```

#### Example: Workflow Request
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I want to search my Google Drive for documents containing project updates and create a summary",
    "session_id": "550e8400-e29b-41d4-a716-446655440000"
  }'
```

```json
{
  "response": "I can help you search Google Drive and create summaries! To get started, I'll need your Google Drive credentials...",
  "session_id": "550e8400-e29b-41d4-a716-446655440000", 
  "status": "awaiting_credentials",
  "workflow_identified": true,
  "required_servers": [
    {
      "id": "google-drive",
      "name": "Google Drive MCP",
      "description": "Official Google Drive MCP server...",
      "tools": ["search", "gdrive_read_file"]
    }
  ],
  "missing_credentials": [
    {
      "server_id": "google-drive",
      "server_name": "Google Drive MCP", 
      "credentials": [
        {
          "key": "GDRIVE_OAUTH_PATH",
          "description": "Path to gcp-oauth.keys.json file",
          "type": "file_path"
        }
      ]
    }
  ]
}
```

---

### 3. Submit Credentials

**POST /api/submit-credentials**

Submit credentials required for MCP server authentication.

#### Request Body
```json
{
  "session_id": "string",
  "credential_key": "string", 
  "credential_value": "string",
  "credential_type": "text"
}
```

#### Request Parameters
- **session_id** (string, required): Session identifier
- **credential_key** (string, required): The credential key (e.g., "GITHUB_PERSONAL_ACCESS_TOKEN")
- **credential_value** (string, required): The credential value
- **credential_type** (string, optional): Type of credential ("text", "file", "json")

#### Response
```json
{
  "status": "success",
  "message": "string",
  "remaining_credentials": number
}
```

#### Example
```bash
curl -X POST "http://localhost:8000/api/submit-credentials" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "credential_key": "GITHUB_PERSONAL_ACCESS_TOKEN",
    "credential_value": "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "credential_type": "text"
  }'
```

---

### 4. Chat History

**GET /api/chat/history**

Retrieve conversation history for sessions.

#### Query Parameters
- **session_id** (string, optional): Specific session ID to retrieve
- **limit** (integer, optional): Number of messages to return (1-100, default: 20)
- **offset** (integer, optional): Number of messages to skip (default: 0)

#### Response (for specific session)
```json
{
  "session_id": "string",
  "messages": [
    {
      "id": "string",
      "session_id": "string",
      "user_id": "string", 
      "message": "string",
      "role": "user|assistant",
      "created_at": "timestamp",
      "metadata": {}
    }
  ],
  "pagination": {
    "limit": number,
    "offset": number,
    "total": number
  }
}
```

#### Response (for all sessions)
```json
{
  "sessions": [
    {
      "id": "string",
      "user_id": "string",
      "status": "string",
      "context": {},
      "created_at": "timestamp"
    }
  ],
  "total_sessions": number
}
```

#### Example
```bash
curl -X GET "http://localhost:8000/api/chat/history?session_id=550e8400-e29b-41d4-a716-446655440000&limit=10" \
  -H "Authorization: Bearer <token>"
```

---

## Data Models

### ChatRequest
```typescript
interface ChatRequest {
  message: string;           // User's message
  session_id: string;        // Session UUID  
  credentials?: {            // Optional credentials
    [key: string]: string;
  };
}
```

### ChatResponse
```typescript
interface ChatResponse {
  response: string;                    // AI response
  session_id: string;                  // Session UUID
  status: string;                      // Conversation status
  workflow_identified?: boolean;       // Whether workflow was identified
  required_servers?: Server[];         // Required MCP servers
  missing_credentials?: Credential[];  // Missing credentials
  agent_execution_result?: {           // Execution results
    status: "success" | "error";
    message: string;
    details?: any;
  };
}
```

### CredentialSubmission
```typescript
interface CredentialSubmission {
  session_id: string;        // Session UUID
  credential_key: string;    // Credential identifier
  credential_value: string;  // Credential value
  credential_type: string;   // Type: "text" | "file" | "json"
}
```

---

## Supported MCP Servers

The API supports the following MCP servers:

| Server ID | Name | Type | Description |
|-----------|------|------|-------------|
| `supabase` | Supabase MCP | Database | Database operations, auth, real-time |
| `context7` | Context7 MCP | Documentation | Library documentation access |
| `github` | GitHub MCP | Version Control | Repository management, issues, PRs |
| `filesystem` | Filesystem MCP | File Management | Secure file system operations |
| `brave-search` | Brave Search MCP | Search | Web search capabilities |
| `slack` | Slack MCP | Communication | Team communication tools |
| `gmail` | Gmail MCP | Email | Email management and search |
| `google-drive` | Google Drive MCP | Cloud Storage | File access and search |
| `google-calendar` | Google Calendar MCP | Calendar | Schedule management |
| `notion` | Notion MCP | Documentation | Workspace content management |
| `puppeteer` | Puppeteer MCP | Automation | Browser automation |
| `memory` | Memory MCP | Persistence | Knowledge graph memory |

---

## Credential Requirements

Each MCP server has specific credential requirements:

### Google Drive MCP
```json
{
  "GDRIVE_OAUTH_PATH": "/path/to/gcp-oauth.keys.json",
  "GDRIVE_CREDENTIALS_PATH": "/path/to/.gdrive-server-credentials.json"
}
```

### GitHub MCP  
```json
{
  "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
}
```

### Slack MCP
```json
{
  "SLACK_BOT_TOKEN": "xoxb-xxxxxxxxxxxx-xxxxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxx",
  "SLACK_TEAM_ID": "T1234567890"
}
```

### Supabase MCP
```json
{
  "SUPABASE_URL": "https://your-project.supabase.co", 
  "SUPABASE_ANON_KEY": "your-anon-key"
}
```

For complete credential setup instructions, see the `credinfo.json` file.

---

## Error Handling

### Error Response Format
```json
{
  "error": "string",
  "message": "string", 
  "status_code": number,
  "request_id": "string"
}
```

### Common Error Codes
- **400 Bad Request**: Invalid request data or missing required fields
- **401 Unauthorized**: Invalid or missing authentication token
- **404 Not Found**: Session or resource not found
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Unexpected server error

### Example Error Response
```json
{
  "error": "Validation Error",
  "message": "message field is required",
  "status_code": 400,
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

## Rate Limiting

- **Rate Limit**: 100 requests per minute per user
- **Headers**: Rate limit information included in response headers
  - `X-RateLimit-Limit`: Requests allowed per window
  - `X-RateLimit-Remaining`: Requests remaining in current window  
  - `X-RateLimit-Reset`: Time when rate limit resets

---

## CORS Configuration

The API is configured with permissive CORS settings for development:
- **Allowed Origins**: All (`*`)
- **Allowed Methods**: All HTTP methods
- **Allowed Headers**: All headers
- **Credentials**: Supported

For production, configure CORS appropriately for your domain.

---

## Database Schema

### Sessions Table
```sql
CREATE TABLE sessions (
  id UUID PRIMARY KEY,
  user_id VARCHAR(255) NOT NULL,
  status VARCHAR(50) NOT NULL,
  context JSONB,
  collected_credentials JSONB,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);
```

### Chat Messages Table  
```sql
CREATE TABLE chat_messages (
  id UUID PRIMARY KEY,
  session_id UUID REFERENCES sessions(id),
  user_id VARCHAR(255) NOT NULL,
  message TEXT NOT NULL,
  role VARCHAR(20) NOT NULL,
  metadata JSONB,
  created_at TIMESTAMP DEFAULT NOW()
);
```

---

## Development

### Environment Variables
```bash
# Required
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key
ANTHROPIC_API_KEY=your-anthropic-api-key

# Optional
LOG_LEVEL=INFO
FAST_AGENT_PATH=/path/to/fast-agent
```

### Running the API
```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Access documentation
open http://localhost:8000/docs
```

### Testing
```bash
# Run tests
python -m pytest

# Test specific endpoint
python test_live_workflow.py
```

---

## Examples

### Complete Workflow Example

1. **Start Session**
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "",
    "session_id": "550e8400-e29b-41d4-a716-446655440000"
  }'
```

2. **Request Workflow**
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a GitHub issue for a new feature request",
    "session_id": "550e8400-e29b-41d4-a716-446655440000"
  }'
```

3. **Submit Credentials**
```bash
curl -X POST "http://localhost:8000/api/submit-credentials" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "credential_key": "GITHUB_PERSONAL_ACCESS_TOKEN",
    "credential_value": "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
  }'
```

4. **Execute Workflow**
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a GitHub issue titled \"Add dark mode support\" with description \"Users have requested dark mode functionality for better accessibility\"",
    "session_id": "550e8400-e29b-41d4-a716-446655440000"
  }'
```

---

## Support

For issues and questions:
- Check the `/docs` endpoint for interactive API documentation
- Review error messages and logs in `app.log`
- Ensure all required environment variables are set
- Verify MCP server credentials are correctly configured

---

*Last Updated: January 2025*  
*API Version: 2.0.0* 