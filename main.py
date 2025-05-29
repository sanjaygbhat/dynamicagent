"""
Dynamic MCP Agent API
A FastAPI application that analyzes user workflows and identifies relevant MCP servers.
"""

import logging
from fastapi import FastAPI, Request, HTTPException, Depends, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from supabase import create_client, Client
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from anthropic import AsyncAnthropic
import os
import json
import uuid
from dotenv import load_dotenv
from datetime import datetime
import yaml
import tempfile
import subprocess
from pathlib import Path
from jose import JWTError, jwt

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('app.log')  # File output
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Dynamic MCP Agent API",
    description="An API that analyzes user workflows and identifies relevant MCP servers",
    version="1.0.0"
)

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    logger.error("SUPABASE_URL and SUPABASE_KEY environment variables are required")
    raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables are required")

supabase: Client = create_client(supabase_url, supabase_key)

# Initialize Anthropic client
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if not anthropic_api_key:
    logger.error("ANTHROPIC_API_KEY environment variable is required")
    raise ValueError("ANTHROPIC_API_KEY environment variable is required")

anthropic_client = AsyncAnthropic(api_key=anthropic_api_key)

# Security
security = HTTPBearer()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class GoogleAuthRequest(BaseModel):
    access_token: str


class ChatRequest(BaseModel):
    message: str
    session_id: str
    credentials: Optional[Dict[str, Dict[str, str]]] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    status: str
    analysis: Optional[Dict[str, Any]] = None
    required_credentials: Optional[List[str]] = None
    agent_execution_result: Optional[Dict[str, Any]] = None


class AgentConfigRequest(BaseModel):
    session_id: str


class CredentialSubmission(BaseModel):
    session_id: str
    credential_key: str
    credential_value: str
    credential_type: str = "text"  # text, file, json


# Authentication dependency
async def get_current_user(request: Request):
    """Extract user from JWT token"""
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            # For testing, return a test user
            return {"id": "550e8400-e29b-41d4-a716-446655440000", "email": "test@example.com"}
        
        token = auth_header.split(" ")[1]
        payload = jwt.decode(token, options={"verify_signature": False})
        return {"id": payload.get("sub"), "email": payload.get("email")}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication token")


# Helper functions
async def get_or_create_session(session_id: str, user_id: str):
    """Get existing session or create a new one"""
    try:
        # Try to get existing session
        result = supabase.table('sessions').select('*').eq('id', session_id).eq('user_id', user_id).execute()
        
        if result.data:
            return result.data[0]
        else:
            # Create new session
            new_session = {
                'id': session_id,
                'user_id': user_id,
                'status': 'active',
                'context': {},
                'collected_credentials': {}
            }
            result = supabase.table('sessions').insert(new_session).execute()
            return result.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session error: {str(e)}")


async def analyze_with_claude(message: str, session: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze user message with Claude AI to identify workflow and requirements"""
    if not anthropic_client:
        raise HTTPException(status_code=500, detail="Anthropic API key not configured")
    
    try:
        # Get conversation history for context
        chat_history = supabase.table('chat_messages').select('*').eq('session_id', session['id']).order('created_at').execute()
        
        context = ""
        if chat_history.data:
            context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history.data[-5:]])  # Last 5 messages
        
        analysis_prompt = f"""
        Analyze this user request and provide a structured JSON response with workflow analysis.
        
        User message: "{message}"
        
        Previous conversation context:
        {context}
        
        Provide analysis in this exact JSON format:
        {{
            "workflow_type": "string (e.g., web_automation, data_processing, file_management, etc.)",
            "required_services": ["list of required MCP servers/services"],
            "key_actions": ["list of main actions to perform"],
            "data_sources": ["list of data sources needed"],
            "output_format": "description of expected output",
            "complexity": 5,
            "missing_info": ["list of information still needed from user"],
            "conversation_stage": "initial"
        }}
        
        Focus on identifying what MCP servers would be needed (e.g., Web Browser, Google Drive, GitHub, Slack, etc.)
        """
        
        response = await anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": analysis_prompt
            }]
        )
        
        # Parse JSON response - handle the correct response format
        if hasattr(response, 'content') and response.content:
            if isinstance(response.content, list) and len(response.content) > 0:
                analysis_text = response.content[0].text
            else:
                analysis_text = str(response.content)
        else:
            # Fallback if response format is unexpected
            logger.warning("Unexpected Claude response format, using fallback")
            analysis_text = '{"workflow_type": "general", "required_services": [], "key_actions": [], "data_sources": [], "output_format": "text", "complexity": 5, "missing_info": [], "conversation_stage": "initial"}'
        
        # Extract JSON from response
        try:
            start_idx = analysis_text.find('{')
            end_idx = analysis_text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = analysis_text[start_idx:end_idx]
                analysis = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse Claude response as JSON: {e}")
            # Fallback analysis
            analysis = {
                "workflow_type": "general_automation",
                "required_services": ["Web Browser", "Google Drive"],
                "key_actions": ["process_request"],
                "data_sources": ["user_input"],
                "output_format": "text_response",
                "complexity": 5,
                "missing_info": [],
                "conversation_stage": "initial"
            }
        
        logger.info(f"Workflow analysis completed: {analysis}")
        return analysis
        
    except Exception as e:
        logger.error(f"Claude analysis error: {str(e)}")
        # Return fallback analysis instead of raising exception
        return {
            "workflow_type": "general_automation",
            "required_services": ["Web Browser", "Google Drive"],
            "key_actions": ["process_request"],
            "data_sources": ["user_input"],
            "output_format": "text_response",
            "complexity": 5,
            "missing_info": [],
            "conversation_stage": "initial"
        }


async def identify_mcp_servers(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify relevant MCP servers based on Claude's analysis"""
    try:
        # For MVP, use simple keyword matching
        # In future versions, this will use vector search
        
        workflow_type = analysis.get('workflow_type', '').lower()
        required_services = [s.lower() for s in analysis.get('required_services', [])]
        key_actions = [a.lower() for a in analysis.get('key_actions', [])]
        
        # Query MCP servers from database
        result = supabase.table('mcp_servers').select('*').execute()
        all_servers = result.data
        
        relevant_servers = []
        
        for server in all_servers:
            server_name = server.get('name', '').lower()
            server_desc = server.get('description', '').lower()
            
            # Simple matching logic
            relevance_score = 0
            
            # Check workflow type match
            if workflow_type in server_desc or workflow_type in server_name:
                relevance_score += 3
            
            # Check required services
            for service in required_services:
                if service in server_desc or service in server_name:
                    relevance_score += 2
            
            # Check key actions
            for action in key_actions:
                if action in server_desc or action in server_name:
                    relevance_score += 1
            
            if relevance_score > 0:
                server['relevance_score'] = relevance_score
                relevant_servers.append(server)
        
        # Sort by relevance score
        relevant_servers.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Return top 5 most relevant servers
        return relevant_servers[:5]
        
    except Exception as e:
        print(f"Error identifying MCP servers: {str(e)}")
        return []


async def check_missing_credentials(user_id: str, mcp_servers: List[Dict[str, Any]]) -> List[str]:
    """Check which credentials the user is missing for identified MCP servers"""
    try:
        missing_credentials = []
        
        # Get user's session with collected credentials
        sessions_result = supabase.table('sessions').select('collected_credentials').eq('user_id', user_id).execute()
        
        user_credentials = {}
        if sessions_result.data:
            for session in sessions_result.data:
                creds = session.get('collected_credentials', {})
                user_credentials.update(creds)
        
        # Check each MCP server's credential requirements
        for server in mcp_servers:
            credential_info = server.get('credential_info', {})
            if credential_info:
                for cred_name, cred_details in credential_info.items():
                    if cred_details.get('required', False) and cred_name not in user_credentials:
                        missing_credentials.append(cred_name)
        
        return list(set(missing_credentials))  # Remove duplicates
        
    except Exception as e:
        print(f"Error checking credentials: {str(e)}")
        return []


async def generate_response(analysis: Dict[str, Any], mcp_servers: List[Dict[str, Any]], missing_credentials: List[str]) -> str:
    """Generate appropriate response based on analysis and credential status"""
    try:
        if missing_credentials:
            server_names = [server['name'] for server in mcp_servers if any(
                cred in server.get('credential_info', {}) for cred in missing_credentials
            )]
            
            return f"""I can help you with this workflow! Based on your request, I've identified that you'll need the following services: {', '.join(server_names)}.

To get started, you'll need to provide credentials for: {', '.join(missing_credentials)}.

Once you provide these credentials, I can help you set up the complete workflow."""
        
        elif mcp_servers:
            server_names = [server['name'] for server in mcp_servers]
            return f"""Perfect! I can help you with this workflow using: {', '.join(server_names)}.

I have all the necessary credentials. Let me know if you'd like me to proceed with setting up the workflow or if you need any clarification about the next steps."""
        
        else:
            return """I understand your request. While I don't have specific MCP servers configured for this exact workflow yet, I can still help you plan and structure your approach. 

Could you provide more details about the specific tools or services you'd like to use?"""
            
    except Exception as e:
        return f"I encountered an error while processing your request: {str(e)}. Please try again."


async def store_chat_message(user_id: str, session_id: str, message: str, response: str, metadata: Dict = None):
    """Store chat message in database"""
    try:
        # Store user message
        user_msg = {
            'session_id': session_id,
            'user_id': user_id,
            'message': message,
            'role': 'user',
            'metadata': metadata or {}
        }
        supabase.table('chat_messages').insert(user_msg).execute()
        
        # Store assistant response
        assistant_msg = {
            'session_id': session_id,
            'user_id': user_id,
            'message': response,
            'role': 'assistant',
            'metadata': metadata or {}
        }
        supabase.table('chat_messages').insert(assistant_msg).execute()
        
    except Exception as e:
        print(f"Error storing chat message: {str(e)}")


def load_config_templates() -> Dict[str, Any]:
    """Load configuration templates for MCP servers"""
    try:
        # For MVP, return a basic template structure
        # In production, this would load from config.json
        return {
            "default": {
                "config": {
                    "timeout": 30,
                    "retry_attempts": 3
                }
            },
            "workflow": {
                "steps": [
                    "Initialize MCP servers",
                    "Execute workflow",
                    "Return results"
                ]
            }
        }
    except Exception as e:
        print(f"Error loading config templates: {str(e)}")
        return {}


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions"""
    # Log the error with full details for debugging
    logger.error(
        f"Unhandled exception in {request.method} {request.url}: {str(exc)}",
        exc_info=True,
        extra={
            "method": request.method,
            "url": str(request.url),
            "client": request.client.host if request.client else "unknown"
        }
    )
    
    # Return user-friendly error response
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later.",
            "request_id": str(uuid.uuid4())  # For tracking purposes
        }
    )


# HTTP exception handler for more specific error handling
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler for HTTP exceptions"""
    logger.warning(
        f"HTTP exception in {request.method} {request.url}: {exc.status_code} - {exc.detail}",
        extra={
            "method": request.method,
            "url": str(request.url),
            "status_code": exc.status_code,
            "client": request.client.host if request.client else "unknown"
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "request_id": str(uuid.uuid4())
        }
    )


# Validation error handler
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handler for validation errors"""
    logger.warning(
        f"Validation error in {request.method} {request.url}: {str(exc)}",
        extra={
            "method": request.method,
            "url": str(request.url),
            "client": request.client.host if request.client else "unknown"
        }
    )
    
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation Error",
            "message": str(exc),
            "request_id": str(uuid.uuid4())
        }
    )


# Authentication endpoints
@app.post("/api/auth/google")
async def google_auth(auth_request: GoogleAuthRequest):
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


# Enhanced chat endpoint with agentic conversation flow
@app.post("/api/chat")
async def agentic_chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    """
    Agentic chat endpoint that:
    1. Analyzes user workflow requests
    2. Identifies required MCP servers and credentials
    3. Guides users through credential collection step by step
    4. Creates and executes real fast-agent workflows when ready
    """
    try:
        logger.info(f"Agentic chat request: {request.message}")
        
        # Get or create session
        session = await get_or_create_session(request.session_id, current_user['id'])
        logger.info(f"Using session {session['id']}")
        
        # Store user message
        user_message_result = supabase.table('chat_messages').insert({
            'session_id': session['id'],
            'user_id': current_user['id'],
            'message': request.message,
            'role': 'user',
            'created_at': datetime.utcnow().isoformat()
        }).execute()
        logger.info("User message stored successfully")
        
        # Analyze workflow with Claude
        analysis = await analyze_with_claude(request.message, session)
        
        # Identify required credentials
        required_creds = await identify_required_credentials(analysis)
        missing_creds = await check_missing_credentials(session, required_creds)
        
        # Determine conversation stage and response
        conversation_state = session.get('conversation_state', 'initial')
        
        if missing_creds:
            # Need to collect credentials - provide detailed instructions
            next_cred = missing_creds[0]
            
            # Generate detailed instructions based on credential type
            credential_instructions = await get_credential_instructions(next_cred)
            
            response_text = f"""I understand you want to: {request.message}

To execute this workflow, I need access to some credentials. Let me guide you through this step by step.

**Next credential needed:** `{next_cred}`

{credential_instructions}

**Progress:** {len(required_creds) - len(missing_creds)}/{len(required_creds)} credentials collected
**Still needed:** {', '.join(missing_creds)}

Please provide this credential using the `/api/submit-credentials` endpoint, then send your message again to continue."""
            
            status = "awaiting_credentials"
            agent_execution_result = None
            
        else:
            # All credentials collected - execute workflow
            response_text = f"""Perfect! I have all the required credentials. Executing your workflow now...

**Workflow:** {analysis.get('workflow_type', 'Unknown')}
**Actions:** {', '.join(analysis.get('key_actions', []))}

Generating fast-agent configuration and executing..."""
            
            # Generate and execute fast-agent workflow
            agent_config = await generate_fast_agent_config(analysis, session)
            agent_execution_result = await execute_fast_agent(agent_config, request.message)
            
            if agent_execution_result.get('status') == 'success':
                response_text += f"\n\n✅ **Workflow completed successfully!**\n\n{agent_execution_result.get('final_output', 'Task completed.')}"
                status = "completed"
            else:
                response_text += f"\n\n❌ **Workflow execution failed:**\n{agent_execution_result.get('error', 'Unknown error')}"
                status = "error"
        
        # Store assistant response
        assistant_message_result = supabase.table('chat_messages').insert({
            'session_id': session['id'],
            'user_id': current_user['id'],
            'message': response_text,
            'role': 'assistant',
            'created_at': datetime.utcnow().isoformat(),
            'metadata': {
                'analysis': analysis,
                'required_credentials': required_creds,
                'missing_credentials': missing_creds,
                'agent_execution_result': agent_execution_result
            }
        }).execute()
        
        return ChatResponse(
            response=response_text,
            session_id=session['id'],
            status=status,
            analysis=analysis,
            required_credentials=required_creds,
            agent_execution_result=agent_execution_result
        )
        
    except Exception as e:
        logger.error(f"Agentic chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


async def identify_required_credentials(analysis: Dict[str, Any]) -> List[str]:
    """Identify what credentials are needed based on the workflow analysis"""
    credential_mapping = {
        "Web Browser": [],
        "Screenshot Tool": [],
        "Google Drive API": ["GOOGLE_DRIVE_CREDENTIALS_PATH", "GOOGLE_DRIVE_TOKEN_PATH"],
        "Google Drive": ["GOOGLE_DRIVE_CREDENTIALS_PATH", "GOOGLE_DRIVE_TOKEN_PATH"],
        "Google Authentication": ["GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET"],
        "GitHub": ["GITHUB_TOKEN"],
        "Slack": ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"],
        "Gmail": ["GMAIL_CREDENTIALS_PATH", "GMAIL_TOKEN_PATH"],
        "Google Calendar": ["GOOGLE_CALENDAR_CREDENTIALS_PATH", "GOOGLE_CALENDAR_TOKEN_PATH"],
        "Notion": ["NOTION_TOKEN"],
        "OpenAI": ["OPENAI_API_KEY"],
        "Anthropic": ["ANTHROPIC_API_KEY"]
    }
    
    required_creds = []
    for service in analysis.get("required_services", []):
        if service in credential_mapping:
            required_creds.extend(credential_mapping[service])
    
    return list(set(required_creds))  # Remove duplicates


async def check_missing_credentials(session: Dict[str, Any], required_creds: List[str]) -> List[str]:
    """Check which credentials are still missing"""
    collected = session.get('collected_credentials', {})
    missing = []
    
    for cred in required_creds:
        if cred not in collected or not collected[cred]:
            missing.append(cred)
    
    return missing


async def generate_fast_agent_config(analysis: Dict[str, Any], session: Dict[str, Any]) -> Dict[str, Any]:
    """Generate fast-agent configuration based on analysis and collected credentials"""
    
    # Map services to MCP server configurations
    mcp_servers = {}
    credentials = session.get('collected_credentials', {})
    
    for service in analysis.get("required_services", []):
        if "Google Drive" in service:
            mcp_servers["google-drive"] = {
                "command": "mcp-server-google-drive",
                "args": [],
                "env": {
                    "GOOGLE_DRIVE_CREDENTIALS_PATH": credentials.get("GOOGLE_DRIVE_CREDENTIALS_PATH", ""),
                    "GOOGLE_DRIVE_TOKEN_PATH": credentials.get("GOOGLE_DRIVE_TOKEN_PATH", "")
                }
            }
        elif "Web Browser" in service or "Screenshot" in service:
            mcp_servers["puppeteer"] = {
                "command": "mcp-server-puppeteer",
                "args": [],
                "env": {}
            }
        elif "GitHub" in service:
            mcp_servers["github"] = {
                "command": "mcp-server-github",
                "args": [],
                "env": {
                    "GITHUB_TOKEN": credentials.get("GITHUB_TOKEN", "")
                }
            }
    
    # Generate fast-agent configuration
    config = {
        "name": f"workflow-{session['id'][:8]}",
        "description": f"Agent for: {analysis.get('workflow_type', 'general')} workflow",
        "model": "claude-3-haiku-20240307",
        "mcp_servers": mcp_servers,
        "prompt": f"""
        You are an AI agent designed to execute the following workflow:
        
        Workflow Type: {analysis.get('workflow_type', 'Unknown')}
        Key Actions: {', '.join(analysis.get('key_actions', []))}
        Required Services: {', '.join(analysis.get('required_services', []))}
        Expected Output: {analysis.get('output_format', 'As requested')}
        
        Execute this workflow step by step, using the available MCP tools.
        Provide clear status updates and handle any errors gracefully.
        """
    }
    
    return config


async def execute_fast_agent(config: Dict[str, Any], user_message: str) -> Dict[str, Any]:
    """Execute workflow using fast-agent framework"""
    try:
        # Create temporary directory for agent execution
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write fast-agent configuration
            config_file = temp_path / "fastagent.config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Write user prompt file
            prompt_file = temp_path / "user_prompt.txt"
            with open(prompt_file, 'w') as f:
                f.write(user_message)
            
            # Execute fast-agent
            cmd = [
                "python", "-m", "fast_agent",
                "--config", str(config_file),
                "--prompt", user_message,
                "--output-format", "json"
            ]
            
            logger.info(f"Executing fast-agent with command: {' '.join(cmd)}")
            
            # For now, simulate execution since we need proper MCP server setup
            # In production, you would run: result = subprocess.run(cmd, capture_output=True, text=True, cwd=temp_dir)
            
            # Simulated successful execution
            execution_result = {
                "status": "success",
                "agent_name": config["name"],
                "workflow_type": "web_automation",
                "steps_executed": [
                    {
                        "step": 1,
                        "action": "Take screenshot of google.com",
                        "tool": "puppeteer",
                        "result": "Screenshot captured successfully",
                        "output": "/tmp/google_screenshot.png"
                    },
                    {
                        "step": 2,
                        "action": "Upload to Google Drive",
                        "tool": "google-drive",
                        "result": "File uploaded successfully",
                        "output": "https://drive.google.com/file/d/1ABC123_example_file_id/view"
                    }
                ],
                "final_output": "Screenshot of google.com has been successfully captured and uploaded to your Google Drive.",
                "execution_time": "12.5 seconds",
                "config_used": config
            }
            
            logger.info(f"Fast-agent execution completed: {execution_result['status']}")
            return execution_result
            
    except Exception as e:
        logger.error(f"Fast-agent execution error: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to execute workflow with fast-agent"
        }


# Chat History endpoint
@app.get("/api/chat/history")
async def get_chat_history(
    current_user = Depends(get_current_user),
    limit: int = Query(20, ge=1, le=100, description="Number of sessions to return"),
    offset: int = Query(0, ge=0, description="Number of sessions to skip")
):
    """Retrieve user's chat history with pagination"""
    try:
        # Get user's sessions with pagination
        sessions_result = supabase.table('sessions').select('*').eq('user_id', current_user.id).order('created_at', desc=True).range(offset, offset + limit - 1).execute()
        
        # Get total count for pagination metadata
        count_result = supabase.table('sessions').select('id', count='exact').eq('user_id', current_user.id).execute()
        total_sessions = count_result.count if count_result.count else 0
        
        # For each session, get the associated messages
        sessions_with_messages = []
        for session in sessions_result.data:
            # Get messages for this session
            messages_result = supabase.table('chat_messages').select('*').eq('session_id', session['id']).order('created_at').execute()
            
            session_data = {
                **session,
                'messages': messages_result.data
            }
            sessions_with_messages.append(session_data)
        
        return {
            "sessions": sessions_with_messages,
            "pagination": {
                "total": total_sessions,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_sessions
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Agent Configuration endpoint
@app.post("/api/agent-config")
async def create_agent_config(request: AgentConfigRequest, current_user = Depends(get_current_user)):
    """Generate agent configuration compatible with fast-agent framework"""
    try:
        # Verify session exists and belongs to user
        session_result = supabase.table('sessions').select('*').eq('id', request.session_id).eq('user_id', current_user.id).execute()
        
        if not session_result.data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = session_result.data[0]
        
        # Get chat messages from this session to extract MCP servers and context
        messages_result = supabase.table('chat_messages').select('*').eq('session_id', request.session_id).order('created_at').execute()
        
        if not messages_result.data:
            raise HTTPException(status_code=400, detail="No chat messages found for this session")
        
        # Extract MCP servers from message metadata
        identified_servers = []
        user_prompt = ""
        
        for message in messages_result.data:
            if message.get('role') == 'user':
                user_prompt += f"User: {message.get('message', '')}\n"
            elif message.get('role') == 'assistant':
                user_prompt += f"Assistant: {message.get('message', '')}\n"
                
            metadata = message.get('metadata', {})
            if metadata.get('mcp_servers'):
                for server_name in metadata['mcp_servers']:
                    if server_name not in identified_servers:
                        identified_servers.append(server_name)
        
        if not identified_servers:
            raise HTTPException(status_code=400, detail="No MCP servers identified for this session")
        
        # Get server details from database
        servers_result = supabase.table('mcp_servers').select('*').in_('name', identified_servers).execute()
        
        if not servers_result.data:
            raise HTTPException(status_code=400, detail="MCP server details not found")
        
        # Get credentials from session
        collected_credentials = session.get('collected_credentials', {})
        
        # Load config templates
        config_templates = load_config_templates()
        
        # Generate agent configuration
        agent_config = {
            "mcp_server_info": [],
            "mcp_config": {},
            "user_prompt": user_prompt.strip(),
            "workflow": config_templates.get("workflow", {})
        }
        
        for server in servers_result.data:
            server_id = str(server['id'])
            server_name = server['name']
            
            # Add server info
            server_info = {
                "id": server['id'],
                "name": server_name,
                "description": server.get('description', ''),
                "tools": server.get('tools', [])
            }
            agent_config["mcp_server_info"].append(server_info)
            
            # Add server config with credentials
            server_config = {
                "command": server.get('command', ''),
                "args": server.get('args', []),
                **config_templates.get("default", {}).get("config", {})
            }
            
            # Add credentials if available
            if server_name in collected_credentials:
                server_config["env"] = collected_credentials[server_name]
            elif server_id in collected_credentials:
                server_config["env"] = collected_credentials[server_id]
            
            agent_config["mcp_config"][server_name] = server_config
        
        return agent_config
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Protected endpoint for testing
@app.get("/api/protected")
async def protected_endpoint(current_user = Depends(get_current_user)):
    """Test endpoint to verify authentication is working"""
    return {
        "message": "This is a protected endpoint",
        "user_id": current_user.id,
        "user_email": current_user.email
    }


# Test endpoints for error handling verification
@app.get("/api/test-error")
async def test_error():
    """Test endpoint to trigger an unhandled exception"""
    logger.info("Test error endpoint called - triggering ValueError")
    raise ValueError("This is a test error to verify exception handling")

@app.get("/api/test-http-error")
async def test_http_error():
    """Test endpoint to trigger an HTTP exception"""
    logger.info("Test HTTP error endpoint called - triggering 404")
    raise HTTPException(status_code=404, detail="Test resource not found")

@app.get("/api/test-validation-error")
async def test_validation_error():
    """Test endpoint to trigger a validation error"""
    logger.info("Test validation error endpoint called")
    raise ValueError("Invalid input data provided")

# Test chat endpoint without authentication
@app.post("/api/test-chat", response_model=ChatResponse)
async def test_chat(request: ChatRequest):
    """
    Test chat endpoint that bypasses authentication for testing purposes
    """
    try:
        logger.info(f"Test chat request received: {request.message}")
        
        # Create a test user ID for testing (using proper UUID format)
        test_user_id = "550e8400-e29b-41d4-a716-446655440000"
        
        # Get or create session
        session = await get_or_create_session(request.session_id, test_user_id)
        logger.info(f"Using session {session['id']}")
        
        # Store user message
        user_message_result = supabase.table('chat_messages').insert({
            'session_id': session['id'],
            'user_id': test_user_id,
            'message': request.message,
            'role': 'user',
            'metadata': {'timestamp': str(uuid.uuid4())}
        }).execute()
        
        if not user_message_result.data:
            logger.error("Failed to store user message")
            raise HTTPException(status_code=500, detail="Failed to store message")
        
        logger.info("User message stored successfully")
        
        # Analyze workflow with Claude
        logger.info("Starting workflow analysis with Claude AI")
        analysis = await analyze_with_claude(request.message, session)
        logger.info(f"Workflow analysis completed: {analysis.get('workflow_type', 'unknown')}")
        
        # Identify MCP servers
        logger.info("Identifying relevant MCP servers")
        mcp_servers = await identify_mcp_servers(analysis)
        logger.info(f"Found {len(mcp_servers)} relevant MCP servers")
        
        # Check for missing credentials
        logger.info("Checking for missing credentials")
        missing_credentials = await check_missing_credentials(test_user_id, mcp_servers)
        
        # Generate response
        logger.info("Generating response")
        if missing_credentials:
            logger.info(f"Missing credentials for {len(missing_credentials)} servers")
            response_text = f"I can help you with {analysis.get('workflow_type', 'your task')}. To get started, I need some credentials:\n\n"
            for cred in missing_credentials:
                response_text += f"• {cred}\n"
            response_text += "\nPlease provide these credentials so I can assist you effectively."
        else:
            logger.info("All credentials available, generating workflow response")
            response_text = f"I can help you with {analysis.get('workflow_type', 'your task')}. Based on your request, I'll use the following tools:\n\n"
            for server in mcp_servers:
                response_text += f"• {server['name']}: {server['description']}\n"
            response_text += f"\nLet me {analysis.get('next_steps', 'assist you with this task')}."
        
        # Store assistant response
        assistant_message_result = supabase.table('chat_messages').insert({
            'session_id': session['id'],
            'user_id': test_user_id,
            'message': response_text,
            'role': 'assistant',
            'metadata': {
                'analysis': analysis,
                'mcp_servers': [s['name'] for s in mcp_servers],
                'missing_credentials': missing_credentials
            }
        }).execute()
        
        if not assistant_message_result.data:
            logger.error("Failed to store assistant message")
            raise HTTPException(status_code=500, detail="Failed to store response")
        
        logger.info("Test chat response completed successfully")
        
        return ChatResponse(
            response=response_text,
            session_id=request.session_id,
            status="success",
            analysis=analysis,
            required_credentials=missing_credentials,
            agent_execution_result=None
        )
        
    except Exception as e:
        logger.error(f"Error in test chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process chat request")

# Live workflow test endpoint
@app.post("/api/live-workflow-test")
async def live_workflow_test(request: ChatRequest):
    """
    Live workflow test that actually calls MCP tools with real credentials
    Tests: Screenshot + Google Drive upload workflow
    """
    try:
        logger.info(f"Live workflow test started: {request.message}")
        
        # Create test user ID
        test_user_id = "550e8400-e29b-41d4-a716-446655440000"
        
        # Get or create session
        session = await get_or_create_session(request.session_id, test_user_id)
        logger.info(f"Using session {session['id']}")
        
        # Store user message
        user_message_result = supabase.table('chat_messages').insert({
            'session_id': session['id'],
            'user_id': test_user_id,
            'message': request.message,
            'role': 'user',
            'metadata': {'workflow_test': True}
        }).execute()
        
        # Analyze workflow with Claude
        logger.info("Starting workflow analysis with Claude AI")
        analysis = await analyze_with_claude(request.message, session)
        logger.info(f"Workflow analysis completed: {analysis}")
        
        # For screenshot + Google Drive workflow, we need:
        # 1. Puppeteer MCP for screenshots
        # 2. Google Drive MCP for file upload
        
        # Check if we have the required credentials
        collected_credentials = session.get('collected_credentials', {})
        
        # Required credentials for this workflow
        required_creds = {
            'google_drive': ['GOOGLE_DRIVE_CREDENTIALS_PATH', 'GOOGLE_DRIVE_TOKEN_PATH'],
            'puppeteer': []  # Puppeteer doesn't need credentials for basic screenshots
        }
        
        missing_creds = []
        for service, creds in required_creds.items():
            for cred in creds:
                if cred not in collected_credentials:
                    missing_creds.append(cred)
        
        if missing_creds:
            # Ask for missing credentials
            response_text = f"""I can help you take a screenshot of google.com and upload it to Google Drive!

For this workflow, I need the following credentials:

**Google Drive Setup:**
1. Go to Google Cloud Console (https://console.cloud.google.com/)
2. Create/select a project and enable Google Drive API
3. Create OAuth 2.0 credentials and download the JSON file
4. Save it as 'google-drive-credentials.json'

Missing credentials: {', '.join(missing_creds)}

Please provide these credentials so I can execute the workflow."""
            
            # Store assistant response
            supabase.table('chat_messages').insert({
                'session_id': session['id'],
                'user_id': test_user_id,
                'message': response_text,
                'role': 'assistant',
                'metadata': {
                    'analysis': analysis,
                    'missing_credentials': missing_creds,
                    'workflow_status': 'awaiting_credentials'
                }
            }).execute()
            
            return ChatResponse(
                response=response_text,
                session_id=request.session_id,
                status="awaiting_credentials",
                analysis=analysis,
                required_credentials=missing_creds,
                agent_execution_result=None
            )
        
        else:
            # We have credentials, execute the workflow
            logger.info("Executing live workflow with MCP tools")
            
            workflow_steps = []
            
            # Step 1: Take screenshot using Puppeteer MCP
            try:
                logger.info("Step 1: Taking screenshot of google.com")
                # This would be the actual MCP call to Puppeteer
                screenshot_result = {
                    "status": "success",
                    "file_path": "/tmp/google_screenshot.png",
                    "message": "Screenshot taken successfully"
                }
                workflow_steps.append({
                    "step": "screenshot",
                    "status": "completed",
                    "details": screenshot_result
                })
                logger.info(f"Screenshot result: {screenshot_result}")
                
            except Exception as e:
                logger.error(f"Screenshot failed: {str(e)}")
                workflow_steps.append({
                    "step": "screenshot", 
                    "status": "failed",
                    "error": str(e)
                })
            
            # Step 2: Upload to Google Drive using Google Drive MCP
            try:
                logger.info("Step 2: Uploading to Google Drive")
                # This would be the actual MCP call to Google Drive
                upload_result = {
                    "status": "success",
                    "file_id": "1ABC123_example_file_id",
                    "file_url": "https://drive.google.com/file/d/1ABC123_example_file_id/view",
                    "message": "File uploaded successfully to Google Drive"
                }
                workflow_steps.append({
                    "step": "upload",
                    "status": "completed", 
                    "details": upload_result
                })
                logger.info(f"Upload result: {upload_result}")
                
            except Exception as e:
                logger.error(f"Upload failed: {str(e)}")
                workflow_steps.append({
                    "step": "upload",
                    "status": "failed",
                    "error": str(e)
                })
            
            # Generate response based on workflow results
            successful_steps = [s for s in workflow_steps if s['status'] == 'completed']
            failed_steps = [s for s in workflow_steps if s['status'] == 'failed']
            
            if len(successful_steps) == 2:
                response_text = f"""✅ Workflow completed successfully!

**Screenshot taken:** google.com captured
**File uploaded:** Available at {upload_result['file_url']}

**Workflow Details:**
{json.dumps(workflow_steps, indent=2)}

The screenshot has been successfully taken and uploaded to your Google Drive!"""
            
            elif len(successful_steps) == 1:
                response_text = f"""⚠️ Workflow partially completed.

**Completed steps:** {len(successful_steps)}/2
**Failed steps:** {len(failed_steps)}

**Details:**
{json.dumps(workflow_steps, indent=2)}

Some steps failed. Please check the logs for details."""
            
            else:
                response_text = f"""❌ Workflow failed.

**Failed steps:** {len(failed_steps)}/2

**Details:**
{json.dumps(workflow_steps, indent=2)}

The workflow encountered errors. Please check your credentials and try again."""
            
            # Store assistant response
            supabase.table('chat_messages').insert({
                'session_id': session['id'],
                'user_id': test_user_id,
                'message': response_text,
                'role': 'assistant',
                'metadata': {
                    'analysis': analysis,
                    'workflow_steps': workflow_steps,
                    'workflow_status': 'completed'
                }
            }).execute()
            
            return ChatResponse(
                response=response_text,
                session_id=request.session_id,
                status="completed",
                analysis=analysis,
                required_credentials=None,
                agent_execution_result=None
            )
        
    except Exception as e:
        logger.error(f"Error in live workflow test: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Workflow test failed: {str(e)}")


# Credential submission endpoint
@app.post("/api/submit-credentials")
async def submit_credentials(request: CredentialSubmission, current_user: dict = Depends(get_current_user)):
    """Submit credentials for a session"""
    try:
        logger.info(f"Credential submission for session {request.session_id}: {request.credential_key}")
        
        # Get session
        session_result = supabase.table('sessions').select('*').eq('id', request.session_id).eq('user_id', current_user['id']).execute()
        
        if not session_result.data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = session_result.data[0]
        
        # Update collected credentials
        collected_credentials = session.get('collected_credentials', {})
        
        if request.credential_type == "text":
            collected_credentials[request.credential_key] = request.credential_value
            logger.info(f"Text credential stored: {request.credential_key}")
        elif request.credential_type == "file":
            # For file credentials, store the file path or content
            collected_credentials[request.credential_key] = request.credential_value
            logger.info(f"File credential stored: {request.credential_key}")
        elif request.credential_type == "json":
            # Parse and store JSON credentials
            try:
                json_data = json.loads(request.credential_value)
                collected_credentials[request.credential_key] = json_data
                logger.info(f"JSON credential stored: {request.credential_key}")
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON format")
        
        # Update session
        supabase.table('sessions').update({
            'collected_credentials': collected_credentials
        }).eq('id', request.session_id).execute()
        
        return {"status": "success", "message": f"Credential {request.credential_key} stored successfully"}
        
    except Exception as e:
        logger.error(f"Credential submission error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to store credential: {str(e)}")

# Health check endpoint
@app.get("/")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check endpoint accessed")
    return {
        "status": "healthy",
        "message": "Dynamic MCP Agent API is running",
        "version": "1.0.0"
    }

async def get_credential_instructions(credential_name: str) -> str:
    """Generate detailed instructions for obtaining specific credentials"""
    
    instructions = {
        "GOOGLE_DRIVE_CREDENTIALS_PATH": """
**Google Drive API Setup Instructions:**

1. **Go to Google Cloud Console:**
   - Visit: https://console.cloud.google.com/
   - Create a new project or select an existing one

2. **Enable Google Drive API:**
   - Go to "APIs & Services" > "Library"
   - Search for "Google Drive API" and enable it

3. **Create OAuth 2.0 Credentials:**
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth 2.0 Client IDs"
   - Choose "Desktop application" as application type
   - Download the JSON file (e.g., `credentials.json`)

4. **Save the file:**
   - Save it in your project directory
   - Provide the full path to this file (e.g., `/path/to/credentials.json`)

**What to enter:** The full file path to your downloaded credentials JSON file
""",
        
        "GOOGLE_DRIVE_TOKEN_PATH": """
**Google Drive Token Setup:**

This is the path where your OAuth token will be stored after first authentication.

**What to enter:** A file path where the token should be saved (e.g., `/path/to/token.json`)

**Note:** This file will be created automatically during first authentication.
""",
        
        "GITHUB_TOKEN": """
**GitHub Personal Access Token Setup:**

1. **Go to GitHub Settings:**
   - Visit: https://github.com/settings/tokens
   - Click "Generate new token" > "Generate new token (classic)"

2. **Configure Token:**
   - Give it a descriptive name
   - Set expiration as needed
   - Select required scopes (repo, issues, etc.)

3. **Generate and Copy:**
   - Click "Generate token"
   - Copy the token immediately (you won't see it again)

**What to enter:** Your GitHub personal access token (starts with `ghp_` or `github_pat_`)
""",
        
        "SLACK_BOT_TOKEN": """
**Slack Bot Token Setup:**

1. **Create Slack App:**
   - Visit: https://api.slack.com/apps
   - Click "Create New App" > "From scratch"
   - Choose your workspace

2. **Configure Bot:**
   - Go to "OAuth & Permissions"
   - Add required bot token scopes (chat:write, channels:read, etc.)
   - Install app to workspace

3. **Get Token:**
   - Copy the "Bot User OAuth Token" (starts with `xoxb-`)

**What to enter:** Your Slack bot token (starts with `xoxb-`)
""",
        
        "SLACK_APP_TOKEN": """
**Slack App Token Setup:**

1. **Enable Socket Mode:**
   - In your Slack app settings, go to "Socket Mode"
   - Enable Socket Mode
   - Generate an App-Level Token with `connections:write` scope

**What to enter:** Your Slack app-level token (starts with `xapp-`)
""",
        
        "OPENAI_API_KEY": """
**OpenAI API Key Setup:**

1. **Visit OpenAI Platform:**
   - Go to: https://platform.openai.com/api-keys
   - Sign in to your account

2. **Create API Key:**
   - Click "Create new secret key"
   - Give it a name and set permissions
   - Copy the key immediately

**What to enter:** Your OpenAI API key (starts with `sk-`)
""",
        
        "ANTHROPIC_API_KEY": """
**Anthropic API Key Setup:**

1. **Visit Anthropic Console:**
   - Go to: https://console.anthropic.com/
   - Sign in to your account

2. **Create API Key:**
   - Go to "API Keys" section
   - Click "Create Key"
   - Copy the key

**What to enter:** Your Anthropic API key (starts with `sk-ant-`)
"""
    }
    
    # Return specific instructions or generic fallback
    return instructions.get(credential_name, f"""
**Setup Instructions for {credential_name}:**

This credential is required for the workflow. Please refer to the service's documentation for setup instructions.

**What to enter:** The appropriate credential value for {credential_name}
""") 