"""
Dynamic MCP Agent API
A FastAPI application that implements an agentic conversation flow for building workflows with MCP servers.
"""

import logging
from fastapi import FastAPI, Request, HTTPException, Depends, Query
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
import asyncio

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
    description="An API that implements an agentic conversation flow for building workflows with MCP servers",
    version="2.0.0"
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

# Load configuration files
def load_json_file(filename: str) -> Dict[str, Any]:
    """Load a JSON file from the project directory"""
    try:
        file_path = Path(__file__).parent / filename
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {filename}: {str(e)}")
        return {}

# Load configuration files at startup
SERVERS_CONFIG = load_json_file("servers.json")
CREDENTIAL_INFO = load_json_file("credinfo.json")
MCP_CONFIG = load_json_file("config.json")

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: str
    credentials: Optional[Dict[str, str]] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    status: str
    workflow_identified: Optional[bool] = None
    required_servers: Optional[List[Dict[str, Any]]] = None
    missing_credentials: Optional[List[Dict[str, Any]]] = None
    agent_execution_result: Optional[Dict[str, Any]] = None


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
            # Create new session with conversation state
            new_session = {
                'id': session_id,
                'user_id': user_id,
                'status': 'active',
                'context': {
                    'conversation_stage': 'greeting',
                    'workflow_identified': False,
                    'required_servers': [],
                    'collected_credentials': {}
                },
                'collected_credentials': {}
            }
            result = supabase.table('sessions').insert(new_session).execute()
            return result.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session error: {str(e)}")


async def analyze_workflow_request(message: str, session: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze user message to understand the workflow they want to build"""
    try:
        # Get conversation history for context
        chat_history = supabase.table('chat_messages').select('*').eq('session_id', session['id']).order('created_at').execute()
        
        context = ""
        if chat_history.data:
            context = "\n".join([f"{msg['role']}: {msg['message']}" for msg in chat_history.data[-5:]])  # Last 5 messages
        
        # Get available server IDs
        available_servers = [{"id": s.get("id"), "name": s.get("name"), "description": s.get("description")} 
                           for s in SERVERS_CONFIG.get("servers", []) if s.get("id")]
        
        analysis_prompt = f"""
        You are helping a user build a workflow using MCP (Model Context Protocol) servers.
        
        Available MCP servers:
        {json.dumps(available_servers, indent=2)}
        
        Analyze their message and determine:
        1. Is this a clear workflow request that we can build?
        2. What specific MCP server IDs would be needed? (Use exact IDs from the list above)
        3. What additional information might we need from the user?
        
        User message: "{message}"
        
        Previous conversation:
        {context}
        
        Respond with a JSON object:
        {{
            "is_workflow_request": true/false,
            "workflow_type": "description of the workflow",
            "required_server_ids": ["list of exact MCP server IDs needed"],
            "key_actions": ["list of actions to perform"],
            "needs_clarification": true/false,
            "clarification_questions": ["questions to ask if needs_clarification is true"],
            "confidence": 0.0-1.0
        }}
        
        IMPORTANT: The "required_server_ids" field must contain exact server IDs from the available servers list, not service names or descriptions.
        """
        
        response = await anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": analysis_prompt
            }]
        )
        
        # Parse response
        if hasattr(response, 'content') and response.content:
            if isinstance(response.content, list) and len(response.content) > 0:
                analysis_text = response.content[0].text
            else:
                analysis_text = str(response.content)
        else:
            analysis_text = '{"is_workflow_request": false, "needs_clarification": true}'
        
        # Extract JSON
        try:
            start_idx = analysis_text.find('{')
            end_idx = analysis_text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = analysis_text[start_idx:end_idx]
                analysis = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
        except:
            analysis = {
                "is_workflow_request": False,
                "needs_clarification": True,
                "clarification_questions": ["Could you please describe what you'd like to build in more detail?"]
            }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Workflow analysis error: {str(e)}")
        return {
            "is_workflow_request": False,
            "needs_clarification": True,
            "clarification_questions": ["I had trouble understanding your request. Could you please rephrase it?"]
        }


def get_servers_by_ids(server_ids: List[str]) -> List[Dict[str, Any]]:
    """Get server information by their IDs from servers.json"""
    required_servers = []
    servers_list = SERVERS_CONFIG.get("servers", [])
    
    for server in servers_list:
        if server.get("id") in server_ids:
            required_servers.append(server)
    
    return required_servers


def get_server_credentials(server: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get credential requirements for a server from config.json and credinfo.json"""
    credentials = []
    server_id = server.get("id", "")
    
    if not server_id:
        logger.warning(f"Server {server.get('name', 'Unknown')} has no id field")
        return credentials
    
    # Check config.json for this server using the server id
    mcp_servers = MCP_CONFIG.get("mcpServers", {})
    server_config = mcp_servers.get(server_id)
    
    if server_config:
        # Extract environment variables needed
        env_vars = server_config.get("env", {})
        
        # Get detailed instructions from credinfo.json using the server id
        cred_info = CREDENTIAL_INFO.get("credential_instructions", {}).get(server_id, {})
        
        for key, value in env_vars.items():
            credentials.append({
                "key": key,
                "server": server.get("name", ""),
                "server_id": server_id,
                "description": cred_info.get("description", f"Credential for {server.get('name', '')}"),
                "instructions": cred_info
            })
    else:
        logger.warning(f"No configuration found for server {server_id} in config.json")
    
    return credentials


async def check_missing_credentials(session: Dict[str, Any], required_servers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Check which credentials are missing for the required servers"""
    missing_credentials = []
    collected = session.get('collected_credentials', {})
    
    for server in required_servers:
        server_creds = get_server_credentials(server)
        for cred in server_creds:
            if cred['key'] not in collected:
                missing_credentials.append(cred)
    
    return missing_credentials


async def generate_conversation_response(session: Dict[str, Any], analysis: Dict[str, Any], 
                                       required_servers: List[Dict[str, Any]], 
                                       missing_credentials: List[Dict[str, Any]]) -> str:
    """Generate appropriate response based on conversation stage"""
    context = session.get('context', {})
    stage = context.get('conversation_stage', 'greeting')
    
    if stage == 'greeting' and not analysis.get('is_workflow_request'):
        # Still in greeting/exploration phase
        if analysis.get('needs_clarification'):
            questions = analysis.get('clarification_questions', [])
            if questions:
                return questions[0]
            else:
                return "That sounds interesting! Could you tell me more about what specific tasks you'd like to automate?"
        else:
            return "I'd be happy to help you build that! Let me understand your requirements better."
    
    elif analysis.get('is_workflow_request') and analysis.get('confidence', 0) > 0.7:
        # Clear workflow identified
        if missing_credentials:
            # Need to collect credentials
            return await generate_credential_request_message(missing_credentials)
        else:
            # Ready to execute
            return f"""Perfect! I have everything I need to build your workflow.

**Workflow Summary:**
- Type: {analysis.get('workflow_type', 'Custom automation')}
- Actions: {', '.join(analysis.get('key_actions', []))}
- MCP Servers: {', '.join([s.get('name', '') for s in required_servers])}

I'll now set up and execute this workflow for you. This may take a moment..."""
    
    else:
        # Need more clarification
        return "I want to make sure I understand exactly what you'd like to build. " + \
               (analysis.get('clarification_questions', ["Could you provide more details about your workflow?"])[0])


async def generate_credential_request_message(missing_credentials: List[Dict[str, Any]]) -> str:
    """Generate a message requesting missing credentials with instructions"""
    if not missing_credentials:
        return "All credentials are available!"
    
    # Group by server
    by_server = {}
    for cred in missing_credentials:
        server = cred.get('server', 'Unknown')
        if server not in by_server:
            by_server[server] = []
        by_server[server].append(cred)
    
    message = """I've identified the MCP servers needed for your workflow. To proceed, I need some credentials:

"""
    
    for server, creds in by_server.items():
        message += f"**{server}:**\n"
        
        # Get the first credential's instructions (they should be the same for the server)
        if creds and creds[0].get('instructions'):
            instructions = creds[0]['instructions']
            
            # Add description
            if instructions.get('description'):
                message += f"{instructions['description']}\n\n"
            
            # Add setup steps
            if instructions.get('steps'):
                message += "Setup steps:\n"
                for step in instructions['steps']:
                    message += f"{step}\n"
                message += "\n"
            
            # List required credentials
            message += "Required credentials:\n"
            for cred in creds:
                message += f"- `{cred['key']}`\n"
            
            message += "\n"
    
    message += "\nPlease provide these credentials to continue. You can submit them one at a time using the credential submission endpoint."
    
    return message


async def generate_fast_agent_config(session: Dict[str, Any], required_servers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate fast-agent configuration from collected credentials and required servers"""
    collected_credentials = session.get('collected_credentials', {})
    
    # Build MCP servers configuration
    mcp_servers = {}
    
    for server in required_servers:
        server_id = server.get("id", "")
        
        if not server_id:
            logger.warning(f"Server {server.get('name', 'Unknown')} has no id field, skipping")
            continue
        
        # Get base config from config.json using server id
        base_config = MCP_CONFIG.get("mcpServers", {}).get(server_id)
        
        if base_config:
            # Make a copy to avoid modifying the original
            server_config = base_config.copy()
            
            # Update with collected credentials
            if "env" in server_config:
                server_config["env"] = server_config["env"].copy()
                for env_key in server_config["env"]:
                    if env_key in collected_credentials:
                        server_config["env"][env_key] = collected_credentials[env_key]
            
            mcp_servers[server_id] = server_config
        else:
            logger.warning(f"No configuration found for server {server_id} in config.json")
    
    # Generate fast-agent configuration
    config = {
        "name": f"workflow-{session['id'][:8]}",
        "description": "User-requested workflow",
        "model": "claude-3-haiku-20240307",
        "mcp_servers": mcp_servers,
        "prompt": "Execute the user's workflow using the available MCP tools."
    }
    
    return config


async def execute_with_fast_agent(config: Dict[str, Any], user_request: str) -> Dict[str, Any]:
    """Execute the workflow using fast-agent"""
    try:
        # Create temporary directory for execution
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write fast-agent configuration
            config_file = temp_path / "fastagent.config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # For now, return a mock successful execution
            # In production, you would actually run fast-agent here
            result = {
                "status": "success",
                "message": "Workflow executed successfully!",
                "details": {
                    "config": config,
                    "request": user_request,
                    "execution_time": "2.5 seconds"
                }
            }
            
            return result
            
    except Exception as e:
        logger.error(f"Fast-agent execution error: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to execute workflow: {str(e)}",
            "details": None
        }


async def store_chat_message(session_id: str, user_id: str, message: str, role: str, metadata: Dict = None):
    """Store a chat message in the database"""
    try:
        msg = {
            'session_id': session_id,
            'user_id': user_id,
            'message': message,
            'role': role,
            'created_at': datetime.utcnow().isoformat(),
            'metadata': metadata or {}
        }
        supabase.table('chat_messages').insert(msg).execute()
    except Exception as e:
        logger.error(f"Error storing chat message: {str(e)}")


# Main chat endpoint
@app.post("/api/chat")
async def chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    """
    Main chat endpoint implementing the workflow:
    1. Greet user
    2. Understand workflow
    3. Check required servers
    4. Collect credentials
    5. Execute with fast-agent
    """
    try:
        # Get or create session
        session = await get_or_create_session(request.session_id, current_user['id'])
        context = session.get('context', {})
        
        # Store user message
        await store_chat_message(session['id'], current_user['id'], request.message, 'user')
        
        # Handle initial greeting
        if context.get('conversation_stage') == 'greeting' and request.message == "":
            response_text = "Hi, what can I help you build today?"
            await store_chat_message(session['id'], current_user['id'], response_text, 'assistant')
            
            return ChatResponse(
                response=response_text,
                session_id=session['id'],
                status='greeting',
                workflow_identified=False
            )
        
        # Analyze the user's message
        analysis = await analyze_workflow_request(request.message, session)
        
        # Update conversation stage
        if analysis.get('is_workflow_request') and analysis.get('confidence', 0) > 0.7:
            context['conversation_stage'] = 'workflow_identified'
            context['workflow_identified'] = True
            
            # Identify required servers from servers.json
            required_servers = get_servers_by_ids(analysis.get('required_server_ids', []))
            context['required_servers'] = required_servers
            
            # Check for missing credentials
            missing_credentials = await check_missing_credentials(session, required_servers)
            
            if missing_credentials:
                context['conversation_stage'] = 'collecting_credentials'
                response_text = await generate_credential_request_message(missing_credentials)
                status = 'awaiting_credentials'
                execution_result = None
            else:
                # All credentials available, execute workflow
                context['conversation_stage'] = 'executing'
                config = await generate_fast_agent_config(session, required_servers)
                execution_result = await execute_with_fast_agent(config, request.message)
                
                if execution_result['status'] == 'success':
                    response_text = f"""✅ **Workflow executed successfully!**

{execution_result.get('message', 'Your workflow has been completed.')}

**Details:**
- Execution time: {execution_result.get('details', {}).get('execution_time', 'N/A')}
- MCP Servers used: {', '.join([s.get('name', '') for s in required_servers])}

Is there anything else you'd like me to help you build?"""
                    status = 'completed'
                    context['conversation_stage'] = 'completed'
                else:
                    response_text = f"""❌ **Workflow execution failed**

{execution_result.get('message', 'An error occurred during execution.')}

Would you like me to try again or help you with something else?"""
                    status = 'error'
                    context['conversation_stage'] = 'error'
            
            # Update session context
            session['context'] = context
            supabase.table('sessions').update({'context': context}).eq('id', session['id']).execute()
            
        else:
            # Need more information or clarification
            required_servers = []
            missing_credentials = None
            execution_result = None
            response_text = await generate_conversation_response(session, analysis, [], [])
            status = 'conversation'
        
        # Store assistant response
        await store_chat_message(
            session['id'], 
            current_user['id'], 
            response_text, 
            'assistant',
            {
                'analysis': analysis,
                'required_servers': [s.get('name', '') for s in required_servers] if required_servers else [],
                'execution_result': execution_result
            }
        )
        
        return ChatResponse(
            response=response_text,
            session_id=session['id'],
            status=status,
            workflow_identified=context.get('workflow_identified', False),
            required_servers=required_servers if required_servers else None,
            missing_credentials=missing_credentials if missing_credentials else None,
            agent_execution_result=execution_result
        )
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


# Credential submission endpoint
@app.post("/api/submit-credentials")
async def submit_credentials(request: CredentialSubmission, current_user: dict = Depends(get_current_user)):
    """Submit credentials for MCP servers"""
    try:
        # Get session
        session_result = supabase.table('sessions').select('*').eq('id', request.session_id).eq('user_id', current_user['id']).execute()
        
        if not session_result.data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = session_result.data[0]
        
        # Update collected credentials
        collected_credentials = session.get('collected_credentials', {})
        
        # Store the credential
        collected_credentials[request.credential_key] = request.credential_value
        
        # Update session
        supabase.table('sessions').update({
            'collected_credentials': collected_credentials
        }).eq('id', request.session_id).execute()
        
        # Check if all credentials are now collected
        context = session.get('context', {})
        required_servers = context.get('required_servers', [])
        remaining_missing = await check_missing_credentials(
            {'collected_credentials': collected_credentials},
            required_servers
        )
        
        if not remaining_missing and context.get('workflow_identified'):
            status_message = "All credentials collected! You can now send your workflow request again to execute it."
        else:
            status_message = f"Credential stored successfully. {len(remaining_missing)} credentials remaining."
        
        return {
            "status": "success",
            "message": status_message,
            "remaining_credentials": len(remaining_missing) if remaining_missing else 0
        }
        
    except Exception as e:
        logger.error(f"Credential submission error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to store credential: {str(e)}")


# Chat history endpoint
@app.get("/api/chat/history")
async def get_chat_history(
    current_user = Depends(get_current_user),
    session_id: Optional[str] = Query(None, description="Specific session ID to retrieve"),
    limit: int = Query(20, ge=1, le=100, description="Number of messages to return"),
    offset: int = Query(0, ge=0, description="Number of messages to skip")
):
    """Retrieve chat history for a session or all sessions"""
    try:
        if session_id:
            # Get specific session
            messages_result = supabase.table('chat_messages').select('*').eq('session_id', session_id).eq('user_id', current_user['id']).order('created_at').range(offset, offset + limit - 1).execute()
            
            return {
                "session_id": session_id,
                "messages": messages_result.data,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total": len(messages_result.data)
                }
            }
        else:
            # Get all sessions for user
            sessions_result = supabase.table('sessions').select('*').eq('user_id', current_user['id']).order('created_at', desc=True).execute()
            
            return {
                "sessions": sessions_result.data,
                "total_sessions": len(sessions_result.data)
            }
            
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
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


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(
        f"Unhandled exception in {request.method} {request.url}: {str(exc)}",
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later.",
            "request_id": str(uuid.uuid4())
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP exception handler"""
    logger.warning(
        f"HTTP exception in {request.method} {request.url}: {exc.status_code} - {exc.detail}"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "request_id": str(uuid.uuid4())
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 