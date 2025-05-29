"""
MCP Server Data Management Script
Loads MCP server data from JSON files, generates embeddings, and populates the database.
"""

import json
import os
import asyncio
from typing import Dict, List, Any
from openai import AsyncOpenAI
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize clients
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not all([supabase_url, supabase_key, openai_api_key]):
    raise ValueError("SUPABASE_URL, SUPABASE_KEY, and OPENAI_API_KEY environment variables are required")

supabase: Client = create_client(supabase_url, supabase_key)
openai_client = AsyncOpenAI(api_key=openai_api_key)


async def generate_embedding(text: str) -> List[float]:
    """Generate embedding for text using OpenAI"""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return []


def load_json_file(filename: str) -> Dict[str, Any]:
    """Load and parse JSON file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing {filename}: {str(e)}")
        return {}


def extract_server_tools(server_data: Dict[str, Any]) -> List[str]:
    """Extract tool names from server data"""
    tools = server_data.get('tools', [])
    if isinstance(tools, list):
        # Handle both string lists and object lists
        tool_names = []
        for tool in tools:
            if isinstance(tool, str):
                tool_names.append(tool)
            elif isinstance(tool, dict) and 'name' in tool:
                tool_names.append(tool['name'])
        return tool_names
    return []


def find_server_config(server_id: str, config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Find configuration for a server by ID"""
    mcp_servers = config_data.get('mcpServers', {})
    
    # Try exact match first
    if server_id in mcp_servers:
        return mcp_servers[server_id]
    
    # Try case-insensitive match
    for key, value in mcp_servers.items():
        if key.lower() == server_id.lower():
            return value
    
    return {}


def find_credential_info(server_id: str, cred_data: Dict[str, Any]) -> Dict[str, Any]:
    """Find credential information for a server by ID"""
    cred_instructions = cred_data.get('credential_instructions', {})
    
    # Try exact match first
    if server_id in cred_instructions:
        return cred_instructions[server_id]
    
    # Try case-insensitive match
    for key, value in cred_instructions.items():
        if key.lower() == server_id.lower():
            return value
    
    return {}


async def process_servers_data(servers_data: Dict[str, Any], config_data: Dict[str, Any], cred_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process and combine server data from all sources"""
    processed_servers = []
    
    # Handle both array and object formats for servers
    servers_list = servers_data.get('servers', []) if 'servers' in servers_data else []
    
    for server in servers_list:
        server_id = server.get('id', server.get('name', '').lower().replace(' ', '-'))
        server_name = server.get('name', '')
        
        if not server_name:
            continue
        
        # Get configuration and credential info
        server_config = find_server_config(server_id, config_data)
        cred_info = find_credential_info(server_id, cred_data)
        
        # Extract tools
        tools = extract_server_tools(server)
        
        # Create embedding text
        embedding_text = f"{server_name} {server.get('description', '')} {server.get('type', '')} {' '.join(tools)}"
        
        # Generate embedding
        embedding = await generate_embedding(embedding_text)
        
        # Prepare server record
        server_record = {
            'name': server_name,
            'description': server.get('description', ''),
            'type': server.get('type', 'general'),
            'tools': tools,
            'command': server.get('command', server_config.get('command', '')),
            'args': server.get('args', server_config.get('args', [])),
            'config_template': server_config,
            'credential_info': cred_info,
            'embedding': embedding
        }
        
        processed_servers.append(server_record)
        print(f"Processed server: {server_name}")
    
    return processed_servers


async def upsert_servers_to_database(servers: List[Dict[str, Any]]) -> None:
    """Upsert server data to the database"""
    try:
        for server in servers:
            # Use upsert to handle both insert and update
            result = supabase.table('mcp_servers').upsert(
                server,
                on_conflict='name'  # Use name as the conflict resolution key
            ).execute()
            
            if result.data:
                print(f"✓ Upserted server: {server['name']}")
            else:
                print(f"✗ Failed to upsert server: {server['name']}")
                
    except Exception as e:
        print(f"Error upserting servers to database: {str(e)}")
        raise


async def load_mcp_data() -> None:
    """Main function to load MCP server data"""
    try:
        print("Loading MCP server data...")
        
        # Load JSON files
        print("Loading JSON files...")
        servers_data = load_json_file('servers.json')
        config_data = load_json_file('config.json')
        cred_data = load_json_file('credinfo.json')
        
        if not servers_data:
            print("No server data found in servers.json")
            return
        
        # Process server data
        print("Processing server data...")
        processed_servers = await process_servers_data(servers_data, config_data, cred_data)
        
        if not processed_servers:
            print("No servers to process")
            return
        
        # Upsert to database
        print("Upserting to database...")
        await upsert_servers_to_database(processed_servers)
        
        print(f"✓ Successfully loaded {len(processed_servers)} MCP servers to database")
        
    except Exception as e:
        print(f"Error loading MCP data: {str(e)}")
        raise


# Allow running as standalone script
if __name__ == "__main__":
    asyncio.run(load_mcp_data()) 