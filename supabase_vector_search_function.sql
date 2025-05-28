-- Supabase Vector Search Function for MCP Servers
-- This function should be created in your Supabase SQL editor

-- Function to search MCP servers by vector similarity
CREATE OR REPLACE FUNCTION match_mcp_servers(
  query_embedding vector(1536),
  match_threshold float,
  match_count int
)
RETURNS TABLE (
  id uuid,
  server_name varchar,
  description text,
  tools jsonb,
  credential_requirements jsonb,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    mcp.id,
    mcp.server_name,
    mcp.description,
    mcp.tools,
    mcp.credential_requirements,
    1 - (mcp.embedding <=> query_embedding) as similarity
  FROM mcp_server_cache mcp
  WHERE 1 - (mcp.embedding <=> query_embedding) > match_threshold
  ORDER BY mcp.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- Function to search MCP servers by keyword
CREATE OR REPLACE FUNCTION search_mcp_servers_by_keyword(
  search_term text,
  max_results int DEFAULT 10
)
RETURNS TABLE (
  id uuid,
  server_name varchar,
  description text,
  tools jsonb,
  credential_requirements jsonb,
  relevance float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    mcp.id,
    mcp.server_name,
    mcp.description,
    mcp.tools,
    mcp.credential_requirements,
    ts_rank(
      to_tsvector('english', mcp.server_name || ' ' || mcp.description || ' ' || mcp.tools::text),
      plainto_tsquery('english', search_term)
    ) as relevance
  FROM mcp_server_cache mcp
  WHERE 
    to_tsvector('english', mcp.server_name || ' ' || mcp.description || ' ' || mcp.tools::text) 
    @@ plainto_tsquery('english', search_term)
  ORDER BY relevance DESC
  LIMIT max_results;
END;
$$;

-- Create text search index for faster keyword searches
CREATE INDEX IF NOT EXISTS idx_mcp_text_search 
ON mcp_server_cache 
USING gin(to_tsvector('english', server_name || ' ' || description || ' ' || tools::text));

-- Function to get credential requirements for a server
CREATE OR REPLACE FUNCTION get_server_credential_requirements(
  p_server_name varchar
)
RETURNS jsonb
LANGUAGE plpgsql
AS $$
DECLARE
  cred_requirements jsonb;
BEGIN
  SELECT credential_requirements 
  INTO cred_requirements
  FROM mcp_server_cache
  WHERE server_name = p_server_name;
  
  RETURN COALESCE(cred_requirements, '{}'::jsonb);
END;
$$;

-- Function to batch search multiple servers
CREATE OR REPLACE FUNCTION batch_search_mcp_servers(
  server_names text[]
)
RETURNS TABLE (
  server_name varchar,
  description text,
  tools jsonb,
  credential_requirements jsonb
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    mcp.server_name,
    mcp.description,
    mcp.tools,
    mcp.credential_requirements
  FROM mcp_server_cache mcp
  WHERE mcp.server_name = ANY(server_names);
END;
$$; 