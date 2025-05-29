#!/usr/bin/env python3

import requests
import json
import uuid
import time

def test_agentic_conversation_flow():
    """Test the complete agentic conversation flow with fast-agent"""
    
    base_url = "http://localhost:8000"
    session_id = str(uuid.uuid4())
    
    print("🚀 TESTING AGENTIC CONVERSATION FLOW WITH FAST-AGENT")
    print("=" * 60)
    print(f"Session ID: {session_id}")
    print()
    
    # Step 1: Initial workflow request
    print("📝 Step 1: Initial workflow request")
    print("-" * 40)
    
    chat_data = {
        "message": "Take screenshot of google.com and upload it on my google drive",
        "session_id": session_id,
        "credentials": {}
    }
    
    response = requests.post(f"{base_url}/api/chat", json=chat_data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result['response']}")
        print(f"Status: {result['status']}")
        print(f"Required Credentials: {result.get('required_credentials', [])}")
        print()
        
        if result['status'] == 'awaiting_credentials':
            required_creds = result.get('required_credentials', [])
            
            # Step 2: Submit credentials one by one
            print("🔐 Step 2: Submitting credentials")
            print("-" * 40)
            
            for i, cred in enumerate(required_creds):
                print(f"Submitting credential {i+1}/{len(required_creds)}: {cred}")
                
                # Submit credential
                cred_data = {
                    "session_id": session_id,
                    "credential_key": cred,
                    "credential_value": f"/path/to/{cred.lower()}.json",  # Mock credential path
                    "credential_type": "text"
                }
                
                cred_response = requests.post(f"{base_url}/api/submit-credentials", json=cred_data)
                print(f"  Credential submission status: {cred_response.status_code}")
                
                if cred_response.status_code == 200:
                    cred_result = cred_response.json()
                    print(f"  Result: {cred_result['message']}")
                else:
                    print(f"  Error: {cred_response.text}")
                
                time.sleep(0.5)  # Small delay between submissions
            
            print()
            
            # Step 3: Send the same message again to trigger execution
            print("⚡ Step 3: Triggering workflow execution")
            print("-" * 40)
            
            execution_response = requests.post(f"{base_url}/api/chat", json=chat_data)
            print(f"Status: {execution_response.status_code}")
            
            if execution_response.status_code == 200:
                exec_result = execution_response.json()
                print(f"Response: {exec_result['response']}")
                print(f"Status: {exec_result['status']}")
                
                if exec_result.get('agent_execution_result'):
                    agent_result = exec_result['agent_execution_result']
                    print()
                    print("🤖 Agent Execution Results:")
                    print(f"  Agent Name: {agent_result.get('agent_name', 'N/A')}")
                    print(f"  Status: {agent_result.get('status', 'N/A')}")
                    print(f"  Execution Time: {agent_result.get('execution_time', 'N/A')}")
                    
                    if agent_result.get('steps_executed'):
                        print("  Steps Executed:")
                        for step in agent_result['steps_executed']:
                            print(f"    {step['step']}. {step['action']} -> {step['result']}")
                    
                    print(f"  Final Output: {agent_result.get('final_output', 'N/A')}")
                
            else:
                print(f"Error: {execution_response.text}")
        
    else:
        print(f"Error: {response.text}")
    
    print()
    print("✅ Test completed!")

def test_different_workflows():
    """Test different types of workflows"""
    
    workflows = [
        "Send a message to my team on Slack about the project update",
        "Create a GitHub issue for the bug we discussed",
        "Search for recent emails about the meeting and summarize them",
        "Upload the presentation file to Google Drive and share it with the team"
    ]
    
    print("🔄 TESTING DIFFERENT WORKFLOW TYPES")
    print("=" * 50)
    
    for i, workflow in enumerate(workflows, 1):
        print(f"\n{i}. Testing: {workflow}")
        print("-" * 40)
        
        session_id = str(uuid.uuid4())
        chat_data = {
            "message": workflow,
            "session_id": session_id,
            "credentials": {}
        }
        
        response = requests.post("http://localhost:8000/api/chat", json=chat_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Workflow Type: {result.get('analysis', {}).get('workflow_type', 'Unknown')}")
            print(f"Required Services: {result.get('analysis', {}).get('required_services', [])}")
            print(f"Status: {result['status']}")
            print(f"Required Credentials: {len(result.get('required_credentials', []))}")
        else:
            print(f"Error: {response.status_code}")

if __name__ == "__main__":
    try:
        # Test main agentic flow
        test_agentic_conversation_flow()
        
        print("\n" + "="*60 + "\n")
        
        # Test different workflow types
        test_different_workflows()
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the server. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}") 