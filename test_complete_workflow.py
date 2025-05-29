#!/usr/bin/env python3

import requests
import json
import uuid

def test_complete_workflow():
    """Test the complete workflow with simulated credentials"""
    
    session_id = str(uuid.uuid4())
    
    print("🚀 LIVE WORKFLOW TEST: Screenshot + Google Drive Upload")
    print("=" * 60)
    
    # Step 1: Initial workflow request
    print("📝 Step 1: Initial workflow request")
    url = "http://localhost:8000/api/live-workflow-test"
    test_data = {
        "message": "Take screenshot of google.com and upload it on my google drive",
        "session_id": session_id,
        "credentials": {}
    }
    
    response = requests.post(url, json=test_data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Workflow analyzed successfully!")
        print(f"Required credentials: {result.get('required_credentials', [])}")
        print(f"Session ID: {session_id}")
    else:
        print(f"❌ Failed: {response.text}")
        return
    
    print("\n" + "-" * 60)
    
    # Step 2: Submit mock credentials
    print("🔑 Step 2: Submitting Google Drive credentials")
    
    # Submit Google Drive credentials path
    cred_url = "http://localhost:8000/api/submit-credentials"
    
    # Mock credential 1
    cred_data1 = {
        "session_id": session_id,
        "credential_name": "GOOGLE_DRIVE_CREDENTIALS_PATH",
        "credential_value": "/path/to/google-drive-credentials.json"
    }
    
    response1 = requests.post(cred_url, data=cred_data1)
    if response1.status_code == 200:
        print("✅ Google Drive credentials path submitted")
    else:
        print(f"❌ Failed to submit credentials: {response1.text}")
    
    # Mock credential 2
    cred_data2 = {
        "session_id": session_id,
        "credential_name": "GOOGLE_DRIVE_TOKEN_PATH",
        "credential_value": "/path/to/google-drive-token.json"
    }
    
    response2 = requests.post(cred_url, data=cred_data2)
    if response2.status_code == 200:
        print("✅ Google Drive token path submitted")
    else:
        print(f"❌ Failed to submit token: {response2.text}")
    
    print("\n" + "-" * 60)
    
    # Step 3: Re-run workflow with credentials
    print("🚀 Step 3: Executing workflow with credentials")
    
    response3 = requests.post(url, json=test_data)
    print(f"Status: {response3.status_code}")
    
    if response3.status_code == 200:
        result = response3.json()
        print("✅ WORKFLOW EXECUTION COMPLETED!")
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Agent Created: {result.get('agent_created', False)}")
        print(f"Agent ID: {result.get('agent_id', 'N/A')}")
        
        print("\n📋 Full Response:")
        print(result.get('reply', 'No reply'))
        
    else:
        print(f"❌ Workflow execution failed: {response3.text}")
    
    print("\n" + "=" * 60)
    print("🎉 Live workflow test completed!")
    
    return session_id

if __name__ == "__main__":
    session_id = test_complete_workflow()
    print(f"\n💡 Session ID for reference: {session_id}") 