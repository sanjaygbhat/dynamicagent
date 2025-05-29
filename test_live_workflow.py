#!/usr/bin/env python3

import requests
import json
import uuid

def test_live_workflow():
    """Test the live workflow endpoint"""
    
    # API endpoint
    url = "http://localhost:8000/api/live-workflow-test"
    
    # Test data
    session_id = str(uuid.uuid4())
    test_data = {
        "message": "Take screenshot of google.com and upload it on my google drive",
        "session_id": session_id,
        "credentials": {}
    }
    
    print(f"🚀 Testing live workflow...")
    print(f"Session ID: {session_id}")
    print(f"Message: {test_data['message']}")
    print(f"URL: {url}")
    print("-" * 50)
    
    try:
        # Make the request
        response = requests.post(url, json=test_data)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print("-" * 50)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            print(f"Reply: {result.get('reply', 'No reply')}")
            print(f"Status: {result.get('status', 'No status')}")
            print(f"Required Credentials: {result.get('required_credentials', [])}")
            print(f"Agent Created: {result.get('agent_created', False)}")
            
            if result.get('required_credentials'):
                print("\n📋 Next Steps:")
                print("The workflow is asking for credentials. You'll need to:")
                for cred in result.get('required_credentials', []):
                    print(f"  - Provide: {cred}")
                print(f"\nUse session ID: {session_id} for credential submission")
                
        else:
            print("❌ ERROR!")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")

def test_credential_submission(session_id, cred_name, cred_value):
    """Test credential submission"""
    
    url = "http://localhost:8000/api/submit-credentials"
    
    data = {
        "session_id": session_id,
        "credential_name": cred_name,
        "credential_value": cred_value
    }
    
    print(f"📤 Submitting credential: {cred_name}")
    
    try:
        response = requests.post(url, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Credential submitted: {result}")
            return True
        else:
            print(f"❌ Failed to submit credential: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception submitting credential: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Live Workflow Test")
    print("=" * 50)
    
    # Test the workflow
    test_live_workflow()
    
    print("\n" + "=" * 50)
    print("💡 To continue testing with credentials:")
    print("1. Set up Google Drive API credentials")
    print("2. Use test_credential_submission() function")
    print("3. Re-run the workflow test") 