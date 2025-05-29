#!/usr/bin/env python3

import requests
import json
import uuid
import time
import getpass
from typing import Dict, List, Optional
import sys

class AgentTester:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session_id = str(uuid.uuid4())
        
    def print_header(self, title: str):
        """Print a formatted header"""
        print("\n" + "="*60)
        print(f"🤖 {title}")
        print("="*60)
    
    def print_response(self, response: Dict):
        """Print the agent's response"""
        print("\n🤖 AGENT RESPONSE:")
        print("-" * 40)
        print(response.get('response', 'No response'))
        
        if response.get('workflow_identified'):
            print("\n✅ Workflow identified!")
            
        if response.get('required_servers'):
            print("\n📦 Required MCP Servers:")
            for server in response['required_servers']:
                print(f"  - {server.get('name', 'Unknown')}: {server.get('description', '')}")
        
        if response.get('missing_credentials'):
            print(f"\n🔑 Missing Credentials: {len(response['missing_credentials'])}")
    
    def make_chat_request(self, message: str = "") -> Optional[Dict]:
        """Make a chat request to the API"""
        chat_data = {
            "message": message,
            "session_id": self.session_id,
            "credentials": {}
        }
        
        try:
            response = requests.post(f"{self.base_url}/api/chat", json=chat_data)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Error: {response.status_code}")
                print(response.text)
                return None
                
        except requests.exceptions.ConnectionError:
            print("❌ Error: Could not connect to the server.")
            print(f"Make sure it's running on {self.base_url}")
            return None
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return None
    
    def submit_credential(self, key: str, value: str) -> bool:
        """Submit a single credential"""
        cred_data = {
            "session_id": self.session_id,
            "credential_key": key,
            "credential_value": value,
            "credential_type": "text"
        }
        
        try:
            response = requests.post(f"{self.base_url}/api/submit-credentials", json=cred_data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {result.get('message', 'Credential submitted')}")
                remaining = result.get('remaining_credentials', 0)
                if remaining > 0:
                    print(f"   ({remaining} credentials remaining)")
                return True
            else:
                print(f"❌ Error submitting credential: {response.status_code}")
                print(response.text)
                return False
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return False
    
    def run_interactive_test(self):
        """Run the interactive test flow"""
        self.print_header("DYNAMIC MCP AGENT - INTERACTIVE TEST")
        print(f"Session ID: {self.session_id}")
        
        # Check if server is running
        try:
            response = requests.get(f"{self.base_url}/")
            if response.status_code == 200:
                server_info = response.json()
                print(f"✅ Server is running! Version: {server_info.get('version', 'Unknown')}")
            else:
                print("⚠️ Server responded with unexpected status")
                return
        except:
            print("❌ Cannot connect to server. Please start it first.")
            return
        
        # Step 1: Get initial greeting
        print("\n1️⃣ Getting initial greeting...")
        response = self.make_chat_request("")
        if not response:
            return
        
        self.print_response(response)
        
        # Step 2: Send workflow request
        print("\n2️⃣ Describe what you want to build:")
        user_message = input("\n💬 You: ").strip()
        
        if not user_message:
            print("No message provided. Exiting.")
            return
        
        response = self.make_chat_request(user_message)
        if not response:
            return
        
        self.print_response(response)
        
        # Continue conversation until workflow is identified
        while response.get('status') == 'conversation':
            user_message = input("\n💬 You: ").strip()
            if user_message.lower() == 'quit':
                break
                
            response = self.make_chat_request(user_message)
            if not response:
                break
                
            self.print_response(response)
        
        # Handle credentials if needed
        if response and response.get('missing_credentials'):
            print("\n3️⃣ The agent needs credentials. Would you like to provide them?")
            choice = input("Enter 'y' to provide credentials or any other key to skip: ").strip().lower()
            
            if choice == 'y':
                for cred in response['missing_credentials']:
                    print(f"\n📝 {cred.get('server', 'Unknown')} - {cred.get('key', 'Unknown')}")
                    value = input(f"Enter value for {cred['key']} (or press Enter to skip): ").strip()
                    
                    if value:
                        self.submit_credential(cred['key'], value)
                
                # Re-send the workflow request
                print("\n4️⃣ Re-sending workflow request...")
                response = self.make_chat_request(user_message)
                if response:
                    self.print_response(response)
        
        # Check final status
        if response:
            status = response.get('status', 'unknown')
            if status == 'completed':
                print("\n🎉 WORKFLOW EXECUTED SUCCESSFULLY!")
            elif status == 'awaiting_credentials':
                print("\n⏳ Still waiting for credentials.")
            elif status == 'error':
                print("\n❌ Workflow execution failed.")
            else:
                print(f"\n📊 Final status: {status}")

def main():
    """Main function"""
    print("🚀 Starting Dynamic MCP Agent Test")
    
    # Create tester instance
    tester = AgentTester()
    
    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick test mode
        print("\n🚀 QUICK TEST MODE")
        
        # Get greeting
        response = tester.make_chat_request("")
        if response:
            print(f"Greeting: {response.get('response', '')[:50]}...")
        
        # Send test workflow
        test_message = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Take a screenshot of google.com"
        print(f"\nTesting with: '{test_message}'")
        
        response = tester.make_chat_request(test_message)
        if response:
            tester.print_response(response)
    else:
        # Interactive mode
        tester.run_interactive_test()

if __name__ == "__main__":
    main() 