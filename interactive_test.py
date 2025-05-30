#!/usr/bin/env python3

import requests
import json
import uuid
import time
import getpass
from typing import Dict, List, Optional
import sys

class InteractiveAgentTester:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session_id = str(uuid.uuid4())
        
    def print_header(self, title: str):
        """Print a formatted header"""
        print("\n" + "="*60)
        print(f"🤖 {title}")
        print("="*60)
    
    def print_step(self, step: str):
        """Print a formatted step"""
        print(f"\n📋 {step}")
        print("-"*50)
    
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
    
    def interactive_credential_collection(self, missing_credentials: List[Dict]) -> bool:
        """Interactively collect missing credentials"""
        print("\n🔐 CREDENTIAL COLLECTION")
        print("The agent needs some credentials to proceed.")
        print("You can enter them now or skip to test the conversation flow.")
        
        for cred in missing_credentials:
            print(f"\n📝 {cred.get('server', 'Unknown Server')} - {cred.get('key', 'Unknown Key')}")
            
            # Show description if available
            instructions = cred.get('instructions', {})
            if instructions.get('description'):
                print(f"   {instructions['description']}")
            
            # Ask if user wants to provide this credential
            skip = input("\nPress Enter to provide this credential, or type 'skip' to skip: ").strip().lower()
            
            if skip == 'skip':
                print("   ⏭️  Skipped")
                continue
            
            # Get credential value
            if "token" in cred['key'].lower() or "key" in cred['key'].lower() or "secret" in cred['key'].lower():
                value = getpass.getpass(f"Enter {cred['key']} (hidden): ").strip()
            else:
                value = input(f"Enter {cred['key']}: ").strip()
            
            if value:
                if self.submit_credential(cred['key'], value):
                    print("   ✅ Credential stored")
                else:
                    print("   ❌ Failed to store credential")
            else:
                print("   ⏭️  Skipped (no value provided)")
        
        return True
    
    def run_conversation_flow(self):
        """Run the complete conversation flow"""
        self.print_header("DYNAMIC MCP AGENT - INTERACTIVE TEST")
        print(f"Session ID: {self.session_id}")
        print("\nThis test will walk you through the agent's conversation flow.")
        
        # Step 1: Initial greeting (empty message)
        self.print_step("INITIAL GREETING")
        print("Sending empty message to trigger greeting...")
        
        response = self.make_chat_request("")
        if not response:
            print("❌ Failed to get initial greeting. Exiting.")
            return
        
        self.print_response(response)
        
        # Step 2: Conversation loop
        self.print_step("WORKFLOW CONVERSATION")
        print("Now you can describe what you want to build.")
        print("The agent will keep talking until it understands your workflow.")
        print("Type 'quit' to exit at any time.")
        
        workflow_identified = False
        
        while True:
            # Get user input
            user_message = input("\n💬 You: ").strip()
            
            if user_message.lower() == 'quit':
                print("\n👋 Goodbye!")
                break
            
            # Send message to agent
            response = self.make_chat_request(user_message)
            if not response:
                print("❌ Failed to get response. Try again.")
                continue
            
            self.print_response(response)
            
            # Check if workflow is identified
            if response.get('workflow_identified'):
                workflow_identified = True
                
                # Check for missing credentials
                if response.get('missing_credentials'):
                    print("\n🔑 The agent needs credentials to proceed.")
                    collect = input("Would you like to provide credentials now? (y/n): ").strip().lower()
                    
                    if collect == 'y':
                        self.interactive_credential_collection(response['missing_credentials'])
                        
                        # Re-send the workflow request after providing credentials
                        print("\n📤 Re-sending workflow request with credentials...")
                        response = self.make_chat_request(user_message)
                        if response:
                            self.print_response(response)
                            
                            if response.get('status') == 'completed':
                                print("\n🎉 WORKFLOW EXECUTED SUCCESSFULLY!")
                            elif response.get('status') == 'awaiting_credentials':
                                print("\n⏳ Still waiting for some credentials.")
                    else:
                        print("\n⏭️  Skipping credential collection.")
                
                elif response.get('status') == 'completed':
                    print("\n🎉 WORKFLOW EXECUTED SUCCESSFULLY!")
                    break
            
            # Check if agent is still in conversation mode
            if response.get('status') == 'conversation':
                continue
            elif response.get('status') == 'completed':
                print("\n✅ Workflow completed!")
                break
            elif response.get('status') == 'error':
                print("\n❌ Workflow execution failed.")
                break
    
    def run_quick_test(self, workflow_request: str):
        """Run a quick test with a predefined workflow request"""
        self.print_header("QUICK WORKFLOW TEST")
        print(f"Testing with: '{workflow_request}'")
        print(f"Session ID: {self.session_id}")
        
        # Get initial greeting
        print("\n1️⃣ Getting initial greeting...")
        response = self.make_chat_request("")
        if response:
            print(f"   ✅ {response.get('response', '')[:50]}...")
        
        # Send workflow request
        print(f"\n2️⃣ Sending workflow request...")
        response = self.make_chat_request(workflow_request)
        if response:
            self.print_response(response)
            
            if response.get('missing_credentials'):
                print("\n3️⃣ Credentials needed - test complete.")
                print("   Run in interactive mode to provide credentials.")
            elif response.get('status') == 'completed':
                print("\n3️⃣ Workflow executed successfully!")
            else:
                print(f"\n3️⃣ Status: {response.get('status', 'unknown')}")

def main():
    """Main function"""
    print("🚀 Starting Dynamic MCP Agent Interactive Tester")
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8001/")
        if response.status_code == 200:
            print("✅ Server is running!")
            server_info = response.json()
            print(f"   Version: {server_info.get('version', 'Unknown')}")
            print(f"   Features: {', '.join(server_info.get('features', []))}")
        else:
            print("⚠️ Server responded but with unexpected status")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Please start it with:")
        print("   source .venv/bin/activate")
        print("   export SUPABASE_URL='...' && export SUPABASE_KEY='...' && export ANTHROPIC_API_KEY='...'")
        print("   python3 -m uvicorn main:app --reload --port 8001")
        return
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick" and len(sys.argv) > 2:
            # Quick test mode
            workflow = ' '.join(sys.argv[2:])
            tester = InteractiveAgentTester()
            tester.run_quick_test(workflow)
        else:
            print("\nUsage:")
            print("  python3 interactive_test.py              # Interactive mode")
            print("  python3 interactive_test.py --quick <workflow>  # Quick test")
    else:
        # Interactive mode
        tester = InteractiveAgentTester()
        tester.run_conversation_flow()

if __name__ == "__main__":
    main() 