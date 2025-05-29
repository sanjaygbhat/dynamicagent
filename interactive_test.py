#!/usr/bin/env python3

import requests
import json
import uuid
import time
import getpass
from typing import Dict, List

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
    
    def get_user_workflow_request(self) -> str:
        """Get workflow request from user"""
        self.print_step("WORKFLOW REQUEST")
        print("Please describe what you want the AI agent to do.")
        print("Examples:")
        print("  • Take screenshot of google.com and upload it to my Google Drive")
        print("  • Send a message to my team on Slack about project updates")
        print("  • Create a GitHub issue for the bug we discussed")
        print("  • Search my emails for meeting notes and summarize them")
        print()
        
        while True:
            request = input("Your workflow request: ").strip()
            if request:
                return request
            print("Please enter a valid request.")
    
    def make_initial_request(self, workflow_request: str) -> Dict:
        """Make initial chat request to analyze workflow"""
        self.print_step("ANALYZING WORKFLOW")
        print(f"Sending request: {workflow_request}")
        print(f"Session ID: {self.session_id}")
        
        chat_data = {
            "message": workflow_request,
            "session_id": self.session_id,
            "credentials": {}
        }
        
        try:
            response = requests.post(f"{self.base_url}/api/chat", json=chat_data)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Analysis completed!")
                print(f"Status: {result['status']}")
                
                # Display the agent's response
                print("\n🤖 AGENT RESPONSE:")
                print("-" * 40)
                print(result['response'])
                
                if result.get('analysis'):
                    analysis = result['analysis']
                    print(f"\n📊 WORKFLOW ANALYSIS:")
                    print(f"Type: {analysis.get('workflow_type', 'Unknown')}")
                    print(f"Services: {analysis.get('required_services', [])}")
                    print(f"Actions: {analysis.get('key_actions', [])}")
                
                return result
            else:
                print(f"❌ Error: {response.status_code}")
                print(response.text)
                return {}
                
        except requests.exceptions.ConnectionError:
            print("❌ Error: Could not connect to the server. Make sure it's running on http://localhost:8001")
            return {}
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return {}
    
    def collect_credentials(self, required_credentials: List[str]) -> Dict[str, str]:
        """Collect credentials from user based on agent's requirements"""
        self.print_step("CREDENTIAL COLLECTION")
        print(f"The agent requires {len(required_credentials)} credentials:")
        
        for i, cred in enumerate(required_credentials, 1):
            print(f"  {i}. {cred}")
        
        print("\nPlease provide the following credentials:")
        print()
        
        collected_creds = {}
        
        for cred in required_credentials:
            while True:
                print(f"\n🔑 {cred}")
                
                # Determine input method based on credential name
                if "path" in cred.lower() or "file" in cred.lower():
                    value = input(f"Enter file path for {cred}: ").strip()
                elif "token" in cred.lower() or "key" in cred.lower() or "secret" in cred.lower():
                    value = getpass.getpass(f"Enter {cred} (hidden): ").strip()
                else:
                    value = input(f"Enter {cred}: ").strip()
                
                if value:
                    collected_creds[cred] = value
                    break
                print("Please enter a valid value.")
        
        return collected_creds
    
    def submit_credentials(self, credentials: Dict[str, str]) -> bool:
        """Submit credentials to the server"""
        self.print_step("SUBMITTING CREDENTIALS")
        
        success_count = 0
        
        for cred_key, cred_value in credentials.items():
            print(f"Submitting: {cred_key}")
            
            # Determine credential type
            if cred_key.lower().endswith('_path') or 'file' in cred_key.lower() or 'path' in cred_key.lower():
                cred_type = "file"
            elif cred_key.lower().endswith('_json') or (cred_value.startswith('{') and cred_value.endswith('}')):
                cred_type = "json"
            else:
                cred_type = "text"
            
            cred_data = {
                "session_id": self.session_id,
                "credential_key": cred_key,
                "credential_value": cred_value,
                "credential_type": cred_type
            }
            
            try:
                response = requests.post(f"{self.base_url}/api/submit-credentials", json=cred_data)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"  ✅ {result.get('message', 'Success')}")
                    success_count += 1
                else:
                    print(f"  ❌ Error {response.status_code}: {response.text}")
                    
            except Exception as e:
                print(f"  ❌ Exception: {str(e)}")
            
            time.sleep(0.5)  # Small delay between submissions
        
        print(f"\n📊 Credential submission summary: {success_count}/{len(credentials)} successful")
        return success_count == len(credentials)
    
    def execute_workflow(self, original_request: str) -> Dict:
        """Execute the workflow after credentials are submitted"""
        self.print_step("EXECUTING WORKFLOW")
        print("Sending the same request again to trigger execution...")
        
        chat_data = {
            "message": original_request,
            "session_id": self.session_id,
            "credentials": {}
        }
        
        try:
            response = requests.post(f"{self.base_url}/api/chat", json=chat_data)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Execution Status: {result['status']}")
                
                # Display the agent's response
                print("\n🤖 AGENT RESPONSE:")
                print("-" * 40)
                print(result['response'])
                
                if result.get('agent_execution_result'):
                    self.display_execution_results(result['agent_execution_result'])
                
                return result
            else:
                print(f"❌ Error: {response.status_code}")
                print(response.text)
                return {}
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return {}
    
    def display_execution_results(self, agent_result: Dict):
        """Display agent execution results in a formatted way"""
        print("\n🤖 AGENT EXECUTION RESULTS")
        print("-"*40)
        print(f"Agent Name: {agent_result.get('agent_name', 'N/A')}")
        print(f"Status: {agent_result.get('status', 'N/A')}")
        print(f"Execution Time: {agent_result.get('execution_time', 'N/A')}")
        
        if agent_result.get('steps_executed'):
            print("\n📋 Steps Executed:")
            for step in agent_result['steps_executed']:
                print(f"  {step['step']}. {step['action']}")
                print(f"     Tool: {step['tool']}")
                print(f"     Result: {step['result']}")
                if step.get('output'):
                    print(f"     Output: {step['output']}")
                print()
        
        if agent_result.get('final_output'):
            print(f"🎯 Final Output: {agent_result['final_output']}")
        
        if agent_result.get('error'):
            print(f"❌ Error: {agent_result['error']}")
    
    def run_interactive_test(self):
        """Run the complete interactive test"""
        self.print_header("INTERACTIVE AGENTIC WORKFLOW TESTER")
        print(f"Session ID: {self.session_id}")
        print("This tool will guide you through testing the agentic conversation flow.")
        print("The agent will provide all instructions and guidance.")
        
        # Step 1: Get workflow request
        workflow_request = self.get_user_workflow_request()
        
        # Step 2: Make initial request
        initial_result = self.make_initial_request(workflow_request)
        if not initial_result:
            print("❌ Failed to get initial analysis. Exiting.")
            return
        
        # Step 3: Handle credentials if needed
        if initial_result['status'] == 'awaiting_credentials':
            required_creds = initial_result.get('required_credentials', [])
            
            if required_creds:
                # Collect credentials from user
                credentials = self.collect_credentials(required_creds)
                
                # Submit credentials
                if self.submit_credentials(credentials):
                    # Step 4: Execute workflow
                    execution_result = self.execute_workflow(workflow_request)
                    
                    if execution_result['status'] == 'completed':
                        print("\n🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
                    elif execution_result['status'] == 'error':
                        print("\n❌ WORKFLOW EXECUTION FAILED")
                    else:
                        print(f"\n⚠️ WORKFLOW STATUS: {execution_result['status']}")
                else:
                    print("\n❌ Failed to submit all credentials. Cannot proceed with execution.")
            else:
                print("No credentials required, but status is awaiting_credentials. This might be an error.")
        
        elif initial_result['status'] == 'completed':
            print("\n🎉 WORKFLOW COMPLETED IMMEDIATELY!")
            if initial_result.get('agent_execution_result'):
                self.display_execution_results(initial_result['agent_execution_result'])
        
        else:
            print(f"\n⚠️ Unexpected status: {initial_result['status']}")
        
        print("\n✅ Interactive test completed!")

def main():
    """Main function"""
    print("🚀 Starting Interactive Agentic Workflow Tester")
    print("Make sure the server is running on http://localhost:8001")
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8001/")
        if response.status_code == 200:
            print("✅ Server is running!")
        else:
            print("⚠️ Server responded but with unexpected status")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Please start it with:")
        print("   source .venv/bin/activate")
        print("   export SUPABASE_URL='...' && export SUPABASE_KEY='...' && export ANTHROPIC_API_KEY='...'")
        print("   python3 -m uvicorn main:app --reload --port 8001")
        return
    
    # Run the interactive test
    tester = InteractiveAgentTester()
    tester.run_interactive_test()

if __name__ == "__main__":
    main() 