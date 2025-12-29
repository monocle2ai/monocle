#!/usr/bin/env python3
"""
Example script demonstrating how to use the Okahu client to list applications.

Usage:
    export OKAHU_API_KEY=your_api_key
    python examples/list_okahu_apps.py
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from monocle_apptrace.exporters.okahu import OkahuClient


def main():
    """Main function to list Okahu apps."""
    try:
        # Check if API key is set
        if not os.environ.get("OKAHU_API_KEY"):
            print("Error: OKAHU_API_KEY environment variable is not set.")
            print("Please set it with: export OKAHU_API_KEY=your_api_key")
            sys.exit(1)
        
        # Create client and list apps
        print("Connecting to Okahu platform...")
        with OkahuClient() as client:
            apps = client.list_apps()
            
            if not apps:
                print("\nNo applications found on Okahu platform.")
                return
            
            print(f"\nFound {len(apps)} application(s):\n")
            print(f"{'App Name':<30} {'App ID':<40} {'Status':<15}")
            print("-" * 85)
            
            for app in apps:
                app_name = app.get('name', 'N/A')
                app_id = app.get('id', 'N/A')
                status = app.get('status', 'N/A')
                print(f"{app_name:<30} {app_id:<40} {status:<15}")
            
            print()
            
            # Optionally get details for the first app
            if apps:
                first_app_id = apps[0].get('id')
                if first_app_id:
                    print(f"\nGetting details for first app: {first_app_id}")
                    app_details = client.get_app(first_app_id)
                    print(f"App details: {app_details}")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
