"""
Command-line interface for Okahu operations.
"""
import argparse
import json
import logging
import sys
from typing import Optional

from monocle_apptrace.exporters.okahu.okahu_client import OkahuClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def list_apps_command(args):
    """List all applications on Okahu platform."""
    try:
        with OkahuClient() as client:
            apps = client.list_apps()
            
            if args.format == 'json':
                print(json.dumps(apps, indent=2))
            else:
                # Pretty print as table
                if not apps:
                    print("No applications found.")
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
                
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to list apps: {e}")
        sys.exit(1)


def get_app_command(args):
    """Get details for a specific application."""
    try:
        with OkahuClient() as client:
            app = client.get_app(args.app_id)
            print(json.dumps(app, indent=2))
            
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to get app details: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Okahu CLI - Manage and query Okahu observability platform',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List apps command
    list_parser = subparsers.add_parser('list-apps', help='List all applications')
    list_parser.add_argument(
        '--format',
        choices=['table', 'json'],
        default='table',
        help='Output format (default: table)'
    )
    list_parser.set_defaults(func=list_apps_command)
    
    # Get app command
    get_parser = subparsers.add_parser('get-app', help='Get application details')
    get_parser.add_argument('app_id', help='Application ID or workflow name')
    get_parser.set_defaults(func=get_app_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()
