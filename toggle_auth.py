#!/usr/bin/env python3
"""
Simple script to toggle authentication on/off for NeuroLab API

Usage:
    python toggle_auth.py on    # Enable authentication
    python toggle_auth.py off   # Disable authentication
    python toggle_auth.py       # Show current status
"""

import os
import sys
from pathlib import Path

def get_current_status():
    """Get current authentication status"""
    # Check environment variable first
    env_auth = os.getenv('REQUIRE_AUTH', '').lower()
    if env_auth:
        return env_auth == 'true', 'environment variable'
    
    # Check settings file
    settings_file = Path('src/config/settings.py')
    if settings_file.exists():
        with open(settings_file, 'r') as f:
            content = f.read()
            if "os.getenv('REQUIRE_AUTH', 'true')" in content:
                return True, 'settings file (default true)'
            elif "os.getenv('REQUIRE_AUTH', 'false')" in content:
                return False, 'settings file (default false)'
    
    return False, 'unknown'

def set_auth_status(enable: bool):
    """Set authentication status in settings file"""
    settings_file = Path('src/config/settings.py')
    if not settings_file.exists():
        print(f"Error: Settings file not found at {settings_file}")
        return False
    
    with open(settings_file, 'r') as f:
        content = f.read()
    
    # Replace the line
    if enable:
        # Change 'false' to 'true'
        content = content.replace("os.getenv('REQUIRE_AUTH', 'false')", "os.getenv('REQUIRE_AUTH', 'true')")
    else:
        # Change 'true' to 'false'
        content = content.replace("os.getenv('REQUIRE_AUTH', 'true')", "os.getenv('REQUIRE_AUTH', 'false')")
    
    with open(settings_file, 'w') as f:
        f.write(content)
    
    return True

def main():
    if len(sys.argv) == 1:
        # Show current status
        status, source = get_current_status()
        print(f"Authentication is currently: {'ENABLED' if status else 'DISABLED'}")
        print(f"Source: {source}")
        print("\nUsage:")
        print("  python toggle_auth.py on    # Enable authentication")
        print("  python toggle_auth.py off   # Disable authentication")
        print("\nAlternatively, set environment variable:")
        print("  export REQUIRE_AUTH=true     # Enable")
        print("  export REQUIRE_AUTH=false    # Disable")
        return
    
    command = sys.argv[1].lower()
    
    if command in ['on', 'enable', 'true', '1']:
        if set_auth_status(True):
            print("✅ Authentication ENABLED")
            print("Note: Restart the server for changes to take effect")
        else:
            print("❌ Failed to enable authentication")
    
    elif command in ['off', 'disable', 'false', '0']:
        if set_auth_status(False):
            print("✅ Authentication DISABLED")
            print("Note: Restart the server for changes to take effect")
        else:
            print("❌ Failed to disable authentication")
    
    else:
        print(f"Unknown command: {command}")
        print("Use 'on' or 'off'")

if __name__ == "__main__":
    main()