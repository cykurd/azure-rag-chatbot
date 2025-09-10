#!/usr/bin/env python3
"""
Start the RAG Chat Application
"""

import os
import sys
from pathlib import Path

def main():
    print("=" * 50)
    print("RAG Chat Application")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("app/chat_app.py").exists():
        print("Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set")
        print("   The RAG system will not work without an API key")
        print("   Set it with: export OPENAI_API_KEY='your-key'")
        print()
    
    print("Starting chat application...")
    print("URL: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print()
    
    # Start the Flask app
    os.chdir("app")
    os.system("python chat_app.py")

if __name__ == "__main__":
    main()
