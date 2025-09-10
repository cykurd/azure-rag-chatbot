#!/usr/bin/env python3
"""
Simple startup script for the RAG Chatbot
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required dependencies are installed"""
    try:
        import flask
        import openai
        import sklearn
        import numpy
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_api_key():
    """Check if OpenAI API key is set"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Warning: OPENAI_API_KEY not set")
        print("   Set it with: export OPENAI_API_KEY='your-key'")
        print("   The RAG system will not work without an API key")
        return False
    return True

def main():
    """Main startup function"""
    print("=" * 50)
    print("RAG Chatbot")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check API key
    api_key_set = check_api_key()
    
    print("\nStarting chat application...")
    print("URL: http://localhost:5000")
    print("Press Ctrl+C to stop")
    
    if not api_key_set:
        print("\n⚠️  Note: Set OPENAI_API_KEY to enable full functionality")
    
    try:
        # Run the chat application
        subprocess.run([sys.executable, "run_chat.py"])
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
