#!/usr/bin/env python3
"""
Simple test script to verify the chatbot application can start.
"""
import os
import sys

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all critical modules can be imported."""
    print("Testing imports...")

    try:
        print("  - Importing negotiation-chatbot.main...")
        from negotiation_chatbot.main import app
        print("    ✓ Success")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return False

    try:
        print("  - Importing negotiation-chatbot.coach...")
        from negotiation_chatbot.coach import get_advice
        print("    ✓ Success")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return False

    try:
        print("  - Importing negotiation-chatbot.ingest...")
        from negotiation_chatbot.ingest import label_text
        print("    ✓ Success")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return False

    try:
        print("  - Importing negotiation-chatbot.llm_client...")
        from negotiation_chatbot.llm_client import create_llm_client
        print("    ✓ Success")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return False

    return True

def test_api_health():
    """Test if the API can respond to health check."""
    print("\nTesting API health endpoint...")

    try:
        from negotiation_chatbot.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/health")

        if response.status_code == 200:
            print(f"  ✓ Health check passed: {response.json()}")
            return True
        else:
            print(f"  ✗ Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Chatbot Application Test Suite")
    print("=" * 60)

    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed!")
        sys.exit(1)

    # Test API health
    if not test_api_health():
        print("\n❌ API health test failed!")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ All tests passed! The application is ready to run.")
    print("=" * 60)
    print("\nTo start the application:")
    print("  1. Start the API server:")
    print("     python -m negotiation_chatbot.main")
    print("\n  2. In another terminal, start the Gradio UI:")
    print("     python -m negotiation_chatbot.gradio_ui")
    print("\nOr use Docker Compose:")
    print("     docker-compose up --build")
    print("=" * 60)

if __name__ == "__main__":
    main()
