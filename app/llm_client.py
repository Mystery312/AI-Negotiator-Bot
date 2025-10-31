"""
LLM Client module for handling different providers (Gemini API and Ollama).
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional

# Add parent directory to sys.path if running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try package imports first, then fall back to local imports
try:
    from openai import OpenAI
    import google.generativeai as genai
except ImportError:
    # Fall back to local imports when running as script
    from openai import OpenAI
    import google.generativeai as genai

logger = logging.getLogger(__name__)

class LLMProvider:
    """Base class for LLM providers."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response from the LLM."""
        raise NotImplementedError
    
    def get_available_models(self) -> List[str]:
        """Get list of available models for this provider."""
        raise NotImplementedError

class OllamaProvider(LLMProvider):
    """Ollama provider using OpenAI-compatible API."""
    
    def __init__(self, model_name: str = "qwen3:latest"):
        super().__init__(model_name)
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.client = OpenAI(
            api_key="dummy",  # Ollama doesn't need a real API key
            base_url=f"{self.base_url}/v1"
        )
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using Ollama."""
        try:
            # Use model from kwargs if provided, otherwise use self.model_name
            model = kwargs.pop('model', self.model_name)
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response with Ollama: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Get available models from Ollama."""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                if "models" in models_data:
                    return [model["name"] for model in models_data["models"]]
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {e}")
        
        # Return default models if API call fails
        return ["qwen3:latest", "llama3.2:latest", "mistral:latest", "codellama:latest", "phi3:latest"]

class GeminiProvider(LLMProvider):
    """Google Gemini provider."""
    
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        super().__init__(model_name)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using Gemini."""
        try:
            # Use model from kwargs if provided, otherwise use self.model_name
            model_name = kwargs.pop('model', self.model_name)
            if model_name != self.model_name:
                self.model = genai.GenerativeModel(model_name)
                self.model_name = model_name

            # Convert OpenAI format to Gemini format
            gemini_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    # Gemini doesn't have system messages, prepend to user message
                    continue
                elif msg["role"] == "user":
                    gemini_messages.append({"role": "user", "parts": [msg["content"]]})
                elif msg["role"] == "assistant":
                    gemini_messages.append({"role": "model", "parts": [msg["content"]]})
            
            # Handle system message by prepending to first user message
            if messages and messages[0]["role"] == "system":
                if len(messages) > 1 and messages[1]["role"] == "user":
                    gemini_messages[0]["parts"][0] = f"{messages[0]['content']}\n\n{messages[1]['content']}"
            
            response = self.model.generate_content(gemini_messages)
            return response.text
        except Exception as e:
            logger.error(f"Error generating response with Gemini: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Get available Gemini models."""
        return ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro", "gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"]

class LLMClient:
    """Unified LLM client that can switch between providers."""
    
    def __init__(self, provider: str = "ollama", model_name: str = None):
        """
        Initialize LLM client.
        
        Args:
            provider: "ollama" or "gemini"
            model_name: Model name for the provider
        """
        self.provider_name = provider
        
        if provider == "ollama":
            if model_name is None:
                model_name = "qwen3:latest"
            self.provider = OllamaProvider(model_name)
        elif provider == "gemini":
            if model_name is None:
                model_name = "gemini-1.5-flash"
            self.provider = GeminiProvider(model_name)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using the current provider."""
        return self.provider.generate_response(messages, **kwargs)
    
    def get_available_models(self) -> List[str]:
        """Get available models for the current provider."""
        return self.provider.get_available_models()
    
    def switch_provider(self, provider: str, model_name: str = None):
        """Switch to a different provider."""
        self.__init__(provider, model_name)

def create_llm_client(provider: str = "ollama", model_name: str = None) -> LLMClient:
    """Factory function to create an LLM client."""
    return LLMClient(provider, model_name)

def get_provider_from_model(model_name: str) -> str:
    """Determine provider from model name."""
    if model_name.startswith(("gemini-", "gemini")):
        return "gemini"
    else:
        return "ollama"

def get_available_providers() -> Dict[str, List[str]]:
    """Get available models for all providers."""
    providers = {}
    
    # Try Ollama
    try:
        ollama_client = OllamaProvider()
        providers["ollama"] = ollama_client.get_available_models()
    except Exception as e:
        logger.warning(f"Could not fetch Ollama models: {e}")
        providers["ollama"] = ["qwen3:latest", "llama3.2:latest", "mistral:latest"]
    
    # Try Gemini
    try:
        gemini_client = GeminiProvider()
        providers["gemini"] = gemini_client.get_available_models()
    except Exception as e:
        logger.warning(f"Could not fetch Gemini models: {e}")
        providers["gemini"] = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"]
    
    return providers 