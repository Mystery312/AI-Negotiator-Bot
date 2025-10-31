# LLM Provider Support

This application now supports multiple LLM providers for negotiation advice generation.

## Supported Providers

### 1. Ollama (Local)
- **Models**: qwen3:latest, llama3.2:latest, mistral:latest, codellama:latest, phi3:latest
- **Setup**: Requires Ollama running locally (default: http://localhost:11434)
- **Environment Variable**: `OLLAMA_BASE_URL` (optional, defaults to http://localhost:11434)

### 2. Google Gemini (Cloud)
- **Models**: gemini-1.5-flash, gemini-1.5-pro, gemini-1.0-pro
- **Setup**: Requires Google API key
- **Environment Variable**: `GOOGLE_API_KEY` (required)

## Usage

### In the Gradio UI
1. Select a model from the dropdown
2. Models are prefixed with their provider:
   - `ollama:qwen3:latest` - Uses Ollama with Qwen3 model
   - `gemini:gemini-1.5-flash` - Uses Gemini with Flash model

### Programmatically
```python
from app.llm_client import create_llm_client

# Create Ollama client
ollama_client = create_llm_client("ollama", "qwen3:latest")

# Create Gemini client
gemini_client = create_llm_client("gemini", "gemini-1.5-flash")

# Generate responses
messages = [
    {"role": "system", "content": "You are a negotiation coach."},
    {"role": "user", "content": "Provide advice for this situation..."}
]

response = ollama_client.generate_response(messages)
```

## API Endpoints

The `/chat` endpoint now accepts:
- `model`: The model name (e.g., "qwen3:latest", "gemini-1.5-flash")
- `provider`: The provider name (e.g., "ollama", "gemini")

If `provider` is not specified, it will be automatically determined from the model name.

## Setup Instructions

### For Ollama
1. Install Ollama: https://ollama.ai/
2. Pull desired models:
   ```bash
   ollama pull qwen3:latest
   ollama pull llama3.2:latest
   ```
3. Start Ollama service
4. Set `OLLAMA_BASE_URL` if using a different URL

### For Gemini
1. Get a Google API key from Google AI Studio
2. Set environment variable:
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```
3. The application will automatically detect available Gemini models

## Model Selection Logic

The application automatically:
1. Detects available models from both providers
2. Combines them into a single dropdown with provider prefixes
3. Falls back to default models if providers are unavailable
4. Handles provider-specific API differences transparently

## Error Handling

- If Ollama is unavailable, falls back to default Ollama models
- If Gemini API key is missing, Gemini models won't be available
- If both providers fail, uses hardcoded default models
- All errors are logged for debugging

## Testing

Run the test script to verify both providers work:
```bash
python test_llm_client.py
```

This will test:
- Provider detection from model names
- Available model listing
- Ollama client functionality
- Gemini client functionality (if API key is set) 