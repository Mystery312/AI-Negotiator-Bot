# Chat Negotiator

A multi-party negotiation system with AI assistance, real-time analysis, and Pareto-optimal proposal generation.

Abstract: 
 Although there has been many discussion about the philosophical and impacts of AI’s being implemented into a negotiation setting, realistic development to actually apply LLM’s to evaluate and help users navigate negotiation settings have not been addressed.
In this project, we navigate the task of using the prisoner’s dilemma and fundamental game theory ideas, such as the Prisoner's Dilemma, Nash Equilibrium, and the Iterated Prisoner's Dilemma (IPD), to provide goal-oriented and actionable feedback, along with analyzing the overall metrics in a negotiation.

The suggested system stores conversation-based data in the graph database Neo4j to represent the complex relationship between interacting parties in the conversation. The coaching algorithm uses a tic for tat system to build up intentions to cooperate, real-time situation responses, and long-term trust. The algorithm uses a gradio based interface to allow users to access chat history, visual graphs based on negotiations, statistical analyses and direct/intuitive coaching messages. 

After testing, results show that the model showed meaningful attempts at negotiation between parties. This has allowed for consideration of future implication in areas like the simulation economic interactions with material objects, buisness settings where understanding moves between real-time negotiations is key, and even in the use of daily interactions. This paper shows how AI can be used as a strategical negotiation and decision making tool in the future in order to overcome logical traps like the prisoner’s dilemma, but not necessarily as a replacement for human judgement.



> For a deep dive into the architecture and methods, see [AI Negotiation System: Architecture, Methods, and Usage](docs/AI_NEGOTIATION_SYSTEM.md).

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker and Docker Compose (for full stack)
- Ollama (for local LLM models)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd chat-negotiator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Set up data directory (if using DoND samples)**
   ```bash
   # Option 1: Set environment variable
   export DOND_DATA_DIR="/path/to/your/data/directory"
   
   # Option 2: Place data files in expected location
   # Create directory: deal_or_no_dialog/exported/
   # Add files: train.jsonl, validation.jsonl, test.jsonl
   ```

## 🏗️ Building and Running

### Option 1: Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up --build

# Access the UI at http://localhost:7860
```

### Option 2: Local Development

1. **Start the API server**
   ```bash
   cd app
   python api_server.py
   ```

2. **Start Ollama (if using local models)**
   ```bash
   ollama serve
   # Pull a model: ollama pull qwen3:latest
   ```

3. **Launch the Gradio UI**
   ```bash
   cd app
   python gradio_ui.py
   ```

4. **Access the interface**
   - UI: http://localhost:7860
   - API: http://localhost:8000

## 🎯 How to Use the UI

### Basic Usage

1. **Start a Conversation**
   - The UI automatically generates a conversation ID
   - You can customize the ID or leave it auto-generated

2. **Select AI Model**
   - Choose from available Ollama or Gemini models
   - Models are automatically detected and listed

3. **Set Up Negotiation**
   - Enter names for both parties
   - Use the role selector to switch between "You" and "Other Party"

4. **Start Chatting**
   - Type messages in the input field
   - Press Enter or click "Send"
   - The system automatically provides:
     - Coach advice for strategic guidance
     - Bot proposals when simulating the other party

### Advanced Features

#### DoND Sample Loading
- Use the slider to load Deal-or-No-Deal validation samples
- Samples include pre-configured negotiation scenarios
- Item counts are displayed in the Statistics panel

#### Conversation Inspector
- View raw conversation history
- See API-generated statistics
- Monitor conversation dynamics

#### Tools & Exports
- **New Conversation**: Start fresh
- **Export Markdown**: Save conversation as markdown file
- **Update Visualizations**: Generate graphs and statistics

#### Model & API Management
- **Refresh Models**: Update available model list
- **Test Ollama**: Verify Ollama connection
- **Check API Status**: Monitor API health

### Auto-Proposal Feature

When you send a message as "Other Party" (role B), the system automatically:

1. **Estimates Preferences**: Analyzes conversation history to infer both parties' preferences
2. **Generates Pareto-Optimal Proposal**: Uses the `best_offer` function to find optimal allocations
3. **Displays Results**: Shows what each party gets in a clear format

Example output:
```
🤖 Bot proposal:
• You get: item0: 2, item1: 1
• They get: item0: 1, item1: 1, item2: 1
```

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_pareto.py

# Run with coverage
python -m pytest --cov=app tests/
```

### Integration Tests

```bash
# Test API endpoints
python -m pytest tests/test_api.py

# Test UI functionality
python -m pytest tests/test_ui.py
```

### Manual Testing

1. **Test Basic Chat**
   ```bash
   # Start the UI and try sending messages
   cd app
   python gradio_ui.py
   ```

2. **Test Auto-Proposal**
   - Send a message as "Other Party"
   - Verify bot proposal appears
   - Check proposal formatting

3. **Test DoND Sample Loading**
   - Move the sample slider
   - Verify conversation loads
   - Check item counts display

## 🐛 Debugging

### Common Issues

#### 1. Import Errors
```bash
# If you get "No module named 'app'"
export PYTHONPATH="${PYTHONPATH}:/path/to/chat-negotiator"
```

#### 2. Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve
```

#### 3. API Connection Issues
```bash
# Check API health
curl http://localhost:8000/health

# Check API logs
docker-compose logs api
```

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python gradio_ui.py
```

### Troubleshooting

#### UI Not Loading
1. Check if port 7860 is available
2. Verify all dependencies are installed
3. Check browser console for errors

#### Auto-Proposal Not Working
1. Verify `autoplay.py` is in the app directory
2. Check preference estimation model is loaded
3. Look for errors in the browser console

#### Model Loading Issues
1. Check Ollama is running and accessible
2. Verify model names are correct
3. Check network connectivity for Gemini models

### Logs and Monitoring

#### View Logs
```bash
# Docker logs
docker-compose logs -f

# Application logs
tail -f app/logs/app.log
```

#### Debug Information
The UI includes several debug panels:
- **Debug / Status**: Shows API latency and errors
- **Conversation Inspector**: Raw conversation data
- **Model & API**: Connection status and model availability

## 📁 Project Structure

```
chat-negotiator/
├── app/
│   ├── gradio_ui.py          # Main UI interface
│   ├── autoplay.py           # Auto-proposal functionality
│   ├── pareto.py             # Pareto optimization
│   ├── preference.py         # Preference estimation
│   ├── coach.py              # AI coaching logic
│   ├── style.css             # UI styling
│   └── data/                 # Conversation storage
├── scripts/
│   └── simulate_dond.py      # Simulation scripts
├── tests/                    # Test files
├── docker-compose.yml        # Docker configuration
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 📄 app/ Directory File Descriptions

- **automate.py**  
  Automated conversation system for negotiation and coach advice. Orchestrates negotiation sessions using bot proposals and advice.

- **autoplay.py**  
  Implements auto-proposal functionality for the negotiation bot, including generating and formatting bot offers.

- **build_vector_db.py**  
  Script to build a vector database (ChromaDB) from negotiation-related documents (PDF/TXT) using embeddings for retrieval-augmented generation.

- **casino_rag.py**  
  Implements Retrieval-Augmented Generation (RAG) for negotiation using the CaSiNo corpus and ChromaDB, with embedding and search utilities.

- **coach.py**  
  Provides AI coaching logic, including advice generation for negotiation turns, using models and external APIs.

- **dond_data.py**  
  Utilities for loading and managing Deal-or-No-Deal (DoND) negotiation datasets, including train/validation/test splits.

- **gradio_ui.py**  
  Main Gradio-based user interface for multi-party negotiation, chat, visualization, and simulation.

- **graph.py**  
  Handles graph-based storage and analysis of negotiation turns and outcomes, using Neo4j for relationship modeling.

- **ingest.py**  
  Data ingestion and labeling utilities, including PDF extraction and upserting negotiation turns into the graph database.

- **llm_client.py**  
  Abstraction layer for interacting with different LLM providers (Ollama, Gemini, OpenAI), handling API calls and model selection.

- **main.py**  
  FastAPI server entry point, exposing negotiation, advice, and RAG endpoints for the UI and automation scripts.

- **pareto.py**  
  Pareto-utility helpers for Deal-or-No-Dialog, including best-offer computation and allocation enumeration.

- **preference.py**  
  Defines the `PreferenceEstimator` neural network for estimating negotiator preferences from text using transformers.

- **rag.py**  
  Retrieval-Augmented Generation (RAG) utilities for negotiation, including embedding, search, and context retrieval.

- **run_experiments.py**  
  Script for running automated negotiation experiments, evaluating strategies and preference estimation.

- **simulate_dond.py**  
  Runs bot-vs-bot simulations on the DoND dataset, comparing Pareto and baseline strategies for negotiation outcomes.

- **style.css**  
  Custom CSS for the Gradio UI, styling chat bubbles, panels, and visualizations.

- **train_prefs.py**  
  Script for fine-tuning the `PreferenceEstimator` model on the DoND dataset, with training and evaluation routines.

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
API_BASE_URL=http://api:8000
OLLAMA_BASE_URL=http://ollama:11434

# Logging
LOG_LEVEL=INFO

# Model Configuration
DEFAULT_MODEL=ollama:qwen3:latest
```

### Model Configuration

The system supports multiple model providers:

- **Ollama**: Local models (qwen3, llama3.2, mistral, etc.)
- **Gemini**: Google's models (gemini-1.5-pro, gemini-2.0-flash, etc.)

Models are automatically detected and listed in the UI dropdown.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Support

For issues and questions:
1. Check the debugging section above
2. Review the logs for error messages
3. Open an issue on GitHub with:
   - Description of the problem
   - Steps to reproduce
   - Error messages/logs
   - System information

## Gradio UI: Detailed Usage & Features

### Launching the Gradio UI

- **Docker Compose (Recommended):**
  ```bash
  docker-compose up --build
  # Access at http://localhost:7860
  ```
- **Local Development:**
  ```bash
  cd app
  python gradio_ui.py
  # Access at http://localhost:7860
  ```

### Main Features Overview

| Feature | Description |
|---------|-------------|
| **Conversation ID** | Each negotiation session has a unique ID. Use the auto-generated one or enter your own to resume a previous session. |
| **AI Model Selection** | Choose from available models (Ollama or Gemini). The dropdown lists all detected models, and you can enter a custom model name if needed. |
| **Negotiator Names** | Set the display names for yourself and the other party. These names appear in the chat bubbles. |
| **Role Selector** | Switch between sending messages as "You" or "Other Party". This is important for simulating both sides of a negotiation. |
| **Chat Interface** | Type and send messages. The system will: <ul><li>Show your message in the chat</li><li>Provide AI "Coach" advice after each message</li><li>Auto-generate a bot proposal when you send as "Other Party"</li></ul> |
| **DoND Conversation Visualizer** | (Accordion panel) Load and analyze real Deal-or-No-Deal negotiation samples. Includes: <ul><li>Sample selector</li><li>Item counts and speaker stats</li><li>Coach advice for the sample</li><li>Message timeline and analysis plots</li></ul> |
| **Pareto Coach Effectiveness Simulator** | (Accordion panel) Run simulations to see how the AI coach improves negotiation outcomes compared to various baselines. Results and transcripts are displayed. |
| **Export Conversation** | Export the current conversation as a Markdown file for sharing or record-keeping. |
| **Visualizations** | Generate and view negotiation graphs and statistics. |
| **Model Management** | Refresh the list of available models if you add new ones to Ollama or Gemini. |
| **API/Model Status** | (Hidden in UI, but available in code) Check the health of the API and model connections. |

### Step-by-Step Workflow

1. **Start the UI** and open it in your browser.
2. **Enter names** for both parties.
3. **Select your AI model** (e.g., `ollama:qwen3:4b`).
4. **Chat as yourself** or switch to "Other Party" to simulate both sides.
5. **Review coach advice** and bot proposals as you negotiate.
6. **Use the DoND Visualizer** to load and analyze real negotiation samples.
7. **Run the Pareto Coach Simulator** to see how the AI coach impacts outcomes.
8. **Export your conversation** when done.

### Advanced Features

- **Auto-Proposal:** When you send a message as "Other Party", the system estimates preferences and generates a Pareto-optimal proposal using the negotiation history.
- **Conversation Inspector:** (Removed from UI, but code supports it) Lets you view raw conversation data and API-generated stats.
- **Debug/Status Panels:** (Hidden) For advanced troubleshooting.

- The UI is designed for both real negotiations and simulation/testing.
- You can refresh the model list if you add new models to Ollama or Gemini while the UI is running.
- Exporting conversations is useful for sharing negotiation transcripts or for research.

