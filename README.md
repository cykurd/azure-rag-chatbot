# RAG Chatbot

A sophisticated Retrieval-Augmented Generation (RAG) chatbot built with Flask, featuring advanced query processing, document retrieval, and AI-powered responses for document-based question answering.

## Features

### Advanced RAG Capabilities
- **Query Expansion**: Automatically generates query variations to improve retrieval
- **Advanced Reranking**: Uses TF-IDF similarity, token overlap, and title relevance for better document ranking
- **Diversity Selection**: Ensures diverse document selection to avoid redundant information
- **Multiple Retrieval Methods**: Supports both enhanced and default retrieval modes

### Technical Features
- **Clean Architecture**: Modular design with separate components for models, RAG systems, and evaluation
- **Multiple LLM Support**: Compatible with GPT-4o, GPT-4o-mini, GPT-3.5-turbo, and other OpenAI models
- **Flexible Configuration**: Configurable system prompts, retrieval parameters, and model settings
- **Error Handling**: Robust error handling and fallback mechanisms
- **Logging**: Comprehensive logging for debugging and monitoring

### Web Interface
- **Simple Chat Interface**: Clean, ChatGPT-like interface
- **Real-time Responses**: Fast, responsive chat experience
- **Status Monitoring**: Built-in system status and health checks

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/cykurd/rag-chatbot.git
   cd rag-chatbot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

4. **Run the application**:
   ```bash
   python run_chat.py
   ```

5. **Access the web interface**:
   Open your browser and go to `http://localhost:5000`

## Usage

### Basic Usage
1. Start the chat application using `python run_chat.py`
2. Open your browser to `http://localhost:5000`
3. Type your questions in the chat interface
4. The system will retrieve relevant documents and generate responses

### Advanced Configuration

#### Model Configuration
```python
from src.models.base_model import BaseModel

# Create an enhanced model with advanced RAG capabilities
model = BaseModel(
    model_name="gpt-4o-mini",
    retrieval_method="enhanced",
    query_expansion=True,
    advanced_reranking=True,
    diversity_selection=True,
    system_prompt_variant="current"
)
```

#### RAG System Configuration
```python
from src.rag_systems.simple_rag_web import SimpleRAGWeb, Document

# Create documents
documents = [
    Document(
        id="1",
        content="Your document content here",
        metadata={"title": "Document Title", "source": "Source"}
    )
]

# Initialize RAG system
rag_system = SimpleRAGWeb(model, documents)
```

### System Prompt Variants
- `current`: Standard assistant with document-based responses
- `basic`: Simple assistant for basic queries
- `detailed`: Advanced assistant with comprehensive analysis
- `concise`: Direct and concise responses
- `analytical`: In-depth analysis with pattern recognition
- `educational`: Educational responses with clear explanations

## Project Structure

```
rag-chatbot/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── base_model.py          # Advanced LLM model with RAG capabilities
│   ├── rag_systems/
│   │   ├── __init__.py
│   │   └── simple_rag_web.py      # RAG system implementation
│   └── evaluation/
│       ├── __init__.py
│       └── rag_evaluator.py       # RAG evaluation framework
├── app/
│   ├── templates/
│   │   └── chat.html              # Chat interface template
│   ├── static/
│   │   ├── css/
│   │   │   └── chat.css           # Chat interface styles
│   │   └── js/
│   │       └── chat.js            # Chat interface JavaScript
│   ├── requirements.txt           # App-specific dependencies
│   └── chat_app.py                # Flask chat application
├── data/
│   ├── sample_documents.json      # Sample documents for testing
│   └── test_cases.json            # Test cases for evaluation
├── requirements.txt               # Main project dependencies
├── run_chat.py                    # Chat application launcher
└── README.md                      # This file
```

## API Endpoints

### Chat Endpoints
- `GET /`: Main chat interface
- `POST /api/chat`: Send chat messages
- `GET /api/status`: Get system status

### Example API Usage
```bash
# Send a chat message
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is machine learning?"}'

# Check system status
curl http://localhost:5000/api/status
```

## Advanced Features

### Query Expansion
The system automatically expands queries to improve retrieval:
- Removes question words ("what", "how", "when", etc.)
- Extracts key terms by removing stopwords
- Generates multiple query variations

### Advanced Reranking
Documents are reranked using multiple signals:
- TF-IDF similarity (40% weight)
- Token overlap (30% weight)
- Title relevance (20% weight)
- Original retrieval score (10% weight)

### Diversity Selection
The system ensures diverse document selection:
- Balances relevance and diversity
- Prevents redundant information
- Uses semantic similarity to avoid duplicates

## Configuration Options

### Model Parameters
- `model_name`: LLM model to use
- `temperature`: Generation temperature
- `top_p`: Top-p sampling parameter
- `max_tokens`: Maximum tokens to generate
- `seed`: Random seed for reproducibility

### Retrieval Parameters
- `retrieval_method`: "enhanced" or "default"
- `query_expansion`: Enable/disable query expansion
- `advanced_reranking`: Enable/disable advanced reranking
- `diversity_selection`: Enable/disable diversity selection
- `initial_retrieval_k`: Number of documents to retrieve initially
- `final_k`: Final number of documents to use

## Development

### Running Tests
```bash
pytest tests/
```

### Adding New Documents
```python
# Add documents to the system
new_docs = [
    Document(
        id="new_doc_1",
        content="New document content",
        metadata={"title": "New Document", "source": "New Source"}
    )
]
rag_system.add_documents(new_docs)
```

### Custom System Prompts
```python
# Set custom system prompt
model.set_system_prompt_variant("analytical")

# Or use a custom prompt
response = model.generate(
    prompt="Your question here",
    system_prompt="Custom system prompt"
)
```

## Troubleshooting

### Common Issues

1. **API Key Not Set**:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

2. **Dependencies Missing**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Port Already in Use**:
   ```bash
   # Change port in run_chat.py or kill existing process
   lsof -ti:5000 | xargs kill -9
   ```

### Logs
Check the console output for detailed error messages and system status.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with OpenAI's GPT models
- Uses scikit-learn for text processing
- Flask for web framework