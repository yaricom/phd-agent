# PhD Agent - Multi-Agent Research System

A comprehensive multi-agent system for deep research using LangGraph API, featuring PDF analysis, web search, AI-powered essay writing, and intelligent data analysis.

## 🚀 Features

### Core Agents
- **PDF Analysis Agent**: Processes and analyzes PDF documents using PyMuPDF
- **Web Search Agent**: Performs intelligent web searches using DuckDuckGo
- **Analyst Agent**: Assesses data relevance and quality using AI
- **Essay Writer Agent**: Generates comprehensive essays from research data
- **Supervisor Agent**: Orchestrates the entire research workflow

### Key Capabilities
- **Local Vector Database**: Uses Milvus for efficient document storage and retrieval
- **Intelligent Data Processing**: Chunks and embeds documents for semantic search
- **Quality Assessment**: AI-powered relevance and credibility scoring
- **Multi-Source Synthesis**: Combines PDF and web sources seamlessly
- **Academic Writing**: Generates well-structured essays with proper citations
- **Workflow Management**: Intelligent orchestration of research steps

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Supervisor    │    │   PDF Agent     │    │  Web Search     │
│     Agent       │    │                 │    │     Agent       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       ▼
         │              ┌─────────────────┐    ┌─────────────────┐
         │              │   Milvus        │    │   Web Content   │
         │              │ Vector Database │    │   Extraction    │
         │              └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Analyst       │    │   Document      │    │   Search        │
│     Agent       │    │   Storage       │    │   Results       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐
│   Essay Writer  │
│     Agent       │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Final Essay   │
│   Output        │
└─────────────────┘
```

## 📋 Prerequisites

- Python 3.11+
- OpenAI API key
- Milvus vector database (local or cloud)
- Docker (optional, for Milvus)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone git@github.com:yaricom/phd-agent.git
   cd phd-agent
   ```

2. **Install with dependencies**

   To use only Python API:
   ```bash
   pip install -e .
   ```

   To use web API as well:
   ```bash
   pip install -e ".[api]"
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start Milvus (using Docker)**
   ```bash
   # Download and start Milvus
   wget https://github.com/milvus-io/milvus/releases/download/v2.5.14/milvus-standalone-docker-compose.yml
   docker-compose -f milvus-standalone-docker-compose.yml up -d
   ```

   Visit Milvus WEB UI
   http://localhost:9091/webui

## 🔧 Development Setup

### Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality. The hooks will automatically run on each commit to:

- Format code with Black
- Lint code with Ruff
- Check for merge conflicts
- Validate YAML/TOML files
- Run type checking with MyPy
- Perform security checks with Bandit

#### Installation

1. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

2. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

3. **Run hooks manually (optional)**
   ```bash
   pre-commit run --all-files
   ```

#### What the hooks do

- **Black**: Formats Python code to a consistent style
- **Ruff**: Fast Python linter that finds and fixes common issues
- **Pre-commit hooks**: Basic file checks (merge conflicts, YAML validation, etc.)
- **MyPy**: Static type checking for Python code
- **Bandit**: Security linting to find common security issues

#### Skipping hooks (if needed)

If you need to skip pre-commit hooks for a specific commit:
```bash
git commit --no-verify -m "Your commit message"
```

### Running Tests

The project includes comprehensive unit tests for all modules.

#### Install test dependencies
```bash
pip install -e ".[dev,formats]"
```

#### Run all tests
```bash
python -m pytest tests/
```

#### Run specific test file
```bash
python -m pytest tests/test_file_utils.py
```

#### Run tests with coverage
```bash
python -m pytest --cov=phd_agent tests/
```

#### Run tests in verbose mode
```bash
python -m pytest -v tests/
```

## ⚙️ Configuration

Create a `.env` file with the following variables:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-turbo-preview

# Milvus Configuration
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=research_documents

# Web Search Configuration
ENABLE_WEB_SEARCH=true
MAX_SEARCH_RESULTS=10
SEARCH_TIMEOUT=30

# System Configuration
MAX_TOKENS_PER_CHUNK=1000
CHUNK_OVERLAP=200
TEMPERATURE=0.7
RELEVANCE_THRESHOLD=0.6
```

## 🚀 Usage

### Command Line Interface

1. **Basic research without PDFs**
   ```bash
   python src/main.py \
     --topic "Artificial Intelligence in Healthcare" \
     --requirements "Analyze current applications and future trends"
   ```

2. **Research with PDF documents only (no web search)**
   ```bash
   python src/main.py \
     --topic "Machine Learning" \
     --requirements "Review recent advances" \
     --pdfs papers/ \
     --no-web-search
   ```

3. **Research with PDF documents and web search**
   ```bash
   python src/main.py \
     --topic "Machine Learning" \
     --requirements "Review recent advances" \
     --pdfs papers/ research/
   ```

4. **Custom configuration**
   ```bash
   python src/main.py \
     --topic "Climate Change" \
     --requirements "Economic impacts" \
     --max-sources 15 \
     --essay-length long \
     --output climate_essay.txt
   ```

### Web API Interface

1. **Start the API server**
   ```bash
   python api.py
   ```

2. **API Endpoints**
   - `POST /research/start` - Start a new research task
   - `GET /research/{task_id}/status` - Get task status
   - `POST /research/{task_id}/run` - Run research workflow
   - `GET /research/{task_id}/essay` - Get final essay
   - `POST /research/{task_id}/upload-pdfs` - Upload PDF files

3. **Example API usage**
   ```bash
   # Start research with web search enabled
   curl -X POST "http://localhost:8000/research/start" \
     -H "Content-Type: application/json" \
     -d '{
       "topic": "Quantum Computing",
       "requirements": "Current state and applications",
       "max_sources": 10,
       "essay_length": "medium",
       "enable_web_search": true
     }'

   # Start research with web search disabled
   curl -X POST "http://localhost:8000/research/start" \
     -H "Content-Type: application/json" \
     -d '{
       "topic": "Quantum Computing",
       "requirements": "Current state and applications",
       "max_sources": 10,
       "essay_length": "medium",
       "enable_web_search": false
     }'

   # Get status
   curl "http://localhost:8000/research/{task_id}/status"

   # Run workflow
   curl -X POST "http://localhost:8000/research/{task_id}/run"
   ```

### Programmatic Usage

```python
from phd_agent.agents.supervisor_agent import SupervisorAgent

# Initialize supervisor
supervisor = SupervisorAgent()

# Run a research workflow
state = supervisor.run(
   topic="Renewable Energy",
   requirements="Analyze solar and wind energy adoption trends",
   max_relevant_sources=12,
   essay_length="long",
   pdf_paths=["papers/solar.pdf", "papers/wind.pdf"]
)

# Access results
if state.final_essay:
   print(f"Essay: {state.final_essay.title}")
   print(f"Word count: {state.final_essay.word_count}")
   print(f"Content: {state.final_essay.content}")
```

### Support for different output formats

The essay can be written to the file using the following output formats TXT, PDF, DOCX. 

To support PDF and DOCX additional libraries should be installed:

```bash
pip install ".[formats]"
```

The output format is detected from the output file name extension defaulting to .txt if not recognized.  

## 📊 Workflow Steps

1. **PDF Processing**: Extracts and chunks PDF documents
2. **Web Search**: Searches for relevant web content (optional, configurable)
3. **Data Analysis**: Assesses relevance and quality of sources
4. **Essay Writing**: Generates comprehensive essay from analyzed data
5. **Output**: Delivers final essay with sources and metadata

## ⚙️ Web Search Configuration

The system supports configurable web search functionality:

### Environment Variable
```env
ENABLE_WEB_SEARCH=true  # Set to false to disable web search
```

### Command Line
```bash
python src/main.py --topic "Your Topic" --no-web-search
```

### API
```json
{
  "topic": "Your Topic",
  "requirements": "Your requirements",
  "enable_web_search": false
}
```

### Use Cases for Disabling Web Search
- **Offline Research**: Work only with local PDF documents
- **Privacy**: Avoid external web requests
- **Speed**: Faster processing without web search delays
- **Cost**: Reduce API usage for web content extraction
- **Compliance**: Meet data privacy requirements

## 🔧 Customization

### Adding New Agents

1. Create a new agent class in `src/phd_agent/agents/`
2. Implement the `run()` method
3. Add the agent to the supervisor workflow

### Custom Data Sources

1. Extend the `DocumentSource` model
2. Create a new agent for your data source
3. Integrate with the vector store

### Custom Analysis

1. Modify the `AnalystAgent` prompts
2. Add new assessment criteria
3. Implement custom filtering logic

## 🧪 Testing

```bash
# Run basic system check
python check_system.py

# Run unit tests (see Development Setup section for more details)
python -m pytest tests/
```

## 📈 Performance

- **Document Processing**: ~1000 words/second
- **Vector Search**: <100ms for 10k documents
- **Essay Generation**: 2-5 minutes depending on length
- **Memory Usage**: ~2GB for typical research tasks

## 🔒 Security

- API keys stored in environment variables
- No sensitive data logged
- Input validation on all endpoints
- Rate limiting for web searches

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: Create an issue on GitHub
- **Documentation**: Check the `/docs` folder
- **Examples**: See `/examples` for usage examples

## 🔮 Roadmap

- [ ] LangGraph integration for advanced workflows
- [ ] Multi-language support
- [ ] Real-time collaboration
- [ ] Advanced citation management
- [ ] Integration with academic databases
- [ ] Mobile app interface

## 🙏 Acknowledgments

- OpenAI for GPT models
- Milvus for vector database
- LangChain for LLM framework
- PyMuPDF for PDF processing
- DuckDuckGo for web search
