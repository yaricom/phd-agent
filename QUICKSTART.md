# Quick Start Guide - PhD Agent

Get up and running with the multi-agent research system in minutes!

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment
```bash
# Create .env file
cp .env.example .env

# Edit .env and add your OpenAI API key
echo "OPENAI_API_KEY=your_actual_api_key_here" >> .env
```

### 3. Test the System
```bash
python test_system.py
```

### 4. Run Your First Research
```bash
python src/main.py \
  --topic "Machine Learning Applications" \
  --requirements "Analyze current trends and future directions"
```

## 📋 Prerequisites

- **Python 3.8+**
- **OpenAI API Key** (required for AI features)
- **Docker** (optional, for Milvus vector database)

## 🔧 Optional: Set Up Milvus

For better performance with large document collections:

```bash
# Download Milvus
wget https://github.com/milvus-io/milvus/releases/download/v2.5.14/milvus-standalone-docker-compose.yml

# Start Milvus
docker-compose -f milvus-standalone-docker-compose.yml up -d
```

## 🎯 Usage Examples

### Basic Research
```bash
python src/main.py \
  --topic "Climate Change" \
  --requirements "Economic impacts and mitigation strategies"
```

### Research with PDFs
```bash
python src/main.py \
  --topic "Quantum Computing" \
  --requirements "Current state and applications" \
  --pdfs papers/quantum.pdf research/
```

### Custom Configuration
```bash
python src/main.py \
  --topic "AI in Healthcare" \
  --requirements "Ethical considerations and implementation challenges" \
  --max-sources 15 \
  --essay-length long \
  --output healthcare_ai_essay.txt
```

### Web API
```bash
# Start the API server
python src/phd_agent/api.py

# Use the API
curl -X POST "http://localhost:8000/research/start" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Blockchain Technology",
    "requirements": "Applications beyond cryptocurrency",
    "max_sources": 10,
    "essay_length": "medium"
  }'
```

## 📊 What You Get

1. **PDF Analysis**: Automatic extraction and processing of PDF documents
2. **Web Search**: Intelligent web content discovery and extraction
3. **Data Analysis**: AI-powered relevance and quality assessment
4. **Essay Generation**: Comprehensive essays with proper citations
5. **Vector Storage**: Efficient document storage and retrieval

## 🔍 System Components

- **Supervisor Agent**: Orchestrates the entire workflow
- **PDF Agent**: Processes PDF documents
- **Web Search Agent**: Performs web searches
- **Analyst Agent**: Assesses data quality
- **Essay Writer Agent**: Generates final essays

## 🛠️ Troubleshooting

### Common Issues

1. **"OpenAI API key not configured"**
   - Add your API key to the `.env` file

2. **"Milvus connection failed"**
   - The system will use a mock vector store automatically
   - For better performance, start Milvus with Docker

3. **"Module not found"**
   - Run `pip install -r requirements.txt`

4. **"Permission denied"**
   - Check file permissions and Python environment

### Getting Help

- Check the full [README.md](README.md) for detailed documentation
- Run `python test_system.py` to diagnose issues
- See [examples/](examples/) for more usage examples

## 🎉 Next Steps

1. **Explore Examples**: Check out `examples/basic_research.py`
2. **Customize Agents**: Modify agent behavior in `agents/`
3. **Add Data Sources**: Extend the system with new document types
4. **Scale Up**: Deploy with proper Milvus setup for production use

## 📈 Performance Tips

- Use Milvus for better vector search performance
- Adjust `max_sources` based on your needs
- Use `--essay-length` to control output size
- Monitor API usage with OpenAI

---

**Ready to start researching?** Run `python src/main.py --help` to see all options! 