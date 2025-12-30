# INTELLISEARCH 🔍

An advanced AI-powered research pipeline that conducts comprehensive web searches, analyzes content, and generates detailed reports using LangGraph workflows and Google Gemini AI.

## 🌟 Features

- **Google Gemini AI**: Simplified single-provider architecture using Google's latest AI models
- **Advanced Web Scraping**: Multiple strategies including requests-html, aiohttp, and fallback methods
- **Intelligent Content Analysis**: AI-powered relevance evaluation and ranking
- **Vector Search**: Google GenerativeAI embeddings for semantic search
- **Flexible Reports**: Configurable word limits (600-1200 for concise, 800-3000 for detailed)
- **PDF Generation**: Automatic PDF report creation with fpdf2
- **Windows Automation**: One-click batch file setup and execution
- **Clean Architecture**: Streamlined codebase with unified Google AI integration

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ (tested with Python 3.13.7)
- Windows OS (batch files included)
- Google API Key for Gemini AI
- Serper API Key for web search

### Execution Options

**📋 See [docs/EXECUTION_GUIDE.md](docs/EXECUTION_GUIDE.md) for detailed execution options**

#### **Option 1: Complete Setup & Interactive Mode (Recommended for first time)**
```batch
run_interactive.bat
```
- Handles complete environment setup
- Interactive mode with full control
- Best for first-time users

#### **Option 2: Quick Automated Research (Fast execution)**
```batch
run_automated.bat
```
- Fast automated research
- No user prompts during workflow
- Requires environment already set up

#### **Option 3: Command Line Interface (Advanced)**
```bash
# Interactive mode
python app.py --interactive

# Automated with custom settings
python app.py "your research query" --automation full --prompt-type legal

# Batch processing
python app.py --batch-file queries.txt --automation full
```

## 🌐 Web Application

INTELLISEARCH also includes a modern web-based interface built with FastAPI and React:

### **Quick Start Web App**
```bash
# Start both frontend and backend
cd web-app
./start-dev.bat  # Windows
./start-dev.sh   # Linux/Mac
```

**Access the web application:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/api/docs

### **Web Features**
- 🖥️ Modern React + TypeScript interface
- 🚀 Real-time research progress tracking
- 📊 Interactive report generation
- 🔐 API key management
- 📱 Responsive design with Tailwind CSS
- ☁️ Ready for Render.com deployment

**See [web-app/README.md](web-app/README.md) for detailed web application documentation.**

### Installation

1. **Clone this repository**
   ```bash
   git clone <your-repo-url>
   cd INTELLISEARCH
   ```

2. **Run the automated setup**
   ```batch
   run_intellisearch_clean.bat
   ```
   
   This will:
   - Create a virtual environment
   - Install all dependencies
   - Set up configuration
   - Run the application

### Manual Setup

1. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**
   Create a `.env` file:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   SERPER_API_KEY=your_serper_api_key_here
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

## 📁 Project Structure

```
INTELLISEARCH/
├── app.py                          # Main application entry point
├── requirements.txt                # Python dependencies (Google AI only)
├── .env                           # API keys (create this)
├── .gitignore                     # Git ignore rules
├── run_automated.bat              # Automated execution
├── run_interactive.bat            # Interactive mode launcher
├── setup.py                       # Setup utilities
├── startup_validation.py          # Environment validation
├── src/                           # Source code
│   ├── api_keys.py               # API key management (Google only)
│   ├── automation_config.py      # Automation configuration
│   ├── conditions.py             # Workflow conditions
│   ├── config.py                 # Configuration settings
│   ├── data_types.py             # Data structures and models
│   ├── graph.py                  # LangGraph workflow definition
│   ├── import_validator.py       # Import validation utilities
│   ├── llm_calling.py            # LLM interaction utilities
│   ├── llm_utils.py              # LLM utilities (Google GenAI)
│   ├── nodes.py                  # Workflow node implementations
│   ├── prompt.py                 # Prompt templates
│   ├── scraper.py                # Web scraping utilities
│   ├── search.py                 # Search engine integration
│   └── utils.py                  # General utilities (PDF, ranking, etc.)
├── tests/                         # Test files
│   └── test_workflow.py          # Workflow tests
└── docs/                          # Documentation and support files
    ├── CURRENT_DATE_CONTEXT_IMPLEMENTATION.md  # Date context documentation
    ├── EXECUTION_GUIDE.md         # Detailed execution guide
    ├── requirements_original.txt  # Original requirements backup
    ├── IntelliSearchReport.txt    # Sample report output
    ├── IntelliSearchReport.pdf    # Sample PDF report
    ├── intellisearch_data_flow.png # Data flow diagram
    └── intellisearch_workflow_graph.png # Workflow visualization
```

## 🛠️ Configuration

### Report Types
- **Concise Report**: 600-1200 words, focused summary
- **Detailed Report**: 800-3000 words, comprehensive analysis

### AI Provider
- **Google Gemini**: Unified AI provider for both LLM and embeddings
  - LLM Model: `gemini-2.0-flash-lite`
  - Embeddings: `models/text-embedding-004`

### Search Configuration
- **Serper API**: Primary search provider
- **Google Custom Search**: Optional alternative
- **Fallback methods**: Built-in alternatives when APIs unavailable

## 🔧 Advanced Usage

### Custom Prompts
The system supports multiple prompt types:
- `general`: Standard research queries
- `legal`: Legal document analysis
- `macro`: Economic and market research
- `deepsearch`: Deep investigative research
- `person_search`: People and biography research

### API Configuration
Set your Google AI keys in `.env`:
```env
# Google AI (required)
GOOGLE_API_KEY=your_google_key_here

# Search provider (required)  
SERPER_API_KEY=your_serper_key_here
```

## 🐛 Troubleshooting

### Common Issues

1. **Package Installation Failures**
   - Use `run_interactive.bat` for automated resolution
   - Python 3.13 compatibility ensured

2. **API Key Issues**
   - Verify `.env` file exists and contains valid Google and Serper keys
   - Check key formats match Google API requirements

3. **Web Scraping Failures**
   - Application includes multiple fallback methods
   - Check internet connectivity and firewall settings

### Python 3.13 Compatibility
All packages have been updated for Python 3.13 compatibility:
- `pydantic>=2.8.0`
- `requests-html>=0.10.0`
- `ratelimit>=2.2.1` (replaced problematic ratelimiter)

## 📊 Performance

- **Concurrent Processing**: Async/await throughout
- **Rate Limiting**: Built-in API rate limiting
- **Caching**: Intelligent content caching
- **Resource Management**: Proper session cleanup

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain/LangGraph**: Workflow orchestration
- **Google Gemini**: AI provider for LLM and embeddings
- **Serper API**: Web search capabilities
- **Python Community**: Open source packages and tools

## 📞 Support

For issues and questions:
1. Check the `docs/` folder for detailed documentation
2. Review troubleshooting section above
3. Open an issue in this repository

---

**Made with ❤️ and AI** - Combining the power of Google Gemini AI for comprehensive research automation.

## Local startup (Windows Playwright-safe)

If you run the backend on Windows and use Playwright, ensure the Windows Proactor
event loop policy is set before Playwright starts. Two safe options are provided:

- Run the bootstrap script (recommended):
```powershell
python run.py
```

- Or ensure this repo root is on `PYTHONPATH` and let Python import `sitecustomize.py` automatically:
```powershell
# $env:PYTHONPATH = "C:\path\to\deepsearch-1"
# python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

`run.py` sets `asyncio.WindowsProactorEventLoopPolicy()` on Windows before starting Uvicorn.
`sitecustomize.py` does the same automatically when the repo root is on `PYTHONPATH`.