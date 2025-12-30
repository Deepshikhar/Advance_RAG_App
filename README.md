# 📚 Advanced RAG Application

A comprehensive Retrieval-Augmented Generation (RAG) application built with Streamlit, featuring multiple document processing options, chunking strategies, embedding methods, and RAG retrieval techniques.

## 🚀 Try it Now!
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://advance-rag-app.streamlit.app/)

## 🎥 Demo

<div align="center">
  
### 📽️ Watch the Full Demo
  
[![RAG Application Demo](https://img.shields.io/badge/▶️_Watch_Full_Demo-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://drive.google.com/file/d/1IedTiKoRu7QMzlh5JsqcpZuMfOR7zdoe/view?usp=sharing)

*Click the button above to watch the demo video on Google Drive*

</div>

## ✨ Features

### 📁 **Document Processing**
- **Multi-format Support**: Upload and process TXT, PDF, and CSV files
- **Automatic Organization**: Files are automatically categorized and stored in organized directories
- **Batch Processing**: Upload and process multiple files simultaneously
- **Persistent Storage**: Files are saved locally for future use

### 🔪 **Multiple Chunking Strategies**
- **Recursive Character Splitter**: Best for general text documents
- **Character Splitter**: Simple character-based chunking
- **Token Splitter**: Token-based chunking for consistent sizes
- **Spacy Splitter**: NLP-aware chunking using sentence boundaries

### 🔤 **Embedding & Vector Store**
- **Sentence Transformer**: Uses `all-MiniLM-L6-v2` for high-quality embeddings
- **FAISS Vector Store**: Fast and efficient similarity search
- **Local Persistence**: Vector store is saved locally for faster reloads
- **Custom Vector Store Class**: Modular and extensible design

### 🤖 **LLM Integration**
- **Groq API Integration**: High-performance LLM inference
- **Multiple Models**: Choose from various Groq models:

### 🔍 **RAG Retrieval Methods**
- **Simple RAG**: Basic retrieval and generation with concise answers
- **Enhanced RAG**: Improved prompting with structured responses
- **Advanced RAG**: Document citations and reference tracking

### 🎨 **User Interface**
- **Streamlit-based UI**: Clean, intuitive, and responsive interface
- **Real-time Processing**: Live updates and progress indicators
- **Sidebar Configuration**: All settings accessible from sidebar
- **Context Viewing**: Expand to see retrieved context chunks

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Groq API key (get one at [groq.com](https://console.groq.com))

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/rag-application.git
cd rag-application
```

2. **Create a virtual environment** (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install Spacy model** (if using spacy splitter)
```bash
python -m spacy download en_core_web_sm
```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📖 Usage Guide

### Step 1: Upload Documents
1. Click "Browse files" in the File Upload section
2. Select one or more TXT, PDF, or CSV files
3. Click "Save and Load Files" to process them

### Step 2: Configure Settings (Sidebar)
1. **Groq API Key**: Enter your API key (or set in .env)
2. **Select LLM Model**: Choose from available Groq models
3. **Document Processing**: Adjust chunk size and overlap
4. **Split Method**: Select your preferred chunking strategy
5. **RAG Type**: Choose between Simple, Enhanced, or Advanced RAG
6. **Top K**: Set number of chunks to retrieve

### Step 3: Process Documents
1. Click "Process Documents" button
2. Wait for the processing to complete
3. You'll see confirmation when documents are ready

### Step 4: Ask Questions
1. Enter your question in the query input box
2. Click "Get Answer"
3. View the generated answer and retrieved context

## 🏗️ Architecture

```
📁 rag-application/
├── app.py                    # Main application file
├── requirements.txt          # Python dependencies
├── 📁 data/                  # Uploaded documents storage
│   ├── 📁 text_files/       # Text documents
│   ├── 📁 pdf_files/        # PDF documents
│   ├── 📁 csv_files/        # CSV documents
│   └── 📁 other_files/      # Other file types
├── 📁 faiss_store/          # FAISS vector store
└── README.md                # This file
```

### Core Components

1. **DocumentProcessor Class**
   - Handles file uploads and organization
   - Manages document loading from multiple formats
   - Implements various text splitting strategies

2. **FaissVectorStore Class**
   - Custom wrapper for FAISS operations
   - Handles embedding generation and storage
   - Manages vector store persistence

3. **RAGPipeline Class**
   - Implements different RAG retrieval methods
   - Manages context retrieval and answer generation
   - Provides enhanced prompting strategies

4. **Streamlit UI**
   - Interactive file upload interface
   - Real-time configuration panel
   - Query and answer display

## ⚙️ Configuration Options

### Document Processing
- **Chunk Size**: 200-2000 characters (default: 1000)
- **Chunk Overlap**: 0-500 characters (default: 200)
- **Split Methods**: Recursive, Character, Token, Spacy

### RAG Settings
- **Retrieval Methods**: Simple, Enhanced, Advanced
- **Top K**: 1-10 chunks retrieved (default: 3)
- **LLM Temperature**: 0.1 (configurable in code)

### Embedding Settings
- **Model**: `all-MiniLM-L6-v2` (Sentence Transformers)
- **Embedding Dimension**: 384
- **Similarity Metric**: Cosine similarity

## 🔧 Technical Details

### Dependencies

```txt
streamlit==1.28.0            # Web application framework
langchain==0.1.0             # LLM framework
langchain-community==0.0.10  # Community integrations
langchain-groq==0.1.0        # Groq LLM integration
faiss-cpu==1.7.4             # Vector similarity search
sentence-transformers==2.2.2 # Embedding models
pypdf==3.17.0                # PDF processing
python-dotenv==1.0.0         # Environment management
spacy==3.7.0                 # NLP processing
pandas==2.0.3                # CSV processing
```

### API Integrations

1. **Groq API**
   - Used for LLM inference
   - Multiple model options available
   - Fast response times

2. **Sentence Transformers**
   - Local embedding generation
   - No external API calls needed
   - Consistent embedding quality

3. **FAISS**
   - Local vector database
   - Fast similarity search
   - Efficient memory usage

## 🛠️ Customization

### Adding New File Types
To add support for additional file types:

1. Update the `save_uploaded_file` method in `DocumentProcessor`
2. Add appropriate loader in `load_documents_from_directory`
3. Update the file uploader in the Streamlit UI

### Adding New Chunking Methods
```python
def new_chunking_method(documents, **kwargs):
    # Implement your custom chunking logic
    pass
```

### Modifying RAG Prompts
Edit the prompt templates in the `RAGPipeline` class methods:
- `simple_rag`
- `enhanced_rag`
- `advanced_rag`

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'langchain'"**
   ```bash
   pip install --upgrade langchain langchain-community
   ```

2. **Groq API Key Error**
   - Ensure API key is set in `.env` or UI
   - Check API key validity at [console.groq.com](https://console.groq.com)

3. **FAISS Loading Error**
   ```bash
   pip install --upgrade faiss-cpu
   ```

4. **Spacy Model Not Found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **File Upload Issues**
   - Check directory permissions
   - Ensure `data/` directory exists
   - Verify file type is supported

### Debug Mode
Run with debug information:
```bash
streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false
```

## 📊 Performance Tips

1. **Optimal Chunk Sizes**
   - Documents: 500-1500 characters
   - Overlap: 10-20% of chunk size
   - Adjust based on document type

2. **Memory Management**
   - Process documents in batches for large collections
   - Clear cache regularly using the "Clear All Data" button
   - Monitor FAISS index size

3. **Query Optimization**
   - Use specific, clear questions
   - Adjust top_k based on document complexity
   - Experiment with different RAG types

## 🔮 Future Enhancements

### Planned Features
- [ ] **Hybrid Search**: Combine semantic and keyword search
- [ ] **Metadata Filtering**: Filter documents by metadata
- [ ] **Multi-modal Support**: Image and audio processing
- [ ] **Batch Processing**: Process multiple queries
- [ ] **Export Results**: Save queries and answers
- [ ] **User Authentication**: Multi-user support
- [ ] **API Endpoints**: REST API for integration
- [ ] **Monitoring Dashboard**: Usage analytics

### Integration Possibilities
- **Database Backends**: PostgreSQL, Chroma, Pinecone
- **Alternative LLMs**: OpenAI, Anthropic, Local models
- **Additional Embeddings**: OpenAI, Cohere, Voyage AI

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) for the amazing LLM framework
- [Groq](https://groq.com/) for high-performance LLM inference
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- [Streamlit](https://streamlit.io/) for the intuitive web framework
- [Sentence Transformers](https://www.sbert.net/) for embedding models

**Built with ❤️ for the AI/ML community**