import streamlit as st
import os
import shutil
from pathlib import Path
import time
from utility import *
# LangChain imports
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter, 
    CharacterTextSplitter,
    TokenTextSplitter,
    SpacyTextSplitter
)

from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create necessary directories
def create_directories():
    """Create necessary directories for file storage"""
    directories = ["data", "data/text_files", "data/pdf_files", "data/csv_files", "faiss_store"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

create_directories()

class DocumentProcessor:
    """Handle document loading and processing"""
    
    @staticmethod
    def save_uploaded_file(uploaded_file):
        """Save uploaded file to appropriate directory"""
        try:
            file_name = uploaded_file.name
            file_extension = Path(file_name).suffix.lower()
            
            # Determine save directory based on file extension
            if file_extension == '.txt':
                save_dir = "data/text_files"
            elif file_extension == '.pdf':
                save_dir = "data/pdf_files"
            elif file_extension == '.csv':
                save_dir = "data/csv_files"
            else:
                save_dir = "data/other_files"
            
            # Ensure directory exists
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            
            # Create save path
            save_path = os.path.join(save_dir, file_name)
            
            # Save the file
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            return save_path
        except Exception as e:
            st.error(f"Error saving file {uploaded_file.name}: {str(e)}")
            return None
    
    @staticmethod
    def load_documents_from_directory():
        """Load all documents from saved directories"""
        documents = []
        
        # Load text files
        text_dir = "data/text_files"
        if os.path.exists(text_dir):
            for file_path in Path(text_dir).glob("*.txt"):
                try:
                    loader = TextLoader(str(file_path))
                    loaded_docs = loader.load()
                    documents.extend(loaded_docs)
                    st.info(f"Loaded text file: {file_path.name}")
                except Exception as e:
                    st.warning(f"Error loading {file_path}: {e}")
        
        # Load PDF files
        pdf_dir = "data/pdf_files"
        if os.path.exists(pdf_dir):
            for file_path in Path(pdf_dir).glob("*.pdf"):
                try:
                    loader = PyPDFLoader(str(file_path))
                    loaded_docs = loader.load()
                    documents.extend(loaded_docs)
                    st.info(f"Loaded PDF file: {file_path.name}")
                except Exception as e:
                    st.warning(f"Error loading {file_path}: {e}")
        
        # Load CSV files
        csv_dir = "data/csv_files"
        if os.path.exists(csv_dir):
            for file_path in Path(csv_dir).glob("*.csv"):
                try:
                    loader = CSVLoader(str(file_path))
                    loaded_docs = loader.load()
                    documents.extend(loaded_docs)
                    st.info(f"Loaded CSV file: {file_path.name}")
                except Exception as e:
                    st.warning(f"Error loading {file_path}: {e}")
        
        # Load other files (text-based)
        other_dir = "data/other_files"
        if os.path.exists(other_dir):
            for file_path in Path(other_dir).glob("*"):
                try:
                    # Try loading as text file
                    if file_path.suffix.lower() in ['.md', '.json', '.xml', '.html', '.htm']:
                        loader = TextLoader(str(file_path))
                        loaded_docs = loader.load()
                        documents.extend(loaded_docs)
                        st.info(f"Loaded other file: {file_path.name}")
                except Exception as e:
                    st.warning(f"Error loading {file_path}: {e}")
        
        return documents
    
    @staticmethod
    def split_documents(documents, split_method="recursive", **kwargs):
        """Split documents using selected method"""
        if not documents:
            return []
        
        if split_method == "recursive":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=kwargs.get("chunk_size", 1000),
                chunk_overlap=kwargs.get("chunk_overlap", 200),
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
        elif split_method == "character":
            text_splitter = CharacterTextSplitter(
                chunk_size=kwargs.get("chunk_size", 1000),
                chunk_overlap=kwargs.get("chunk_overlap", 200),
                separator="\n"
            )
        elif split_method == "token":
            text_splitter = TokenTextSplitter(
                chunk_size=kwargs.get("chunk_size", 1000),
                chunk_overlap=kwargs.get("chunk_overlap", 200)
            )
        elif split_method == "spacy":
            try:
                text_splitter = SpacyTextSplitter(
                    pipeline="en_core_web_sm",
                    chunk_size=kwargs.get("chunk_size", 1000)
                )
            except:
                st.error("Spacy not installed. Install with: pip install spacy && python -m spacy download en_core_web_sm")
                return documents
        
        split_docs = text_splitter.split_documents(documents)
        st.success(f"Split {len(documents)} documents into {len(split_docs)} chunks")
        return split_docs
    
class RAGPipeline:
    """Handle different RAG retrieval methods"""
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.history = []

    ## 2. Simple RAG function: retrieve context + generate response
    def rag_simple(self, query, top_k=3):
        results = self.retriever.retrieve(query, top_k=top_k)

        context = "\n\n".join([doc["content"] for doc in results]) if results else ""
        if not context:
            return "No relevant context found to answer the question."

        prompt = f"""Use the following context to answer the question concisely.

                    Context:
                    {context}

                    Question: {query}

                    Answer:
                """

        response = self.llm.invoke(prompt)
        return response.content, context
    
    # --- Enhanced RAG Pipeline Features ---
    def rag_enhance(self,query, top_k=5, min_score=0.2, return_context=True):
        """
        RAG pipeline with extra features:
        - Returns answer, sources, confidence score, and optionally full context.
        """
        results = self.retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
        if not results:
            return {'answer': 'No relevant context found.', 'sources': [], 'confidence': 0.0, 'context': ''}
        
        # Prepare context and sources
        context = "\n\n".join([doc['content'] for doc in results])
        sources = [{
            'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
            'page': doc['metadata'].get('page', 'unknown'),
            'score': doc['similarity_score'],
            'preview': doc['content'][:300] + '...'
        } for doc in results]
        confidence = max([doc['similarity_score'] for doc in results])
        
        # Generate answer
        prompt = f"""Use the following context to answer the question concisely.\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
        response = self.llm.invoke([prompt.format(context=context, query=query)])
        
        output = {
            'answer': response.content,
            'sources': sources,
            'confidence': confidence
        }

        return output, context

    def rag_advance(self, question: str, top_k: int = 5, min_score: float = 0.2, stream: bool = False, summarize: bool = False) -> Dict[str, Any]:
        # Retrieve relevant documents
        results = self.retriever.retrieve(question, top_k=top_k, score_threshold=min_score)
        if not results:
            answer = "No relevant context found."
            sources = []
            context = ""
        else:
            context = "\n\n".join([doc['content'] for doc in results])
            sources = [{
                'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
                'page': doc['metadata'].get('page', 'unknown'),
                'score': doc['similarity_score'],
                'preview': doc['content'][:120] + '...'
            } for doc in results]
            # Streaming answer simulation
            prompt = f"""Use the following context to answer the question concisely.\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"""
            if stream:
                print("Streaming answer:")
                for i in range(0, len(prompt), 80):
                    print(prompt[i:i+80], end='', flush=True)
                    time.sleep(0.05)
                print()
            response = self.llm.invoke([prompt.format(context=context, question=question)])
            answer = response.content

        # Add citations to answer
        citations = [f"[{i+1}] {src['source']} (page {src['page']})" for i, src in enumerate(sources)]
        answer_with_citations = answer + "\n\nCitations:\n" + "\n".join(citations) if citations else answer

        # Optionally summarize answer
        summary = None
        if summarize and answer:
            summary_prompt = f"Summarize the following answer in 2 sentences:\n{answer}"
            summary_resp = self.llm.invoke([summary_prompt])
            summary = summary_resp.content
        
        # Store query history
        self.history.append({
            'question': question,
            'answer': answer,
            'sources': sources,
            'summary': summary
        })

        return {
            'question': question,
            'answer': answer_with_citations,
            'sources': sources,
            'summary': summary,
            'history': self.history
        }
    
def main():
    st.set_page_config(
        page_title="RAG Application",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Advanced RAG Application")
    st.markdown("---")
    
    # Initialize session state
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'split_docs' not in st.session_state:
        st.session_state.split_docs = []
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key Input
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Enter your Groq API key"
        )
            
        # LLM Model Selection
        llm_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b",
            "groq/compound",
            "meta-llama/llama-guard-4-12b"
        ]
        selected_model = st.selectbox("Select LLM Model", llm_models)
        
        st.subheader("Document Processing")
        adv_settings = st.checkbox('Advance Settings')
        # Document processing parameters
        if adv_settings:
            chunk_size = st.slider("Chunk Size", 200, 2000, 1000, 100)
            chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 50)
            
            # Split method selection
            split_method = st.radio(
                "Select Split Method",
                ["recursive", "character", "token", "spacy"],
                help="Recursive: Best for general text, Character: By characters, Token: By tokens, Spacy: NLP-based"
            )
            # Top K for retrieval
            top_k = st.slider("Number of chunks to retrieve (top_k)", 1, 10, 3)
            
        
        # RAG type selection
        rag_type = st.radio(
            "Select RAG Type",
            ["simple_rag", "enhanced_rag", "advanced_rag"],
            help="Simple: Basic retrieval, Enhanced: Better prompting, Advanced: Citations"
        )
        
        # Process button
        if st.button("🔄 Process Documents", type="primary", use_container_width=True):
            if st.session_state.documents:
                with st.spinner("Processing documents..."):
                    # Split documents
                    st.session_state.split_docs = DocumentProcessor.split_documents(
                        st.session_state.documents,
                        split_method=split_method,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                    if st.session_state.split_docs:
                        # Create and save vector store
                        store = FaissVectorStore()
                        store.build_from_documents(st.session_state.split_docs)
                        store.load()
                        st.session_state.vector_store = store
                        st.session_state.processed = True
                        st.success("Documents processed and indexed successfully!")
            else:
                st.warning("No documents loaded. Please upload files first.")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📁 File Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload files",
            type=["txt", "pdf", "csv"],
            accept_multiple_files=True,
            help="Upload text, PDF, or CSV files"
        )
        
        if uploaded_files:
            if st.button("💾 Save and Load Files"):
                with st.spinner("Saving and loading files..."):
                    for uploaded_file in uploaded_files:
                        save_path = DocumentProcessor.save_uploaded_file(uploaded_file)
                        st.success(f"Saved: {uploaded_file.name}")
                    
                    # Load all documents
                    st.session_state.documents = DocumentProcessor.load_documents_from_directory()
                    st.success(f"Loaded {len(st.session_state.documents)} documents")
        
        # Display loaded documents
        if st.session_state.documents:
            st.subheader("📋 Loaded Documents")
            show = st.checkbox('Show Loaded Documents')
            if show:
                for i, doc in enumerate(st.session_state.documents[:10], 1):
                    with st.expander(f"Document {i}: {doc.metadata.get('source', 'Unknown')}"):
                        st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
            
            if len(st.session_state.documents) > 10:
                st.info(f"... and {len(st.session_state.documents) - 10} more documents")
        
        # Display processing status
        if st.session_state.processed:
            st.success("✅ Documents are processed and ready for queries")
            st.info(f"Chunks created: {len(st.session_state.split_docs)}")
    
    with col2:
        st.header("💬 Query Interface")
        
        # Query input
        query = st.text_input("Enter your question:")
        
        if query and st.session_state.processed and groq_api_key:
            if st.button("🔍 Get Answer", type="primary"):
                with st.spinner("Retrieving and generating answer..."):
                    try:
                        # Initialize LLM
                        llm = ChatGroq(
                            groq_api_key=groq_api_key,
                            model_name=selected_model,
                            temperature=0.1,
                            max_tokens=1024
                        )
                        
                        # Get retriever
                        embedding_manager=EmbeddingManager()
                        retriever=RAGRetriever(st.session_state.vector_store,embedding_manager)
                        # retriever = st.session_state.vector_store.get_retriever(top_k=top_k)
                        
                        # Initialize RAG pipeline
                        rag_pipeline = RAGPipeline(retriever, llm)
                        
                        # Execute based on selected RAG type
                        if rag_type == "simple_rag":
                            answer, context = rag_pipeline.rag_simple(query, top_k)
                        elif rag_type == "enhanced_rag":
                            answer, context = rag_pipeline.rag_enhance(query, top_k)
                        else:  # advanced_rag
                            result = rag_pipeline.rag_advance(query, top_k)
                            answer = result['answer']
                            context = result['sources']
                                      
                        
                        # Display answer
                        st.subheader("📝 Answer")
                        st.write(answer)
                        
                        # Display context in expander
                        with st.expander("🔍 View Retrieved Context"):
                            st.text(context)
                        
                        # Display metadata
                        with st.expander("📊 Query Information"):
                            st.write(f"**Model:** {selected_model}")
                            st.write(f"**RAG Type:** {rag_type}")
                            st.write(f"**Chunks Retrieved:** {top_k}")
                            st.write(f"**Split Method:** {split_method}")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.info("Make sure your Groq API key is valid and you have sufficient credits.")
        
        elif query and not groq_api_key:
            st.warning("Please enter your Groq API key in the sidebar.")
        elif query and not st.session_state.processed:
            st.warning("Please process the documents first.")
        
        # Clear button
        if st.button("🗑️ Clear All Data"):
            for folder in ["data/text_files", "data/pdf_files", "data/csv_files", "faiss_store"]:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
                    os.makedirs(folder)
            
            st.session_state.clear()
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with Streamlit, LangChain, FAISS, and Groq</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()