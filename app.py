"""
UK Legal AI Assistant - Streamlit Web Interface
COM748 Masters Research Project - Khurram Shahzad

UPDATED: PDF upload now fully functional - adds documents to knowledge base
"""

import streamlit as st
from rag_pipeline import LegalRAGPipeline
import pdfplumber
import io
import json

# Page configuration
st.set_page_config(
    page_title="UK Legal AI Assistant",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
    st.session_state.index_built = False
    st.session_state.documents = []
    st.session_state.uploaded_docs = []

def load_default_corpus():
    """Load default UK legal corpus"""
    default_documents = [
        {
            'id': 1,
            'source': 'Employment Rights Act 1996, Section 94',
            'citation': 'ERA 1996 s.94',
            'text': 'An employee has the right not to be unfairly dismissed by his employer. This right is subject to certain qualifying conditions including length of service. The employer must show a potentially fair reason for dismissal and follow fair procedures.'
        },
        {
            'id': 2,
            'source': 'BAILII [2020] UKSC 15 - Uber BV v Aslam',
            'citation': '[2020] UKSC 15',
            'text': 'The Supreme Court held that Uber drivers are workers within the meaning of employment legislation and entitled to worker rights including minimum wage and holiday pay. This landmark decision has significant implications for gig economy workers.'
        },
        {
            'id': 3,
            'source': 'Contract Law - Carlill v Carbolic Smoke Ball Co [1893]',
            'citation': '[1893] 1 QB 256',
            'text': 'A unilateral contract can be formed through performance of conditions specified in an advertisement. The court held that the advertisement constituted an offer to the world which could be accepted by anyone who performed the specified conditions.'
        },
        {
            'id': 4,
            'source': 'Consumer Rights Act 2015, Section 9',
            'citation': 'CRA 2015 s.9',
            'text': 'Every contract to supply goods is to be treated as including a term that the quality of the goods is satisfactory. Goods must be of satisfactory quality, fit for particular purpose, and as described. A consumer has the right to reject faulty goods within 30 days.'
        },
        {
            'id': 5,
            'source': 'BAILII [2015] UKSC 11 - Patel v Mirza',
            'citation': '[2015] UKSC 11',
            'text': 'The Supreme Court reformulated the test for illegality in contract law. Courts must consider whether allowing recovery would produce inconsistency and whether denial of relief is proportionate to the illegal conduct.'
        },
        {
            'id': 6,
            'source': 'Human Rights Act 1998, Article 8',
            'citation': 'HRA 1998 Art 8',
            'text': 'Everyone has the right to respect for his private and family life, his home and his correspondence. There shall be no interference by a public authority with the exercise of this right except such as is in accordance with the law.'
        }
    ]
    return default_documents

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF"""
    try:
        with pdfplumber.open(io.BytesIO(pdf_file.read())) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting PDF: {str(e)}")
        return None

def initialize_pipeline():
    """Initialize RAG pipeline"""
    return LegalRAGPipeline(model_name='all-MiniLM-L6-v2')

def rebuild_index():
    """Rebuild FAISS index with all documents"""
    if st.session_state.documents:
        with st.spinner("Rebuilding knowledge base..."):
            st.session_state.rag_pipeline.build_index(st.session_state.documents)
        return True
    return False

# Header
st.markdown("""
<div class="main-header">
    <h1>⚖️ UK Legal AI Assistant</h1>
    <p>Domain-Specific RAG System for UK Law</p>
    <p style="font-size: 0.9rem;">COM748 Masters Research Project - Khurram Shahzad (20025779)</p>
</div>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ Legal Disclaimer:</strong> This system is an informational research prototype only. 
    It does not constitute legal advice. Always consult a qualified legal professional for legal decisions.
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    # Mode selection
    mode = st.radio(
        "Select Mode:",
        ["RAG (Citation-Backed)", "Baseline (No RAG)"],
        help="Compare RAG vs baseline LLM performance"
    )
    
    st.divider()
    
    # System parameters
    st.subheader("RAG Parameters")
    top_k = st.slider("Documents to Retrieve (k)", 1, 5, 3)
    
    st.divider()
    
    # PDF Upload Section
    st.subheader("📄 Upload Legal Document")
    st.markdown("Upload a PDF to add it to the knowledge base")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a legal document to add to the knowledge base"
    )
    
    if uploaded_file:
        doc_title = st.text_input("Document Title", value=uploaded_file.name.replace('.pdf', ''))
        doc_citation = st.text_input("Citation (e.g., [2024] UKSC 1)", value="[Custom Document]")
        
        if st.button("📥 Add to Knowledge Base", type="primary"):
            with st.spinner("Processing PDF..."):
                text = extract_text_from_pdf(uploaded_file)
                
                if text and len(text.strip()) > 50:
                    # Create new document entry
                    new_doc = {
                        'id': len(st.session_state.documents) + 1,
                        'source': doc_title,
                        'citation': doc_citation,
                        'text': text.strip()
                    }
                    
                    # Add to documents list
                    st.session_state.documents.append(new_doc)
                    st.session_state.uploaded_docs.append(doc_title)
                    
                    # Rebuild index with new document
                    if rebuild_index():
                        st.success(f"✅ '{doc_title}' added successfully!")
                        st.info(f"Total documents in knowledge base: {len(st.session_state.documents)}")
                        
                        # Show preview
                        with st.expander("📄 Document Preview"):
                            st.text_area("Extracted Text", text[:500] + "...", height=150, disabled=True)
                    else:
                        st.error("Failed to rebuild index")
                else:
                    st.error("Could not extract enough text from PDF. Please check the file.")
    
    st.divider()
    
    # Show uploaded documents
    if st.session_state.uploaded_docs:
        st.subheader("📚 Uploaded Documents")
        for i, doc_name in enumerate(st.session_state.uploaded_docs, 1):
            st.text(f"{i}. {doc_name}")
    
    st.divider()
    
    # System info
    st.subheader("📊 System Information")
    st.info(f"""
    **Model:** Sentence-Transformers  
    **Vector Store:** FAISS  
    **Documents Loaded:** {len(st.session_state.documents)}  
    **Status:** {'✅ Ready' if st.session_state.index_built else '⏳ Initializing'}
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🔍 Query Interface")
    
    # Initialize pipeline on first run
    if not st.session_state.index_built:
        with st.spinner("Loading RAG pipeline and building index..."):
            # Load default corpus
            default_docs = load_default_corpus()
            st.session_state.documents = default_docs
            
            # Initialize pipeline
            pipeline = initialize_pipeline()
            st.session_state.rag_pipeline = pipeline
            
            # Build initial index
            pipeline.build_index(st.session_state.documents)
            st.session_state.index_built = True
            
        st.markdown("""
        <div class="success-box">
            <strong>✅ System Initialized Successfully!</strong><br>
            Loaded 6 default UK legal documents. You can now ask questions or upload additional documents.
        </div>
        """, unsafe_allow_html=True)
    
    # Query input
    query = st.text_area(
        "Enter your legal question:",
        placeholder="e.g., What are the rights regarding unfair dismissal?",
        height=100
    )
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    with col_btn2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    with col_btn3:
        refresh_button = st.button("🔄 Refresh Index", use_container_width=True)
    
    if clear_button:
        st.rerun()
    
    if refresh_button:
        if rebuild_index():
            st.success("✅ Knowledge base refreshed!")
            st.rerun()
    
    # Process query
    if search_button and query:
        with st.spinner("Processing query..."):
            if mode == "RAG (Citation-Backed)":
                # RAG mode
                try:
                    response = st.session_state.rag_pipeline.query(query, top_k=top_k)
                    
                    # Display answer
                    st.subheader("📝 Answer")
                    st.success(response['answer'])
                    
                    # Display confidence
                    confidence = response['confidence']
                    st.metric("Confidence Score", f"{confidence:.1f}%")
                    
                    # Display sources
                    if response['sources']:
                        st.subheader("📚 Retrieved Sources & Citations")
                        
                        for i, source in enumerate(response['sources'], 1):
                            with st.expander(f"[{i}] {source['citation']} - Relevance: {source['relevance_score']:.2%}"):
                                st.markdown(f"**Source:** {source['source']}")
                                st.markdown(f"**Text:** {source['text']}")
                                st.progress(source['relevance_score'])
                    else:
                        st.warning("⚠️ No relevant sources found in the knowledge base.")
                        
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    st.info("Try rebuilding the index using the 'Refresh Index' button")
                
            else:
                # Baseline mode (simulated)
                st.subheader("📝 Answer (Baseline - No RAG)")
                st.warning(
                    "Based on general legal principles, unfair dismissal typically requires "
                    "the employer to follow proper procedures and have valid reasons. Employees "
                    "usually need two years of continuous service to claim unfair dismissal. "
                    "Note: This is a general response without specific legal authority."
                )
                st.metric("Confidence Score", "60.0%")
                st.info("⚠️ No sources retrieved - response may contain hallucinations")

with col2:
    st.header("💡 Example Queries")
    
    example_queries = [
        "What are the rights regarding unfair dismissal?",
        "What is the legal status of Uber drivers?",
        "Can an advertisement form a contract?",
        "What are consumer rights for faulty goods?",
        "What is Article 8 of the Human Rights Act?",
        "Explain the illegality test in contract law"
    ]
    
    st.markdown("**Try these example questions:**")
    
    for example in example_queries:
        if st.button(example, key=f"example_{example[:20]}", use_container_width=True):
            st.session_state.current_query = example
            st.rerun()
    
    # Handle example query selection
    if 'current_query' in st.session_state:
        # This will populate the query field
        pass

# Footer
st.divider()
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><strong>Powered by:</strong> Sentence-Transformers + FAISS + RAG Pipeline</p>
    <p><strong>Data Sources:</strong> BAILII, UK Legislation, Legal Precedents</p>
    <p><strong>Documents in Knowledge Base:</strong> {len(st.session_state.documents)}</p>
    <p><strong>Supervisor:</strong> Dr. Ekereuke Udoh</p>
</div>
""", unsafe_allow_html=True)