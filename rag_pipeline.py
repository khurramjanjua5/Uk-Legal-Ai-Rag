"""
UK Legal AI Assistant - RAG Pipeline Implementation
COM748 Masters Research Project - Khurram Shahzad
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple
import json

class LegalRAGPipeline:
    """
    Retrieval-Augmented Generation Pipeline for UK Legal Documents
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the RAG pipeline with embedding model"""
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.index = None
        self.documents = []
        
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks for better retrieval
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Number of characters to overlap between chunks
        
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start += chunk_size - overlap
            
        return chunks
    
    def preprocess_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Preprocess legal documents: chunk and prepare for indexing
        
        Args:
            documents: List of dicts with 'text', 'source', 'citation' keys
        
        Returns:
            List of processed document chunks
        """
        processed_docs = []
        
        for doc in documents:
            chunks = self.chunk_text(doc['text'])
            
            for i, chunk in enumerate(chunks):
                processed_docs.append({
                    'text': chunk,
                    'source': doc['source'],
                    'citation': doc.get('citation', 'N/A'),
                    'chunk_id': f"{doc.get('id', 0)}-{i}"
                })
        
        self.documents = processed_docs
        print(f"Preprocessed {len(documents)} documents into {len(processed_docs)} chunks")
        return processed_docs
    
    def build_index(self, documents: List[Dict]):
        """
        Build FAISS vector index from documents
        
        Args:
            documents: List of document dictionaries
        """
        processed_docs = self.preprocess_documents(documents)
        
        # Extract text and create embeddings
        texts = [doc['text'] for doc in processed_docs]
        print(f"Creating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Build FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings.astype('float32'))
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """
        Retrieve most relevant documents for a query
        
        Args:
            query: User query string
            top_k: Number of documents to retrieve
        
        Returns:
            List of (document, score) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        
        # Search index
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Retrieve documents with scores
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                # Convert L2 distance to similarity score
                similarity = 1 / (1 + dist)
                results.append((self.documents[idx], similarity))
        
        return results
    
    def generate_response(self, query: str, retrieved_docs: List[Tuple[Dict, float]]) -> Dict:
        """
        Generate response with citations based on retrieved documents
        
        Args:
            query: User query
            retrieved_docs: List of (document, score) tuples from retrieval
        
        Returns:
            Dict with 'answer', 'sources', 'confidence'
        """
        if not retrieved_docs:
            return {
                'answer': "No relevant documents found in the knowledge base.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Build context from retrieved documents
        context = "\n\n".join([
            f"[{i+1}] {doc['source']}: {doc['text']}"
            for i, (doc, score) in enumerate(retrieved_docs)
        ])
        
        # Extract sources with citations
        sources = [
            {
                'citation': doc['citation'],
                'source': doc['source'],
                'text': doc['text'],
                'relevance_score': float(score)
            }
            for doc, score in retrieved_docs
        ]
        
        # Calculate confidence based on top retrieval score
        confidence = min(retrieved_docs[0][1] * 100, 95.0)
        
        # In production, this would call an LLM API with the context
        # For now, return structured response
        answer = f"Based on retrieved legal sources:\n\n{retrieved_docs[0][0]['text']}\n\n"
        answer += f"Source: {retrieved_docs[0][0]['citation']}"
        
        return {
            'answer': answer,
            'sources': sources,
            'confidence': confidence,
            'context': context
        }
    
    def query(self, user_query: str, top_k: int = 3) -> Dict:
        """
        End-to-end query processing: retrieve + generate
        
        Args:
            user_query: User's legal question
            top_k: Number of documents to retrieve
        
        Returns:
            Response dictionary with answer, sources, and confidence
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(user_query, top_k)
        
        # Generate response
        response = self.generate_response(user_query, retrieved_docs)
        
        return response
    
    def save_index(self, path: str):
        """Save FAISS index and documents to disk"""
        faiss.write_index(self.index, f"{path}/faiss.index")
        
        with open(f"{path}/documents.json", 'w') as f:
            json.dump(self.documents, f, indent=2)
        
        print(f"Index saved to {path}")
    
    def load_index(self, path: str):
        """Load FAISS index and documents from disk"""
        self.index = faiss.read_index(f"{path}/faiss.index")
        
        with open(f"{path}/documents.json", 'r') as f:
            self.documents = json.load(f)
        
        print(f"Index loaded from {path}")


# Example usage and testing
if __name__ == "__main__":
    # Sample UK legal documents
    sample_documents = [
        {
            'id': 1,
            'source': 'Employment Rights Act 1996, Section 94',
            'citation': 'ERA 1996 s.94',
            'text': 'An employee has the right not to be unfairly dismissed by his employer. This right is subject to certain qualifying conditions including length of service requirements.'
        },
        {
            'id': 2,
            'source': 'BAILII [2020] UKSC 15 - Uber BV v Aslam',
            'citation': '[2020] UKSC 15',
            'text': 'The Supreme Court held that Uber drivers are workers within the meaning of employment legislation and entitled to worker rights including minimum wage and holiday pay.'
        },
        {
            'id': 3,
            'source': 'Contract Law - Carlill v Carbolic Smoke Ball Co [1893]',
            'citation': '[1893] 1 QB 256',
            'text': 'A unilateral contract can be formed through performance of conditions specified in an advertisement. The court held that the advertisement constituted an offer to the world.'
        }
    ]
    
    # Initialize pipeline
    print("Initializing RAG Pipeline...")
    rag = LegalRAGPipeline()
    
    # Build index
    rag.build_index(sample_documents)
    
    # Test query
    query = "What are the rights regarding unfair dismissal?"
    print(f"\nQuery: {query}")
    
    response = rag.query(query, top_k=2)
    
    print(f"\nAnswer: {response['answer']}")
    print(f"Confidence: {response['confidence']:.2f}%")
    print(f"\nSources Retrieved: {len(response['sources'])}")
    for i, source in enumerate(response['sources'], 1):
        print(f"  [{i}] {source['citation']} (Relevance: {source['relevance_score']:.2f})")
