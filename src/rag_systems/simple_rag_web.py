"""
RAG System for Web Application
Lightweight version without heavy dependencies
"""

import json
import re
from typing import List, Dict, Any
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

@dataclass
class Document:
    """Document representation"""
    id: str
    content: str
    metadata: Dict[str, Any] = None

@dataclass
class RAGResponse:
    """RAG response with metadata"""
    answer: str
    retrieved_documents: List[Document]
    query: str
    metadata: Dict[str, Any] = None

class SimpleRAGWeb:
    """
    RAG system for web application
    Uses TF-IDF for retrieval without heavy ML dependencies
    """
    
    def __init__(self, llm, documents: List[Document]):
        self.llm = llm
        self.documents = documents
        self.vectorizer = None
        self.doc_vectors = None
        
        # Initialize vectorizer
        self._initialize_vectorizer()
    
    def _initialize_vectorizer(self):
        """Initialize TF-IDF vectorizer"""
        if not self.documents:
            return
        
        # Extract text content
        texts = [doc.content for doc in self.documents]
        
        # Create vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Fit and transform documents
        self.doc_vectors = self.vectorizer.fit_transform(texts)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Document]:
        """
        Retrieve relevant documents using TF-IDF similarity
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if not self.vectorizer or self.doc_vectors is None:
            return self.documents[:top_k]
        
        try:
            # Transform query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
            
            # Get top-k indices
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            # Return documents
            return [self.documents[i] for i in top_indices if similarities[i] > 0.1]
            
        except Exception as e:
            print(f"Error in retrieval: {e}")
            return self.documents[:top_k]
    
    def query(self, user_query: str, system_prompt: str = "You are a helpful AI assistant.") -> str:
        """
        Process a query using RAG with enhanced retrieval capabilities
        
        Args:
            user_query: User's question
            system_prompt: System prompt for the LLM
            
        Returns:
            Generated response
        """
        try:
            # Check if the LLM supports enhanced RAG
            if hasattr(self.llm, 'query_with_rag'):
                # Use enhanced RAG with query expansion, reranking, and diversity selection
                documents = []
                for doc in self.documents:
                    documents.append({
                        'content': doc.content,
                        'title': doc.metadata.get('title', f'Document {doc.id}'),
                        'metadata': doc.metadata
                    })
                
                return self.llm.query_with_rag(user_query, documents)
            else:
                # Fallback to simple retrieval
                retrieved_docs = self.retrieve(user_query, top_k=3)
                
                # Build context
                context_parts = []
                for i, doc in enumerate(retrieved_docs, 1):
                    context_parts.append(f"Document {i}: {doc.content[:500]}...")
                
                context = "\n\n".join(context_parts)
                
                # Create prompt
                full_prompt = f"""Context:
{context}

Question: {user_query}

Please answer the question based on the provided context. If the context doesn't contain enough information to answer the question, please say so."""
                
                # Generate response
                response = self.llm.generate(full_prompt, system_prompt)
                
                return response
            
        except Exception as e:
            return f"I apologize, but I encountered an error while processing your query: {str(e)}"
    
    def add_documents(self, documents: List[Document]):
        """Add new documents to the system"""
        self.documents.extend(documents)
        self._initialize_vectorizer()
    
    def get_document_count(self) -> int:
        """Get total number of documents"""
        return len(self.documents)
    
    def search_documents(self, query: str, limit: int = 10) -> List[Dict]:
        """Search documents and return metadata"""
        retrieved_docs = self.retrieve(query, top_k=limit)
        
        results = []
        for doc in retrieved_docs:
            results.append({
                'id': doc.id,
                'title': doc.metadata.get('title', 'Untitled'),
                'content': doc.content,
                'source': doc.metadata.get('source', 'Unknown')
            })
        
        return results
