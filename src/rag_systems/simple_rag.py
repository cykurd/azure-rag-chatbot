"""
Simple RAG System for Portfolio Demonstration

A clean, production-ready RAG system that demonstrates key concepts
without requiring Azure services. Uses local embeddings and vector storage.
"""

import os
import logging
import json
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

from ..models.base_model import BaseModel

load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Simple document structure for RAG system"""
    id: str
    content: str
    title: str
    metadata: Dict[str, Any]

@dataclass
class RAGResponse:
    """Response from RAG system"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    query: str

class SimpleRAG:
    """
    Simple RAG system using local embeddings and ChromaDB
    """
    
    def __init__(self, 
                 vector_db_path: str = "data/vector_db",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the RAG system
        
        Args:
            vector_db_path: Path to store the vector database
            embedding_model: Name of the embedding model to use
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
        """
        self.vector_db_path = Path(vector_db_path)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.vector_db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize LLM
        self.llm = BaseModel(model_name='gpt-4', temperature=0.0)
        
        logger.info(f"SimpleRAG initialized with {self.collection.count()} documents")
    
    def chunk_text(self, text: str, title: str = "") -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            title: Title of the document
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence boundaries
            if end < len(text):
                last_period = chunk_text.rfind('.')
                if last_period > self.chunk_size * 0.7:
                    chunk_text = chunk_text[:last_period + 1]
                    end = start + last_period + 1
            
            chunk_text = chunk_text.strip()
            
            if not chunk_text:
                start = end - self.chunk_overlap
                continue
            
            chunk = {
                'id': f"{title}_{chunk_id}_{uuid.uuid4().hex[:8]}",
                'text': chunk_text,
                'title': title,
                'chunk_id': chunk_id,
                'start': start,
                'end': end
            }
            
            chunks.append(chunk)
            start = end - self.chunk_overlap
            chunk_id += 1
        
        return chunks
    
    def add_documents(self, documents: List[Document]):
        """
        Add documents to the vector database
        
        Args:
            documents: List of Document objects to add
        """
        logger.info(f"Adding {len(documents)} documents to vector database")
        
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc.content, doc.title)
            for chunk in chunks:
                chunk['document_id'] = doc.id
                chunk['metadata'] = doc.metadata
            all_chunks.extend(chunks)
        
        if not all_chunks:
            logger.warning("No chunks generated from documents")
            return
        
        # Prepare data for ChromaDB
        texts = [chunk['text'] for chunk in all_chunks]
        metadatas = [{
            'title': chunk['title'],
            'document_id': chunk['document_id'],
            'chunk_id': chunk['chunk_id'],
            'start': chunk['start'],
            'end': chunk['end'],
            **chunk.get('metadata', {})
        } for chunk in all_chunks]
        ids = [chunk['id'] for chunk in all_chunks]
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Add to ChromaDB
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Successfully added {len(all_chunks)} chunks to vector database")
        except Exception as e:
            logger.error(f"Error adding chunks to ChromaDB: {e}")
            raise
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results['documents'] or not results['documents'][0]:
                return []
            
            # Format results
            chunks = []
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            
            for doc, metadata, distance in zip(documents, metadatas, distances):
                similarity = 1.0 - distance  # Convert distance to similarity
                
                chunk = {
                    'text': doc,
                    'metadata': metadata,
                    'similarity': similarity,
                    'distance': distance
                }
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
    
    async def query(self, 
                   query: str, 
                   top_k: int = 5,
                   max_tokens: int = 1000) -> RAGResponse:
        """
        Query the RAG system
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            max_tokens: Maximum tokens for response
            
        Returns:
            RAGResponse with answer and sources
        """
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            # Retrieve relevant chunks
            chunks = self.retrieve(query, top_k)
            
            if not chunks:
                return RAGResponse(
                    answer="I couldn't find any relevant information for your question. Please try rephrasing or providing more specific details.",
                    sources=[],
                    confidence=0.0,
                    query=query
                )
            
            # Prepare context for LLM
            context_parts = []
            for i, chunk in enumerate(chunks, 1):
                metadata = chunk['metadata']
                title = metadata.get('title', 'Unknown')
                similarity = chunk['similarity']
                
                context_parts.append(
                    f"Source {i} (similarity: {similarity:.3f}):\n"
                    f"Title: {title}\n"
                    f"Content: {chunk['text']}"
                )
            
            context = "\n\n".join(context_parts)
            
            # Create prompt
            prompt = f"""You are a helpful AI assistant. Use the following information to answer the user's question.

Context:
{context}

Question: {query}

Instructions:
- Answer based on the provided context
- If the context doesn't contain relevant information, say so clearly
- Be specific and cite relevant sources
- Use clear, professional language
- If you need more information, ask for clarification

Answer:"""
            
            # Generate response
            response = await self.llm.query(prompt)
            answer = response['output']
            
            # Calculate confidence based on source quality
            avg_similarity = sum(chunk['similarity'] for chunk in chunks) / len(chunks)
            confidence = min(0.95, avg_similarity * 1.2)
            
            # Format sources
            sources = []
            for chunk in chunks:
                metadata = chunk['metadata']
                sources.append({
                    'title': metadata.get('title', 'Unknown'),
                    'content': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                    'similarity': chunk['similarity'],
                    'metadata': metadata
                })
            
            return RAGResponse(
                answer=answer,
                sources=sources,
                confidence=confidence,
                query=query
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return RAGResponse(
                answer=f"I encountered an error while processing your question: {str(e)}",
                sources=[],
                confidence=0.0,
                query=query
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database"""
        try:
            count = self.collection.count()
            return {
                'total_chunks': count,
                'embedding_model': self.embedding_model_name,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'vector_db_path': str(self.vector_db_path)
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'error': str(e)}
    
    def clear_database(self):
        """Clear all documents from the vector database"""
        try:
            self.chroma_client.delete_collection("documents")
            self.collection = self.chroma_client.create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Vector database cleared")
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            raise
