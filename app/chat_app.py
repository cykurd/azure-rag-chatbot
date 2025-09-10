"""
RAG Chat Application
Clean, minimal ChatGPT-like interface for document-based question answering
"""

from flask import Flask, render_template, request, jsonify
import json
import os
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')

# Global variables
rag_system = None
documents = []

def load_data():
    """Load sample data"""
    global documents
    
    try:
        with open("../data/sample_documents.json", "r") as f:
            documents = json.load(f)
        logger.info(f"Loaded {len(documents)} documents")
        return True
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return False

def initialize_rag_system():
    """Initialize RAG system"""
    global rag_system
    
    try:
        from models.base_model import BaseModel
        from rag_systems.simple_rag_web import SimpleRAGWeb, Document
        
        # Create enhanced model with advanced RAG capabilities
        model = BaseModel(
            model_name="gpt-4o-mini",
            retrieval_method="enhanced",
            query_expansion=True,
            advanced_reranking=True,
            diversity_selection=True,
            system_prompt_variant="current"
        )
        
        # Convert documents
        doc_objects = []
        for doc in documents:
            doc_objects.append(Document(
                id=doc.get('id', ''),
                content=doc['content'],
                metadata=doc.get('metadata', {})
            ))
        
        # Create RAG system
        rag_system = SimpleRAGWeb(model, doc_objects)
        logger.info("RAG system initialized")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        return False

@app.route('/')
def index():
    """Main chat page"""
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        if not rag_system:
            return jsonify({'error': 'RAG system not available'}), 500
        
        # Get RAG response
        response = rag_system.query(
            user_query=message,
            system_prompt="You are a helpful AI assistant. Answer based on the provided context. Be concise and helpful."
        )
        
        return jsonify({'response': response})
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/status')
def status():
    """Get system status"""
    return jsonify({
        'rag_system_ready': rag_system is not None,
        'documents_loaded': len(documents),
        'api_key_set': bool(os.getenv('OPENAI_API_KEY'))
    })

if __name__ == '__main__':
    # Load data
    if not load_data():
        logger.error("Failed to load data")
        sys.exit(1)
    
    # Initialize RAG system
    initialize_rag_system()
    
    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000)
