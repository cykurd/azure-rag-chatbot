"""
Test script for the LLM Portfolio system

This script validates that all components work correctly
and demonstrates the key functionality.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from rag_systems.simple_rag import SimpleRAG, Document
from evaluation.rag_evaluator import RAGEvaluator, TestCase
from models.base_model import BaseModel

async def test_base_model():
    """Test the base model functionality"""
    print("Testing BaseModel...")
    
    try:
        model = BaseModel(model_name='gpt-4', temperature=0.0)
        
        # Test simple query
        response = await model.query("What is machine learning?")
        
        if response['output'] and not response['output'].startswith("Error"):
            print("✓ BaseModel query successful")
            return True
        else:
            print(f"✗ BaseModel query failed: {response['output']}")
            return False
            
    except Exception as e:
        print(f"✗ BaseModel test failed: {e}")
        return False

def test_rag_system():
    """Test the RAG system functionality"""
    print("Testing SimpleRAG...")
    
    try:
        # Initialize RAG system
        rag = SimpleRAG(
            vector_db_path="data/test_vector_db",
            embedding_model="all-MiniLM-L6-v2"
        )
        
        # Create test documents
        test_docs = [
            Document(
                id="test_001",
                title="Test Document 1",
                content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
                metadata={"category": "AI", "difficulty": "beginner"}
            ),
            Document(
                id="test_002", 
                title="Test Document 2",
                content="Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language.",
                metadata={"category": "NLP", "difficulty": "intermediate"}
            )
        ]
        
        # Add documents
        rag.add_documents(test_docs)
        
        # Test retrieval
        chunks = rag.retrieve("What is machine learning?", top_k=2)
        
        if chunks and len(chunks) > 0:
            print("✓ RAG system retrieval successful")
            print(f"  Retrieved {len(chunks)} chunks")
            return True
        else:
            print("✗ RAG system retrieval failed")
            return False
            
    except Exception as e:
        print(f"✗ RAG system test failed: {e}")
        return False

async def test_evaluation_framework():
    """Test the evaluation framework"""
    print("Testing RAG Evaluator...")
    
    try:
        # Initialize evaluator
        evaluator = RAGEvaluator(model_name='gpt-4')
        
        # Create test case
        test_case = TestCase(
            id="eval_test_001",
            query="What is machine learning?",
            expected_answer="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            context="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            category="AI"
        )
        
        # Test evaluation
        result = evaluator.evaluate_single(
            query=test_case.query,
            expected_answer=test_case.expected_answer,
            actual_answer="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            retrieved_docs=["Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data."]
        )
        
        if result.metrics and not result.error:
            print("✓ Evaluation framework successful")
            print(f"  Task Success: {result.metrics.get('task_success', 0):.3f}")
            print(f"  Faithfulness: {result.metrics.get('faithfulness', 0):.3f}")
            return True
        else:
            print(f"✗ Evaluation framework failed: {result.error}")
            return False
            
    except Exception as e:
        print(f"✗ Evaluation framework test failed: {e}")
        return False

async def test_full_rag_pipeline():
    """Test the complete RAG pipeline"""
    print("Testing Full RAG Pipeline...")
    
    try:
        # Initialize RAG system
        rag = SimpleRAG(
            vector_db_path="data/test_vector_db",
            embedding_model="all-MiniLM-L6-v2"
        )
        
        # Load sample documents
        data_dir = Path(__file__).parent / "data"
        with open(data_dir / "sample_documents.json", "r") as f:
            documents_data = json.load(f)
        
        documents = []
        for doc_data in documents_data[:2]:  # Use first 2 documents for testing
            documents.append(Document(
                id=doc_data["id"],
                content=doc_data["content"],
                title=doc_data["title"],
                metadata=doc_data["metadata"]
            ))
        
        # Add documents
        rag.add_documents(documents)
        
        # Test query
        response = await rag.query("What are the main types of machine learning?", top_k=3)
        
        if response.answer and response.confidence > 0:
            print("✓ Full RAG pipeline successful")
            print(f"  Answer length: {len(response.answer)} characters")
            print(f"  Confidence: {response.confidence:.3f}")
            print(f"  Sources: {len(response.sources)}")
            return True
        else:
            print("✗ Full RAG pipeline failed")
            return False
            
    except Exception as e:
        print(f"✗ Full RAG pipeline test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("=" * 50)
    print("LLM Portfolio System Test")
    print("=" * 50)
    
    tests = [
        ("BaseModel", test_base_model()),
        ("RAG System", test_rag_system()),
        ("Evaluation Framework", test_evaluation_framework()),
        ("Full RAG Pipeline", test_full_rag_pipeline())
    ]
    
    results = []
    for test_name, test_coro in tests:
        print(f"\n{test_name}:")
        if asyncio.iscoroutine(test_coro):
            result = await test_coro
        else:
            result = test_coro
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the configuration and dependencies.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
