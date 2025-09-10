"""
RAG Evaluation Framework for Portfolio Demonstration

A simplified but comprehensive evaluation framework that demonstrates
key evaluation concepts without requiring complex infrastructure.
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
from ..models.base_model import BaseModel

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Result of a single evaluation"""
    query: str
    expected_answer: str
    actual_answer: str
    retrieved_docs: List[str]
    metrics: Dict[str, float]
    evaluation_time: float
    error: Optional[str] = None

@dataclass
class TestCase:
    """A test case for evaluation"""
    id: str
    query: str
    expected_answer: str
    context: str
    category: str

class RAGEvaluator:
    """
    Evaluator for RAG systems using LLM-based metrics
    """
    
    def __init__(self, model_name: str = 'gpt-4'):
        """
        Initialize the evaluator
        
        Args:
            model_name: Name of the model to use for evaluation
        """
        self.llm = BaseModel(model_name=model_name, temperature=0.0)
        logger.info(f"RAG Evaluator initialized with model: {model_name}")
    
    def evaluate_single(self, 
                       query: str,
                       expected_answer: str,
                       actual_answer: str,
                       retrieved_docs: List[str]) -> EvaluationResult:
        """
        Evaluate a single query-response pair
        
        Args:
            query: The user's query
            expected_answer: The expected answer
            actual_answer: The actual answer from the RAG system
            retrieved_docs: List of retrieved document texts
            
        Returns:
            EvaluationResult with metrics
        """
        start_time = time.time()
        
        try:
            # Prepare context for evaluation
            context = "\n\n".join(retrieved_docs) if retrieved_docs else "No documents retrieved"
            
            # Create evaluation prompt
            evaluation_prompt = f"""You are an expert evaluator of RAG (Retrieval-Augmented Generation) systems. Evaluate the following response based on the provided context and expected answer.

Query: {query}

Expected Answer: {expected_answer}

Retrieved Documents:
{context}

Actual Answer: {actual_answer}

Please evaluate the following metrics on a scale of 0.0 to 1.0:

1. Task Success: Does the actual answer successfully complete the task requested in the query? Consider if the user gets what they need.
2. Faithfulness: Is the actual answer consistent with the information in the retrieved documents?
3. Answer Relevancy: Does the actual answer directly and accurately address the query?
4. Context Recall: Do the retrieved documents cover the information present in the expected answer?
5. Context Precision: Do the retrieved documents contain the most relevant information needed to answer the query?

Return ONLY a JSON object with these keys and values (0.0 to 1.0):
{{
    "task_success": 0.0,
    "faithfulness": 0.0,
    "answer_relevancy": 0.0,
    "context_recall": 0.0,
    "context_precision": 0.0
}}"""
            
            # Get evaluation from LLM
            response = self.llm.query(evaluation_prompt)
            evaluation_text = response['output']
            
            # Parse JSON response
            try:
                # Extract JSON from response
                start_idx = evaluation_text.find('{')
                end_idx = evaluation_text.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_text = evaluation_text[start_idx:end_idx]
                    metrics = json.loads(json_text)
                else:
                    raise ValueError("No JSON found in response")
                
                # Validate metrics
                required_metrics = ['task_success', 'faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']
                for metric in required_metrics:
                    if metric not in metrics:
                        metrics[metric] = 0.0
                    else:
                        # Ensure values are between 0 and 1
                        metrics[metric] = max(0.0, min(1.0, float(metrics[metric])))
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse evaluation metrics: {e}")
                # Fallback to default metrics
                metrics = {
                    'task_success': 0.0,
                    'faithfulness': 0.0,
                    'answer_relevancy': 0.0,
                    'context_recall': 0.0,
                    'context_precision': 0.0
                }
            
            evaluation_time = time.time() - start_time
            
            return EvaluationResult(
                query=query,
                expected_answer=expected_answer,
                actual_answer=actual_answer,
                retrieved_docs=retrieved_docs,
                metrics=metrics,
                evaluation_time=evaluation_time
            )
            
        except Exception as e:
            logger.error(f"Error evaluating single case: {e}")
            evaluation_time = time.time() - start_time
            
            return EvaluationResult(
                query=query,
                expected_answer=expected_answer,
                actual_answer=actual_answer,
                retrieved_docs=retrieved_docs,
                metrics={
                    'task_success': 0.0,
                    'faithfulness': 0.0,
                    'answer_relevancy': 0.0,
                    'context_recall': 0.0,
                    'context_precision': 0.0
                },
                evaluation_time=evaluation_time,
                error=str(e)
            )
    
    def evaluate_batch(self, 
                      test_cases: List[TestCase],
                      rag_system,
                      max_cases: Optional[int] = None) -> List[EvaluationResult]:
        """
        Evaluate a batch of test cases
        
        Args:
            test_cases: List of TestCase objects
            rag_system: RAG system to evaluate (must have a query method)
            max_cases: Maximum number of cases to evaluate (for testing)
            
        Returns:
            List of EvaluationResult objects
        """
        if max_cases:
            test_cases = test_cases[:max_cases]
        
        logger.info(f"Evaluating {len(test_cases)} test cases")
        
        results = []
        for i, test_case in enumerate(test_cases):
            logger.info(f"Evaluating case {i+1}/{len(test_cases)}: {test_case.id}")
            
            try:
                # Query the RAG system
                rag_response = rag_system.query(test_case.query)
                
                # Extract retrieved documents
                retrieved_docs = []
                if hasattr(rag_response, 'sources'):
                    retrieved_docs = [source.get('content', '') for source in rag_response.sources]
                elif isinstance(rag_response, dict) and 'sources' in rag_response:
                    retrieved_docs = [source.get('content', '') for source in rag_response['sources']]
                
                # Extract actual answer
                if hasattr(rag_response, 'answer'):
                    actual_answer = rag_response.answer
                elif isinstance(rag_response, dict) and 'answer' in rag_response:
                    actual_answer = rag_response['answer']
                else:
                    actual_answer = str(rag_response)
                
                # Evaluate
                result = self.evaluate_single(
                    query=test_case.query,
                    expected_answer=test_case.expected_answer,
                    actual_answer=actual_answer,
                    retrieved_docs=retrieved_docs
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating test case {test_case.id}: {e}")
                results.append(EvaluationResult(
                    query=test_case.query,
                    expected_answer=test_case.expected_answer,
                    actual_answer="",
                    retrieved_docs=[],
                    metrics={
                        'task_success': 0.0,
                        'faithfulness': 0.0,
                        'answer_relevancy': 0.0,
                        'context_recall': 0.0,
                        'context_precision': 0.0
                    },
                    evaluation_time=0.0,
                    error=str(e)
                ))
        
        logger.info(f"Completed evaluation of {len(results)} test cases")
        return results
    
    def calculate_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Calculate aggregate metrics from evaluation results
        
        Args:
            results: List of EvaluationResult objects
            
        Returns:
            Dictionary with aggregate metrics
        """
        if not results:
            return {}
        
        # Calculate averages for each metric
        metrics = ['task_success', 'faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']
        aggregate = {}
        
        for metric in metrics:
            values = [r.metrics.get(metric, 0.0) for r in results if r.error is None]
            if values:
                aggregate[metric] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
            else:
                aggregate[metric] = {
                    'mean': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'count': 0
                }
        
        # Calculate overall score (weighted average)
        weights = {
            'task_success': 0.4,
            'faithfulness': 0.2,
            'answer_relevancy': 0.2,
            'context_recall': 0.1,
            'context_precision': 0.1
        }
        
        overall_score = sum(
            aggregate[metric]['mean'] * weights[metric] 
            for metric in metrics
        )
        
        aggregate['overall_score'] = overall_score
        
        # Add summary statistics
        total_cases = len(results)
        error_cases = len([r for r in results if r.error is not None])
        successful_cases = total_cases - error_cases
        
        aggregate['summary'] = {
            'total_cases': total_cases,
            'successful_cases': successful_cases,
            'error_cases': error_cases,
            'success_rate': successful_cases / total_cases if total_cases > 0 else 0.0,
            'avg_evaluation_time': sum(r.evaluation_time for r in results) / total_cases if total_cases > 0 else 0.0
        }
        
        return aggregate
    
    def results_to_dataframe(self, results: List[EvaluationResult]) -> pd.DataFrame:
        """
        Convert evaluation results to a pandas DataFrame
        
        Args:
            results: List of EvaluationResult objects
            
        Returns:
            pandas DataFrame with results
        """
        data = []
        for result in results:
            row = {
                'query': result.query,
                'expected_answer': result.expected_answer,
                'actual_answer': result.actual_answer,
                'evaluation_time': result.evaluation_time,
                'error': result.error,
                'num_retrieved_docs': len(result.retrieved_docs)
            }
            
            # Add metrics
            for metric, value in result.metrics.items():
                row[metric] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def save_results(self, 
                    results: List[EvaluationResult], 
                    output_path: str,
                    include_aggregate: bool = True):
        """
        Save evaluation results to files
        
        Args:
            results: List of EvaluationResult objects
            output_path: Base path for output files
            include_aggregate: Whether to include aggregate metrics
        """
        # Save detailed results as CSV
        df = self.results_to_dataframe(results)
        csv_path = f"{output_path}_detailed.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Detailed results saved to: {csv_path}")
        
        # Save aggregate metrics as JSON
        if include_aggregate:
            aggregate = self.calculate_aggregate_metrics(results)
            json_path = f"{output_path}_aggregate.json"
            with open(json_path, 'w') as f:
                json.dump(aggregate, f, indent=2)
            logger.info(f"Aggregate metrics saved to: {json_path}")
        
        # Save raw results as JSON
        raw_data = []
        for result in results:
            raw_data.append({
                'query': result.query,
                'expected_answer': result.expected_answer,
                'actual_answer': result.actual_answer,
                'retrieved_docs': result.retrieved_docs,
                'metrics': result.metrics,
                'evaluation_time': result.evaluation_time,
                'error': result.error
            })
        
        json_path = f"{output_path}_raw.json"
        with open(json_path, 'w') as f:
            json.dump(raw_data, f, indent=2)
        logger.info(f"Raw results saved to: {json_path}")
