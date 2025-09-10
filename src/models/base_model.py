"""
Advanced RAG-Enabled Language Model with Comprehensive Retrieval and Generation Capabilities

This module provides a sophisticated base model implementation with advanced RAG features
including query expansion, reranking, diversity selection, and multiple retrieval methods.
"""

import os
import json
import logging
import re
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from openai import OpenAI
from dotenv import load_dotenv

# Enhanced retrieval dependencies - imported only when needed
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Result structure for enhanced retrieval analysis"""
    content: str
    title: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    rerank_score: Optional[float] = None
    final_score: Optional[float] = None

class QueryExpander:
    """Generates lightweight variations of the query using basic reformulation and keyword extraction
    
    For example:
    Input: What are the latest labeling guidelines for oncology drugs?
    
    1. Remove leading wh-word and "?"
    2. Remove stopwords
    
    Reformulated query: latest labeling guidelines oncology drugs
    
    This can increase the chance of matching in many common cases:
        - Different document phrasings
        - Stripped-down section headers
        - Alternative formulations that aren't in question-format
    """

    def expand_query(self, query: str) -> List[str]:
        # Start with the original query
        variations = [query]

        # If it's a question, remove the leading wh-word to create a possible reformulation
        if '?' in query:
            statement = query.replace('?', '').strip()
            if statement.split()[0].lower() in {'what', 'how', 'when', 'where', 'why'}:
                reformed = ' '.join(statement.split()[1:])
                variations.append(reformed)

        # Extract key terms by removing stopwords and short filler words
        key_terms = self._extract_key_terms(query)

        # Add a version that's just the core keywords, useful for matching stripped-down titles or headers
        if len(key_terms) > 1:
            variations.append(' '.join(key_terms))

        # Remove duplicates
        return list(set(variations))

    def _extract_key_terms(self, query: str) -> List[str]:
        # Removes common stopwords and returns the remaining informative words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can'
        }

        # Split into words and remove short/common ones
        words = re.findall(r'\b\w+\b', query.lower())
        return [word for word in words if word not in stop_words and len(word) > 2]

class AdvancedReranker:
    """Reranks retrieved documents using multiple relevance signals to improve ordering"""

    def __init__(self):
        # Initialize tf-idf vectorizer (lazy-loaded on first use for efficiency)
        self._tfidf_vectorizer = None

    def rerank(self, original_query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Reranks documents based on a weighted combination of tf-idf similarity, token overlap, title relevance, and original score"""

        if not results or not SKLEARN_AVAILABLE:
            return results

        for result in results:
            # TF-IDF captures semantic similarity based on term frequency
            tfidf_score = self._calculate_tfidf_similarity(original_query, result.content)

            # Raw token overlap favors direct matches in wording
            overlap_score = self._calculate_query_overlap(original_query, result.content)

            # Checks if the query matches terms in the title (weighted higher than content overlap)
            title_score = self._calculate_title_relevance(original_query, result.title)

            # Combine scores using fixed weights
            result.rerank_score = (
                0.4 * tfidf_score +
                0.3 * overlap_score +
                0.2 * title_score +
                0.1 * min(result.score, 1.0)  # Original retrieval score (capped at 1.0)
            )

        # Sort results by the final rerank score (highest first)
        results.sort(key=lambda x: x.rerank_score or 0, reverse=True)
        return results

    def _calculate_tfidf_similarity(self, query: str, content: str) -> float:
        """Computes cosine similarity between query and content using tf-idf vectors"""

        try:
            if self._tfidf_vectorizer is None:
                self._tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

            tfidf_matrix = self._tfidf_vectorizer.fit_transform([query, content])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0

    def _calculate_query_overlap(self, query: str, content: str) -> float:
        """Calculates proportion of query terms that appear in the content"""

        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        content_terms = set(re.findall(r'\b\w+\b', content.lower()))

        if not query_terms:
            return 0.0

        overlap = len(query_terms.intersection(content_terms))
        return overlap / len(query_terms)

    def _calculate_title_relevance(self, query: str, title: str) -> float:
        """Boosts results whose titles contain more query terms"""

        if not title:
            return 0.0

        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        title_terms = set(re.findall(r'\b\w+\b', title.lower()))

        if not query_terms:
            return 0.0

        overlap = len(query_terms.intersection(title_terms))
        return (overlap / len(query_terms)) * 1.5  # Title gets higher weight per match

class DiversitySelector:
    """Selects a diverse subset of top_k results to reduce redundancy among outputs 
    
    This is used to ensure users don't get back 5 versions of the same paragraph.
    It favors coverage of different aspects of a topic while still preferring highly relevant results.
    """

    def __init__(self):
        # Lazily initialize tf-idf vectorizer for comparing content similarity
        self._content_vectorizer = None

    def select_diverse(self, results: List[RetrievalResult], top_k: int) -> List[RetrievalResult]:
        """Selects top_k results by balancing rerank relevance and semantic diversity"""

        if len(results) <= top_k or not SKLEARN_AVAILABLE:
            return results[:top_k]

        selected = []
        remaining = results.copy()

        # Start by selecting the top result (highest rerank score)
        if remaining:
            selected.append(remaining.pop(0))

        # Iteratively select items that are relevant but dissimilar to what's already selected
        while len(selected) < top_k and remaining:
            best_idx = 0
            best_score = -1

            for i, candidate in enumerate(remaining):
                diversity_penalty = 0

                # Penalize candidates that are too similar to already selected results
                for selected_result in selected:
                    similarity = self._content_similarity(candidate.content, selected_result.content)
                    diversity_penalty += similarity

                # Subtract diversity penalty from rerank score to compute final score
                final_score = (candidate.rerank_score or 0) - (0.2 * diversity_penalty)

                if final_score > best_score:
                    best_score = final_score
                    best_idx = i

            # Select the best tradeoff between relevance and diversity
            selected.append(remaining.pop(best_idx))

        # Store final scores (used for formatting/debugging downstream)
        for result in selected:
            result.final_score = result.rerank_score

        return selected

    def _content_similarity(self, content1: str, content2: str) -> float:
        """Computes cosine similarity between two content strings using tf-idf vectors"""

        try:
            # Truncate long content for speed and relevance
            c1 = content1[:500].lower()
            c2 = content2[:500].lower()

            if self._content_vectorizer is None:
                self._content_vectorizer = TfidfVectorizer(stop_words='english')

            tfidf_matrix = self._content_vectorizer.fit_transform([c1, c2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0

class BaseModel:
    """Advanced RAG-enabled language model with comprehensive retrieval and generation capabilities"""
    
    def __init__(self, 
                 model_name: str = "gpt-4o-mini", 
                 api_key: str = None, 
                 base_url: str = None, 
                 temperature: float = 0.0,
                 top_p: float = 0.9,
                 max_tokens: int = None,
                 seed: int = 42,
                 top_k: int = 25,
                 initial_retrieval_k: int = 25,
                 final_k: int = 20,
                 retrieval_method: str = 'enhanced',
                 query_expansion: bool = True,
                 advanced_reranking: bool = True,
                 diversity_selection: bool = True,
                 system_prompt_variant: str = "current"):
        
        # Basic settings
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_API_BASE")
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.top_k = top_k
        self.system_prompt_variant = system_prompt_variant
        
        # Enhanced retrieval settings - automatically disable for 'default' method
        self.retrieval_method = retrieval_method
        if retrieval_method == 'default':
            # Force all enhanced features off for default retrieval
            self.query_expansion = False
            self.advanced_reranking = False
            self.diversity_selection = False
            self.initial_retrieval_k = 5
            self.final_k = 5
        else:
            # Use provided parameters for non-default methods
            self.query_expansion = query_expansion
            self.advanced_reranking = advanced_reranking
            self.diversity_selection = diversity_selection
            # Separate initial retrieval from final selection for enhanced mode
            self.initial_retrieval_k = initial_retrieval_k  # Retrieve more initially
            self.final_k = final_k  # Cut down after processing

        # Lazy-loaded enhanced components
        self._enhanced_components_initialized = False
        self._query_expander = None
        self._advanced_reranker = None
        self._diversity_selector = None

        # Model configuration with token limits and capabilities
        self.model_config = {
            'gpt-4o-mini': {
                'max_tokens': 16384,
                'supports_temperature': True,
                'supports_top_p': True,
                'supports_seed': True
            },
            'gpt-4o': {
                'max_tokens': 128000,
                'supports_temperature': True,
                'supports_top_p': True,
                'supports_seed': True
            },
            'gpt-4-turbo': {
                'max_tokens': 128000,
                'supports_temperature': True,
                'supports_top_p': True,
                'supports_seed': True
            },
            'gpt-3.5-turbo': {
                'max_tokens': 16384,
                'supports_temperature': True,
                'supports_top_p': True,
                'supports_seed': True
            }
        }
        
        # Get the model configuration
        model_info = self.model_config.get(model_name, {
            'max_tokens': 16384,
            'supports_temperature': True,
            'supports_top_p': True,
            'supports_seed': True
        })
        
        self.max_tokens = max_tokens if max_tokens is not None else model_info['max_tokens']
        self.supports_temperature = model_info['supports_temperature']
        self.supports_top_p = model_info['supports_top_p']
        self.supports_seed = model_info['supports_seed']

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be provided or set as an environment variable.")
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # System prompt variants
        self.system_prompts = {
            "current": (
                "You are a helpful AI assistant designed to answer user questions based on the provided documents.\n"
                "Use the retrieved documents to construct clear, accurate responses while adhering to the following rules:\n"
                "1. Prioritize information from the provided documents\n"
                "2. If information is not available in the documents, clearly state this\n"
                "3. Always cite which documents were used to support your answer\n"
                "4. Provide comprehensive responses that address the user's question thoroughly\n"
                "5. If multiple documents contain relevant information, synthesize the information coherently"
            ),
            "basic": "You are a helpful AI assistant that answers questions based on provided documents.",
            "detailed": (
                "You are an advanced AI assistant with sophisticated document analysis capabilities. "
                "Your primary function is to provide accurate, comprehensive, and contextually relevant responses "
                "based on retrieved documents and your knowledge base. Always prioritize factual accuracy, "
                "provide evidence-based reasoning, and maintain structured clarity in your responses."
            ),
            "concise": (
                "You are a helpful AI assistant. Answer questions based on the provided documents. "
                "Be direct and concise while ensuring accuracy. Cite sources when relevant."
            ),
            "analytical": (
                "You are an analytical AI assistant that provides in-depth analysis of documents. "
                "Your responses should include: 1) Direct answers to questions, 2) Supporting evidence from documents, "
                "3) Analysis of patterns or trends across multiple sources, 4) Identification of potential gaps or limitations, "
                "and 5) Suggestions for further investigation when appropriate."
            ),
            "educational": (
                "You are an educational AI assistant that helps users understand complex topics through document analysis. "
                "Break down complex information into digestible parts, provide clear explanations, use examples when helpful, "
                "and ensure users can follow your reasoning. Always cite your sources and explain any technical terms."
            )
        }
        
        # Set the system prompt based on the variant
        self.system_prompt = self.system_prompts[system_prompt_variant]
        
        logger.info(f"Initialized BaseModel with {model_name} using {retrieval_method} retrieval")

    def _initialize_enhanced_components(self):
        """Lazy initialization of enhanced components"""
        if not self._enhanced_components_initialized and self.retrieval_method == 'enhanced':
            if self.query_expansion:
                self._query_expander = QueryExpander()
            if self.advanced_reranking:
                self._advanced_reranker = AdvancedReranker()
            if self.diversity_selection:
                self._diversity_selector = DiversitySelector()
            self._enhanced_components_initialized = True

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Basic generation method for simple text completion"""
        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens
        }
        
        # Add sampling parameters only if supported by the model
        if self.supports_temperature:
            kwargs["temperature"] = self.temperature
        if self.supports_top_p:
            kwargs["top_p"] = self.top_p
        if self.supports_seed:
            kwargs["seed"] = self.seed
        
        try:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def query_with_rag(self, query: str, documents: List[Dict[str, Any]], 
                      use_enhanced_retrieval: bool = True) -> str:
        """Advanced RAG query with optional enhanced retrieval processing"""
        try:
            if not documents:
                return "No documents available for retrieval."
            
            # Convert documents to RetrievalResult objects
            retrieval_results = []
            for i, doc in enumerate(documents):
                retrieval_results.append(RetrievalResult(
                    content=doc.get('content', ''),
                    title=doc.get('title', f'Document {i+1}'),
                    score=1.0 - (i / len(documents)),  # Simple rank-based scoring
                    metadata=doc.get('metadata', {})
                ))
            
            # Apply enhanced processing if enabled
            if use_enhanced_retrieval and self.retrieval_method == 'enhanced':
                self._initialize_enhanced_components()
                processed_results = self._apply_enhanced_processing(query, retrieval_results)
            else:
                processed_results = retrieval_results[:self.final_k]
            
            # Build context from processed results
            context = self._build_context(processed_results)
            
            # Generate response with context
            full_prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer based on the context provided."
            return self.generate(full_prompt)
            
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return f"Error processing query: {str(e)}"

    def _apply_enhanced_processing(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Apply enhanced processing to retrieval results"""
        try:
            processed_results = results.copy()
            
            # Optional: expand query terms and adjust scores based on overlap
            if self.query_expansion and self._query_expander:
                expanded_queries = self._query_expander.expand_query(query)
                for result in processed_results:
                    for expanded_query in expanded_queries:
                        overlap = self._calculate_overlap(expanded_query, result.content)
                        result.score = max(result.score, overlap)
            
            # Optional: rerank results using advanced reranker
            if self.advanced_reranking and self._advanced_reranker:
                processed_results = self._advanced_reranker.rerank(query, processed_results)
            
            # Always truncate to final_k regardless of diversity selection
            if len(processed_results) > self.final_k:
                processed_results = processed_results[:self.final_k]
            
            # Optional: apply diversity-based selection to final results
            if self.diversity_selection and self._diversity_selector:
                processed_results = self._diversity_selector.select_diverse(processed_results, self.final_k)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Enhanced processing error: {e}")
            return results[:self.final_k]

    def _calculate_overlap(self, query: str, content: str) -> float:
        """Simple unigram overlap calculation for scoring"""
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        content_terms = set(re.findall(r'\b\w+\b', content.lower()))
        
        if not query_terms:
            return 0.0
        
        overlap = len(query_terms.intersection(content_terms))
        return overlap / len(query_terms)

    def _build_context(self, results: List[RetrievalResult]) -> str:
        """Build context string from retrieval results"""
        if not results:
            return "[]"
        
        docs = []
        for i, result in enumerate(results, 1):
            confidence = getattr(result, 'final_score', getattr(result, 'rerank_score', result.score))
            confidence_str = f" (confidence: {confidence:.3f})" if confidence else ""
            docs.append(f"[doc{i}]: {result.title}{confidence_str}\n{result.content}")
        
        return "\n-----\n".join(docs)

    def query_simple(self, query: str) -> str:
        """Simple query function that only uses generation without RAG search"""
        try:
            simple_system_prompt = (
                "You are an AI assistant that helps people find information. "
                "Please answer using only the information provided in the user's message. "
                "Do not include any other information from your own knowledge or any other sources."
            )
            return self.generate(query, system_prompt=simple_system_prompt)
        except Exception as e:
            logger.error(f"Error in simple query: {e}")
            return f"Error processing query: {str(e)}"

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat with the model using a list of messages"""
        try:
            response_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens
            }
            
            # Add sampling parameters only if supported by the model
            if self.supports_temperature:
                response_kwargs["temperature"] = self.temperature
            if self.supports_top_p:
                response_kwargs["top_p"] = self.top_p
            if self.supports_seed:
                response_kwargs["seed"] = self.seed
            
            # Add any additional kwargs
            response_kwargs.update(kwargs)
            
            response = self.client.chat.completions.create(**response_kwargs)
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration"""
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "retrieval_method": self.retrieval_method,
            "enhanced_features": {
                "query_expansion": self.query_expansion,
                "advanced_reranking": self.advanced_reranking,
                "diversity_selection": self.diversity_selection
            },
            "retrieval_parameters": {
                "initial_retrieval_k": self.initial_retrieval_k,
                "final_k": self.final_k
            },
            "system_prompt_variant": self.system_prompt_variant,
            "api_key_set": bool(self.api_key)
        }

    def update_config(self, **kwargs):
        """Update model configuration"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated {key} to {value}")
            else:
                logger.warning(f"Unknown configuration parameter: {key}")

    def set_system_prompt_variant(self, variant: str):
        """Set the system prompt variant"""
        if variant in self.system_prompts:
            self.system_prompt_variant = variant
            self.system_prompt = self.system_prompts[variant]
            logger.info(f"Updated system prompt variant to {variant}")
        else:
            logger.warning(f"Unknown system prompt variant: {variant}")

    def get_available_system_prompts(self) -> List[str]:
        """Get list of available system prompt variants"""
        return list(self.system_prompts.keys())

# Factory function for creating model instances
def create_model(model_name: str = "gpt-4o-mini", **kwargs) -> BaseModel:
    """
    Factory function to create a model instance with advanced RAG capabilities.
    
    Args:
        model_name: Name of the model to create
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured BaseModel instance with enhanced retrieval capabilities
    """
    return BaseModel(model_name=model_name, **kwargs)