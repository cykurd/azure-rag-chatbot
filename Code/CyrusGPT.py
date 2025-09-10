# `CyrusGPT.py``
import os
import json
import logging
from datetime import datetime
import io
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from azure.identity.aio import DefaultAzureCredential
from azure.identity import DefaultAzureCredential as DefaultAzureCredentialSync
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.indexes.models import SearchIndexer
from azure.search.documents.models import VectorizedQuery
from openai import AsyncAzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from urllib.parse import quote
import mimetypes
from dotenv import load_dotenv
import re

# enhanced retrieval dependencies - imported only when needed (no slowdowns on baseline method)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

load_dotenv()

@dataclass
class RetrievalResult:
    """Result structure for enhanced retrieval analysis"""
    content: str
    title: str
    score: float
    metadata: Dict[str, Any]
    rerank_score: Optional[float] = None
    final_score: Optional[float] = None

class QueryExpander:
    """generates lightweight variations of the query using basic reformulation and keyword extraction
    
    for example:

    Input: What are the latest labeling guidelines for oncology drugs?

    1. Remove leading wh-word and "?"
    2. Remove stopwords

    Reformulated query: latest labeling guidelines oncology drugs

    - This can increase the chance of matching in many common cases:
        - Different document phrasings
        - Stripped-down section headers
        - Alternative formulations that aren't in question-format
    """

    def expand_query(self, query: str) -> List[str]:
        # start with the original query
        variations = [query]

        # if it's a question, remove the leading wh-word to create a possible reformulation
        if '?' in query:
            statement = query.replace('?', '').strip()
            if statement.split()[0].lower() in {'what', 'how', 'when', 'where', 'why'}:
                reformed = ' '.join(statement.split()[1:])
                variations.append(reformed)

        # extract key terms by removing stopwords and short filler words
        key_terms = self._extract_key_terms(query)

        # add a version that's just the core keywords, useful for matching stripped-down titles or headers
        if len(key_terms) > 1:
            variations.append(' '.join(key_terms))

        # remove duplicates
        return list(set(variations))

    def _extract_key_terms(self, query: str) -> List[str]:
        # removes common stopwords and returns the remaining informative words
        # could also use the NLTK ones but these ones suffice
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can'
        }

        # split into words and remove short/common ones
        words = re.findall(r'\b\w+\b', query.lower())
        return [word for word in words if word not in stop_words and len(word) > 2]

class AdvancedReranker:
    # reranks retrieved documents using multiple relevance signals to improve ordering

    def __init__(self):
        # initialize tf-idf vectorizer (lazy-loaded on first use for efficiency)
        self._tfidf_vectorizer = None

    def rerank(self, original_query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        # reranks documents based on a weighted combination of tf-idf similarity, token overlap, title relevance, and original score

        if not results or not SKLEARN_AVAILABLE:
            return results

        for result in results:
            # tf-idf captures semantic similarity based on term frequency
            tfidf_score = self._calculate_tfidf_similarity(original_query, result.content)

            # raw token overlap favors direct matches in wording
            overlap_score = self._calculate_query_overlap(original_query, result.content)

            # checks if the query matches terms in the title (weighted higher than content overlap)
            title_score = self._calculate_title_relevance(original_query, result.title)

            # combine scores using fixed weights
            result.rerank_score = (
                0.4 * tfidf_score +
                0.3 * overlap_score +
                0.2 * title_score +
                0.1 * min(result.score, 1.0)  # original retrieval score (capped at 1.0)
            )

        # sort results by the final rerank score (highest first)
        results.sort(key=lambda x: x.rerank_score or 0, reverse=True)
        return results

    def _calculate_tfidf_similarity(self, query: str, content: str) -> float:
        # computes cosine similarity between query and content using tf-idf vectors

        try:
            if self._tfidf_vectorizer is None:
                self._tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000) # limit it to the top 1000 most frequent terms

            tfidf_matrix = self._tfidf_vectorizer.fit_transform([query, content]) # treat the query and content as a 2-document corpus - build joint vocabulary
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] # return the value between 0 and 1 of similarity (measure the angle between query and content vectors)
            return float(similarity)
        except:
            return 0.0

    def _calculate_query_overlap(self, query: str, content: str) -> float:
        # calculates proportion of query terms that appear in the content

        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        content_terms = set(re.findall(r'\b\w+\b', content.lower()))

        if not query_terms:
            return 0.0

        overlap = len(query_terms.intersection(content_terms))
        return overlap / len(query_terms)

    def _calculate_title_relevance(self, query: str, title: str) -> float:
        # boosts results whose titles contain more query terms

        if not title:
            return 0.0

        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        title_terms = set(re.findall(r'\b\w+\b', title.lower()))

        if not query_terms:
            return 0.0

        overlap = len(query_terms.intersection(title_terms))
        return (overlap / len(query_terms)) * 1.5  # title gets higher weight per match

class DiversitySelector:
    """ selects a diverse subset of top_k results to reduce redundancy among outputs 
    
    basically this is used to ensure users don't get back 5 versions of the same paragraph.

    it favors coverage of different aspects of a topic while still prefering highly relevant results.
    """

    def __init__(self):
        # lazily initialize tf-idf vectorizer for comparing content similarity
        self._content_vectorizer = None

    def select_diverse(self, results: List[RetrievalResult], top_k: int) -> List[RetrievalResult]:
        """ selects top_k results by balancing rerank relevance and semantic diversity """

        if len(results) <= top_k or not SKLEARN_AVAILABLE:
            return results[:top_k]

        selected = []
        remaining = results.copy()

        # start by selecting the top result (highest rerank score)
        if remaining:
            selected.append(remaining.pop(0))

        # iteratively select items that are relevant but dissimilar to what's already selected
        while len(selected) < top_k and remaining:
            best_idx = 0
            best_score = -1

            for i, candidate in enumerate(remaining):
                diversity_penalty = 0

                # penalize candidates that are too similar to already selected results
                for selected_result in selected:
                    similarity = self._content_similarity(candidate.content, selected_result.content)
                    diversity_penalty += similarity

                # subtract diversity penalty from rerank score to compute final score
                final_score = (candidate.rerank_score or 0) - (0.2 * diversity_penalty)

                if final_score > best_score:
                    best_score = final_score
                    best_idx = i

            # select the best tradeoff between relevance and diversity
            selected.append(remaining.pop(best_idx))

        # store final scores (used for formatting/debugging downstream)
        for result in selected:
            result.final_score = result.rerank_score

        return selected

    def _content_similarity(self, content1: str, content2: str) -> float:
        # computes cosine similarity between two content strings using tf-idf vectors

        try:
            # truncate long content for speed and relevance
            c1 = content1[:500].lower()
            c2 = content2[:500].lower()

            if self._content_vectorizer is None:
                self._content_vectorizer = TfidfVectorizer(stop_words='english')

            tfidf_matrix = self._content_vectorizer.fit_transform([c1, c2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0

class CyrusGPT:
    def __init__(self, temperature=0.0, top_p=0.9, index_name='uscomm-ana-index', 
                container_name='uscomm-ana', system_prompt_variant="current", 
                deployment='o4-mini', max_tokens=None, seed=42, top_k=25,
                initial_retrieval_k=25, final_k=20,
                retrieval_method='enhanced', query_expansion=True,
                advanced_reranking=True, diversity_selection=True):
        
        # basic settings
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt_variant = system_prompt_variant
        self.seed = seed
        self.top_k = top_k
        
        # enhanced retrieval settings - automatically disable for 'default' method
        self.retrieval_method = retrieval_method
        if retrieval_method == 'default':
            # force all enhanced features off for default retrieval
            self.query_expansion = False
            self.advanced_reranking = False
            self.diversity_selection = False
            self.initial_retrieval_k = 5
            self.final_k = 5
        else:
            # use provided parameters for non-default methods
            self.query_expansion = query_expansion
            self.advanced_reranking = advanced_reranking
            self.diversity_selection = diversity_selection
            # separate initial retrieval from final selection for enhanced mode
            self.initial_retrieval_k = initial_retrieval_k  # retrieve more initially
            self.final_k = final_k  # cut down after processing

        # lazy-loaded enhanced components
        self._enhanced_components_initialized = False
        self._query_expander = None
        self._advanced_reranker = None
        self._diversity_selector = None

        # map model names to deployment names, API versions, and token limits
        self.model_config = {
            'gpt-4-omni': {
                'deployment': 'gpt-4-omni',
                'api_version': '2024-12-01-preview',
                'max_tokens': 4096,
                'uses_max_completion_tokens': False,
                'azure_search_compatible': True
            },
            'o4-mini': {
                'deployment': 'o4-mini',
                'api_version': '2025-04-01-preview',
                'max_tokens': 65536,
                'uses_max_completion_tokens': True,
                'azure_search_compatible': False
            },
            'o3-mini': {
                'deployment': 'o3-mini',
                'api_version': '2025-03-01-preview',
                'max_tokens': 65536,
                'uses_max_completion_tokens': True,
                'azure_search_compatible': False
            },
            'o3': {
                'deployment': 'o3',
                'api_version': '2025-04-01-preview',
                'max_tokens': 100000,
                'uses_max_completion_tokens': True,
                'azure_search_compatible': False
            },
            'o1': {
                'deployment': 'o1',
                'api_version': '2025-03-01-preview',
                'max_tokens': 100000,
                'uses_max_completion_tokens': True,
                'azure_search_compatible': False
            },
            'o1-mini': {
                'deployment': 'o1-mini',
                'api_version': '2025-03-01-preview',
                'max_tokens': 65536,
                'uses_max_completion_tokens': True,
                'azure_search_compatible': False
            },
            'o1-preview': {
                'deployment': 'o1-preview',
                'api_version': '2025-03-01-preview',
                'max_tokens': 32768,
                'uses_max_completion_tokens': True,
                'azure_search_compatible': False
            },
            'gpt-4.1': {
                'deployment': 'gpt-4.1',
                'api_version': '2024-12-01-preview',
                'max_tokens': 8192,
                'uses_max_completion_tokens': False,
                'azure_search_compatible': True
            },
            'model-router': {
                'deployment': 'model-router',
                'api_version': '2024-12-01-preview',
                'max_tokens': 4096,
                'uses_max_completion_tokens': False,
                'azure_search_compatible': True
            }
        }
        
        # get the model configuration
        model_info = self.model_config.get(deployment, {
            'deployment': deployment,
            'api_version': '2024-05-01-preview',
            'max_tokens': 16384,
            'uses_max_completion_tokens': False,
            'azure_search_compatible': True
        })
        
        self.deployment = model_info['deployment']
        self.api_version = model_info['api_version']
        self.uses_max_completion_tokens = model_info['uses_max_completion_tokens']
        self.azure_search_compatible = model_info['azure_search_compatible']
        self.max_tokens = max_tokens if max_tokens is not None else model_info['max_tokens']

        # setup credentials
        self.credential = DefaultAzureCredentialSync()
        self.async_credential = DefaultAzureCredential()

        # initialize openai client for generation
        self.azure_openai_client = AsyncAzureOpenAI(
            azure_ad_token_provider=get_bearer_token_provider(
                self.async_credential, "https://cognitiveservices.azure.com/.default"
            ),
            api_version=self.api_version,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://cog-gryrq-aik-swce-poc.openai.azure.com")
        )

        # setup embeddings client
        self.embedding_client = AsyncAzureOpenAI(
            azure_ad_token_provider=get_bearer_token_provider(
                self.async_credential, "https://cognitiveservices.azure.com/.default"
            ),
            api_version=self.api_version,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://cog-gryrq-aik-swce-poc.openai.azure.com")
        )

        # Azure Search and Blob Storage setup
        self.search_service = "cog-ktpce-aik-euno-poc"
        self.index_name = index_name
        
        # initialize Azure Search client for retrieval
        self.search_client = SearchClient(
            endpoint=f"https://{self.search_service}.search.windows.net",
            index_name=self.index_name,
            credential=self.credential,
        )
        self.indexer_client = SearchIndexerClient(
            f"https://{self.search_service}.search.windows.net", self.credential
        )

        # setup Blob Storage
        self.container_name = container_name
        self.blob_service_client = BlobServiceClient(
            account_url=os.getenv("AZURE_STORAGE_ACCOUNT_URL", "https://stviovcaikeunopoc.blob.core.windows.net/"),
            credential=self.credential
        )

        # system prompt variants
        self.system_prompts = {
            "current": os.getenv(
                "AZURE_OPENAI_SYSTEM_MESSAGE",
                "You are a helpful AI assistant designed to answer user questions based on the provided MyTeva Pages.\n"
                "Use the retrieved MyTeva Pages labeled like '[docX]: {title}\n{content}\n-----\n[docX]: {title}\n{content}\n to construct clear,accurate responses and additional information, while adhering to the following rules:\n Response Guidelines:\n 1. If the user is asking a follow up question refer to the chat history to answer the user question else follow the next set of rules.\n 2. Priority to choose MyTeva Pages to construct the response:\n - Prioritize retrieved MyTeva pages where the {content} language matches the user's question.\n - If unavailable, fall back to English {content} pages.\n - Use other language pages only if no same-language or English content is available.\n 3. Content Usage:\n - URL markdowns from the retrieved MyTeva pages use them as is. \n - Use relative URLs as is in the response without adding base URLs for them.\n - If the user language is right to left like hebrew or arabic, generate markdown hyperlinks normally and not right to left." 
                "\n 4. Unavailable Information:\n - If MyTeva Pages is empty ([]) or contains no relevant content, respond with:\n > Sorry, we couldn't find the information in MyTeva. Please try rerunning your query using the retry button, refining it, or opening a support ticket for further assistance, especially when seeking personal information.\n (Respond in the same language as the user's question.)\n 5. References & Extras:\n - Always mention which MyTeva Pages (by content) were used to support the answer.\n - Summarize any additional relevant information from the MyTeva Pages if it's related to the user's query.\n 6. Answer Format (Strict Requirement):\n - Write your answer normally in the same language as the user question.\n - After your answer, on a new line, add the references using this format:\n [docX][docX] \n - Where [docX] refers to the document IDs of the retrieved MyTeva pages you used.\n Do not include any titles to the references, or titles like 'Answer' or 'Your Response' in your output.\n Follow this format exactly, even if only one document is used.\n 7. Follow-Up Suggestions (Only if MyTeva Pages Were Found):\n - At the end of your answer, infer two logical follow-up questions the user might ask next.]\n - Present them in this format:\n ### Follow-Up Questions:\n - [Follow-up question 1]\n - [Follow-up question 2]"
            ),
            
            "relaxed_refusal": (
                "You are a helpful AI assistant designed to answer user questions based on the provided MyTeva Pages.\n"
                "Use the retrieved MyTeva Pages labeled like '[docX]: {title}\n{content}\n-----\n[docX]: {title}\n{content}\n to construct clear,accurate responses and additional information, while adhering to the following rules:\n Response Guidelines:\n 1. If the user is asking a follow up question refer to the chat history to answer the user question else follow the next set of rules.\n 2. Priority to choose MyTeva Pages to construct the response:\n - Prioritize retrieved MyTeva pages where the {content} language matches the user's question.\n - If unavailable, fall back to English {content} pages.\n - Use other language pages only if no same-language or English content is available.\n 3. Content Usage:\n - URL markdowns from the retrieved MyTeva pages use them as is. \n - Use relative URLs as is in the response without adding base URLs for them.\n - If the user language is right to left like hebrew or arabic, generate markdown hyperlinks normally and not right to left." 
                "\n 4. Unavailable Information:\n - If the answer cannot be found directly in the retrieved documents, attempt to infer a helpful response using any partial or related information available. Only if truly no content is relevant, respond with: 'Information not available'\n 5. References & Extras:\n - Always mention which MyTeva Pages (by content) were used to support the answer.\n - Summarize any additional relevant information from the MyTeva Pages if it's related to the user's query.\n 6. Answer Format (Strict Requirement):\n - Write your answer normally in the same language as the user question.\n - After your answer, on a new line, add the references using this format:\n [docX][docX] \n - Where [docX] refers to the document IDs of the retrieved MyTeva pages you used.\n Do not include any titles to the references, or titles like 'Answer' or 'Your Response' in your output.\n Follow this format exactly, even if only one document is used.\n 7. Follow-Up Suggestions (Only if MyTeva Pages Were Found):\n - At the end of your answer, infer two logical follow-up questions the user might ask next.]\n - Present them in this format:\n ### Follow-Up Questions:\n - [Follow-up question 1]\n - [Follow-up question 2]"
            ),
            
            "fallback_chaining": (
                "You are a helpful AI assistant designed to answer user questions based on the provided MyTeva Pages.\n"
                "Use the retrieved MyTeva Pages labeled like '[docX]: {title}\n{content}\n-----\n[docX]: {title}\n{content}\n to construct clear,accurate responses and additional information, while adhering to the following rules:\n Response Guidelines:\n 1. If the user is asking a follow up question refer to the chat history to answer the user question else follow the next set of rules.\n 2. Priority to choose MyTeva Pages to construct the response:\n - Prioritize retrieved MyTeva pages where the {content} language matches the user's question.\n - If unavailable, fall back to English {content} pages.\n - Use other language pages only if no same-language or English content is available.\n 3. Content Usage:\n - URL markdowns from the retrieved MyTeva pages use them as is. \n - Use relative URLs as is in the response without adding base URLs for them.\n - If the user language is right to left like hebrew or arabic, generate markdown hyperlinks normally and not right to left."
                "\n 4. Unavailable Information:\n - If you cannot find the information, suggest one clarifying question or next step the user can take, such as rephrasing their query or expanding the scope\n 5. References & Extras:\n - Always mention which MyTeva Pages (by content) were used to support the answer.\n - Summarize any additional relevant information from the MyTeva Pages if it's related to the user's query.\n 6. Answer Format (Strict Requirement):\n - Write your answer normally in the same language as the user question.\n - After your answer, on a new line, add the references using this format:\n [docX][docX] \n - Where [docX] refers to the document IDs of the retrieved MyTeva pages you used.\n Do not include any titles to the references, or titles like 'Answer' or 'Your Response' in your output.\n Follow this format exactly, even if only one document is used.\n 7. Follow-Up Suggestions (Only if MyTeva Pages Were Found):\n - At the end of your answer, infer two logical follow-up questions the user might ask next.]\n - Present them in this format:\n ### Follow-Up Questions:\n - [Follow-up question 1]\n - [Follow-up question 2]"
            ),
            
            "confidence_aware": (
                "You are a helpful AI assistant designed to answer user questions based on the provided MyTeva Pages.\n"
                "Use the retrieved MyTeva Pages labeled like '[docX]: {title}\n{content}\n-----\n[docX]: {title}\n{content}\n to construct clear,accurate responses and additional information, while adhering to the following rules:\n Response Guidelines:\n 1. If the user is asking a follow up question refer to the chat history to answer the user question else follow the next set of rules.\n 2. Priority to choose MyTeva Pages to construct the response:\n - Prioritize retrieved MyTeva pages where the {content} language matches the user's question.\n - If unavailable, fall back to English {content} pages.\n - Use other language pages only if no same-language or English content is available.\n 3. Content Usage:\n - URL markdowns from the retrieved MyTeva pages use them as is. \n - Use relative URLs as is in the response without adding base URLs for them.\n - If the user language is right to left like hebrew or arabic, generate markdown hyperlinks normally and not right to left."
                "\n 4. Unavailable Information:\n - If you are uncertain, clearly state any assumptions or explain why the answer may be incomplete, rather than refusing. Always prioritize providing the best-available response using retrieved context\n 5. References & Extras:\n - Always mention which MyTeva Pages (by content) were used to support the answer.\n - Summarize any additional relevant information from the MyTeva Pages if it's related to the user's query.\n 6. Answer Format (Strict Requirement):\n - Write your answer normally in the same language as the user question.\n - After your answer, on a new line, add the references using this format:\n [docX][docX] \n - Where [docX] refers to the document IDs of the retrieved MyTeva pages you used.\n Do not include any titles to the references, or titles like 'Answer' or 'Your Response' in your output.\n Follow this format exactly, even if only one document is used.\n 7. Follow-Up Suggestions (Only if MyTeva Pages Were Found):\n - At the end of your answer, infer two logical follow-up questions the user might ask next.]\n - Present them in this format:\n ### Follow-Up Questions:\n - [Follow-up question 1]\n - [Follow-up question 2]"
            ),
            "basic": (
            "You are a helpful AI assistant designed to answer user questions based on the provided documents."
            ),
            "last": (
            """You are CyrusGPT, an advanced AI assistant with sophisticated document retrieval and reasoning capabilities. Your primary function is to provide accurate, comprehensive, and contextually relevant responses based on retrieved documents and your extensive knowledge base.

            ## Core Principles

            **Accuracy & Reliability**: Always prioritize factual accuracy. When information conflicts between sources, explicitly note the discrepancy and explain which source appears more authoritative or recent. Never fabricate information not present in the retrieved documents.

            **Contextual Intelligence**: Analyze the user's intent beyond surface-level queries. Consider implicit needs, domain context, and potential follow-up requirements. Adapt your response style to match the complexity and formality appropriate for the topic.

            **Source Integration**: Seamlessly weave information from multiple documents while maintaining clear attribution. Synthesize rather than simply concatenate - identify patterns, contradictions, and complementary information across sources.

            ## Response Framework

            **Immediate Value**: Lead with the most direct answer to the user's question. Avoid lengthy preambles - get to the core information quickly while maintaining thoroughness.

            **Evidence-Based Reasoning**: Support all claims with specific references to retrieved documents. When making inferences, clearly distinguish between what the documents explicitly state versus reasonable conclusions drawn from the evidence.

            **Structured Clarity**: Organize complex information hierarchically. Use natural language flow rather than bullet points unless explicitly requested. Employ clear transitions and logical progression of ideas.

            **Completeness with Precision**: Provide comprehensive coverage of the topic while avoiding information overload. Include relevant details that enhance understanding without overwhelming the user.

            ## Advanced Capabilities

            **Cross-Document Analysis**: When multiple documents address the same topic, identify consensus views, highlight disagreements, and note any evolution in thinking or policy over time.

            **Gap Identification**: Explicitly acknowledge when retrieved documents don't fully address the user's question. Suggest what additional information might be helpful and indicate your confidence level in partial answers.

            **Contextual Recommendations**: Based on the user's query and retrieved information, proactively suggest related areas of inquiry that might be valuable for their broader objectives.

            **Technical Precision**: For specialized domains, use appropriate terminology while ensuring accessibility. Define complex concepts when first introduced, and maintain consistency in technical language throughout the response.

            ## Quality Assurance

            **Source Validation**: Prioritize information from authoritative, recent, and relevant sources. When source quality varies, weight your response accordingly and note any limitations.

            **Logical Consistency**: Ensure all parts of your response align logically. If presenting multiple perspectives, clearly delineate between them and avoid conflating different viewpoints.

            **Actionable Intelligence**: When appropriate, translate information into practical next steps or recommendations. Help users understand not just what the information means, but how they might apply it.

            **Confidence Calibration**: Express appropriate levels of certainty. Use precise language to indicate when you're highly confident versus when you're making educated inferences from limited information.

            ## Interaction Excellence

            **Adaptive Communication**: Match the user's communication style and expertise level. Provide more technical detail for expert users, more context for novices, while maintaining accuracy in both cases.

            **Anticipatory Assistance**: Consider what follow-up questions the user might have and address them proactively when space permits, or explicitly suggest valuable next steps.

            **Error Handling**: When documents are incomplete, contradictory, or insufficient, clearly explain the limitations rather than attempting to fill gaps with general knowledge unless explicitly appropriate.

            **Continuous Improvement**: Learn from the specific documents and user interactions within each session to provide increasingly relevant and useful responses as the conversation progresses.

            Remember: Your goal is not just to answer questions, but to provide insights that enable better decision-making and deeper understanding. Every response should add genuine value beyond what users could achieve through simple document search alone."""
        ),
        "upd_calc_refusal": (
            "You are a helpful AI assistant designed to answer user questions based on the provided MyTeva Pages.\n"
            "Use the retrieved MyTeva Pages labeled like '[docX]: {title}\n{content}\n-----\n[docX]: {title}\n{content}\n to construct clear,accurate responses and additional information, while adhering to the following rules:\n Response Guidelines:\n 1. If the user is asking a follow up question refer to the chat history to answer the user question else follow the next set of rules.\n 2. Priority to choose MyTeva Pages to construct the response:\n - Prioritize retrieved MyTeva pages where the {content} language matches the user's question.\n - If unavailable, fall back to English {content} pages.\n - Use other language pages only if no same-language or English content is available.\n 3. Content Usage:\n - URL markdowns from the retrieved MyTeva pages use them as is. \n - Use relative URLs as is in the response without adding base URLs for them.\n - If the user language is right to left like hebrew or arabic, generate markdown hyperlinks normally and not right to left."
            "\n 4. Calculations and Analysis:\n - When users request calculations and you have relevant numerical data in the retrieved MyTeva Pages, perform the requested mathematical operations step-by-step using the provided information.\n - For calculations with uncertain parameters (like ranges or estimates), provide both lower and upper bound results when appropriate. For example, if a growth rate could be 4-5%, calculate and present both scenarios.\n - Show your work clearly, including the source data from the documents and each calculation step.\n 5. Unavailable Information:\n - If MyTeva Pages is empty ([]) or contains no relevant content for non-calculation questions, respond with:\n > Sorry, we couldn't find the information in MyTeva. Please try rerunning your query using the retry button, refining it, or opening a support ticket for further assistance, especially when seeking personal information.\n (Respond in the same language as the user's question.)\n 6. References & Extras:\n - Always mention which MyTeva Pages (by content) were used to support the answer.\n - Summarize any additional relevant information from the MyTeva Pages if it's related to the user's query.\n 7. Answer Format (Strict Requirement):\n - Write your answer normally in the same language as the user question.\n - After your answer, on a new line, add the references using this format:\n [docX][docX] \n - Where [docX] refers to the document IDs of the retrieved MyTeva pages you used.\n Do not include any titles to the references, or titles like 'Answer' or 'Your Response' in your output.\n Follow this format exactly, even if only one document is used.\n 8. Follow-Up Suggestions (Only if MyTeva Pages Were Found):\n - At the end of your answer, infer two logical follow-up questions the user might ask next.]\n - Present them in this format:\n ### Follow-Up Questions:\n - [Follow-up question 1]\n - [Follow-up question 2]"
        )

        }
        
        # set the system prompt based on the variant
        self.system_prompt = self.system_prompts[system_prompt_variant]

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

    def get_indexer_name_for_index(self, index_name):
        """Map index names to their corresponding indexer names"""
        indexer_mapping = {
            'uscomm-ana-index': 'uscomm-ana-indexer',
            'cyrus-index': 'cyrus-indexer',
            'cyrus-index-ada-002': 'cyrus-indexer-ada-002',
            'cyrus-index-3-large': 'cyrus-indexer-3-large', 
            'cyrus-index-3-small': 'cyrus-indexer-3-small',
            'cyrus-page-number-index-ada-002': 'cyrus-page-number-indexer-ada-002',
            'cyrus-page-number-index-3-large': 'cyrus-page-number-indexer-3-large',
            'cyrus-page-number-index-3-small': 'cyrus-page-number-indexer-3-small'
        }
        return indexer_mapping.get(index_name, f"{index_name}-indexer") # fallback to the default naming convention

    async def upload_file(self, file_path, folder_name=None, is_overwrite=True):
        """ uploads a file to blob storage, optionally inside a folder, with overwrite control"""
        
        try:
            # get container client from blob service
            container_client = self.blob_service_client.get_container_client(self.container_name)
            
            # determine blob name with optional folder prefix
            blob_name = os.path.basename(file_path) if folder_name is None else f'{folder_name}/{os.path.basename(file_path)}'
            
            # get blob client for the computed blob name
            blob_client = container_client.get_blob_client(blob_name)
            
            # check if blob exists and overwrite is disabled
            if not is_overwrite and blob_client.exists():
                raise ValueError(f'Blob {blob_name} already exists.')
            
            # upload file as binary blob
            with open(file_path, 'rb') as file:
                blob_client.upload_blob(file, overwrite=is_overwrite)
            
            return True

        except Exception as e:
            # log and re-raise any error during upload
            logging.error(f'Error uploading file {file_path}: {e}')
            raise

    async def trigger_indexer(self):
        """ triggers a search indexer to run, or creates it if it doesn't exist """
        try:
            # get indexer name for current index
            indexer_name = self.get_indexer_name_for_index(self.index_name)
            
            # if indexer exists, run it
            if indexer_name in self.indexer_client.get_indexer_names():
                self.indexer_client.run_indexer(indexer_name)

            else:
                # define indexer config if it doesn't exist
                indexer_dict = {
                    'name': indexer_name,
                    'dataSourceName': f'{self.container_name}-data-source',
                    'targetIndexName': self.index_name,
                    'skillsetName': f'{self.container_name}-general-skillset' if os.getenv('chunking', 'true') == 'true' else None
                }
                
                # create indexer from dict and run it
                indexer = SearchIndexer.from_dict(indexer_dict)
                self.indexer_client.create_indexer(indexer)
                self.indexer_client.run_indexer(indexer_name)

        except Exception as e:
            # log and re-raise any error during indexer trigger
            logging.error(f'Error triggering indexer {indexer_name} for index {self.index_name}: {e}')
            raise

    async def query(self, query, folder_name=None, filename=None):
        """ Main query method """
        try:
            # build filter string for restricting retrieval to folder and/or filename
            filter_str = None
            if folder_name or filename:
                filters = []
                if folder_name:
                    filters.append(f"metadata_storage_path eq '{folder_name}/*'")
                if filename:
                    filters.append(f"metadata_storage_name eq '{filename}'")
                filter_str = " and ".join(filters)

            # use enhanced retrieval if configured
            if self.retrieval_method == 'enhanced':
                return await self._query_with_enhanced_retrieval(query, filter_str)
            
            # fallback to default azure search-compatible method
            if self.azure_search_compatible:
                # configure azure search parameters
                search_parameters = {
                    "endpoint": f"https://{self.search_service}.search.windows.net",
                    "index_name": self.index_name,
                    "authentication": {"type": "system_assigned_managed_identity"},
                    "top_n_documents": self.initial_retrieval_k if self.retrieval_method == 'enhanced' else self.top_k  # conditional logic
                }
                if filter_str:
                    search_parameters["filter"] = filter_str

                # prepare chat completion call with data source and query context
                kwargs = {
                    'model': self.deployment,
                    'messages': [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": query}
                    ],
                    'extra_body': {
                        "data_sources": [{"type": "azure_search", "parameters": search_parameters}]
                    },
                    'max_tokens': self.max_tokens,
                    'temperature': self.temperature,
                    'top_p': self.top_p,
                    'seed': self.seed
                }

                # send request and return formatted response
                response = await self.azure_openai_client.chat.completions.create(**kwargs)
                return {"output": response.choices[0].message.content, "full_output": response}
            
            else:
                # simple rag flow for o-series reasoning models (non-azure-search-compatible)
                
                # step 1: set up client to run gpt-4o retrieval using azure search
                retrieval_client = AsyncAzureOpenAI(
                    azure_ad_token_provider=get_bearer_token_provider(
                        self.async_credential, "https://cognitiveservices.azure.com/.default"
                    ),
                    api_version='2024-12-01-preview',  # gpt-4o api version
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://cog-gryrq-aik-swce-poc.openai.azure.com")
                )

                # configure azure search parameters for retrieval
                search_parameters = {
                    "endpoint": f"https://{self.search_service}.search.windows.net",
                    "index_name": self.index_name,
                    "authentication": {"type": "system_assigned_managed_identity"},
                    "top_n_documents": self.initial_retrieval_k if self.retrieval_method == 'enhanced' else self.top_k  #conditional logic
                }                
                if filter_str:
                    search_parameters["filter"] = filter_str

                # define retrieval prompt and parameters
                retrieval_kwargs = {
                    'model': 'gpt-4-omni',  # retrieval always uses gpt-4o
                    'messages': [
                        {"role": "system", "content": "You are a helpful assistant. Retrieve relevant documents and provide them as context."},
                        {"role": "user", "content": query}
                    ],
                    'extra_body': {
                        "data_sources": [{"type": "azure_search", "parameters": search_parameters}]
                    },
                    'max_tokens': 4096,
                    'temperature': 0.0
                }

                # call gpt-4o to get relevant documents
                retrieval_response = await retrieval_client.chat.completions.create(**retrieval_kwargs)
                
                # step 2: extract citations (retrieved docs) from the response
                citations = []
                if hasattr(retrieval_response.choices[0].message, 'context'):
                    citations = retrieval_response.choices[0].message.context.get('citations', [])

                # step 3: format retrieved docs into a readable context block
                if citations:
                    docs = []
                    for i, citation in enumerate(citations, 1):
                        title = citation.get('title', 'Untitled')
                        content = citation.get('content', '')
                        docs.append(f"[doc{i}]: {title}\n{content}")
                    context = "\n-----\n".join(docs)
                else:
                    context = "[]"  # no documents retrieved

                # step 4: run final reasoning model (e.g. o4-mini) with context + user query
                reasoning_kwargs = {
                    'model': self.deployment,  # reasoning model to use
                    'messages': [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"{context}\n\n{query}"}
                    ]
                }

                # apply appropriate generation parameters based on model support
                if self.uses_max_completion_tokens:
                    reasoning_kwargs['max_completion_tokens'] = self.max_tokens
                    # o-series models don’t support temp, top_p, seed
                else:
                    reasoning_kwargs['max_tokens'] = self.max_tokens
                    reasoning_kwargs['temperature'] = self.temperature
                    reasoning_kwargs['top_p'] = self.top_p
                    reasoning_kwargs['seed'] = self.seed

                # send the final reasoning query
                reasoning_response = await self.azure_openai_client.chat.completions.create(**reasoning_kwargs)

                # step 5: inject citations back into response object for consistency in evaluation
                if citations:
                    class CitationContext:
                        def __init__(self, citations):
                            self.citations = citations
                        def get(self, key, default=None):
                            return self.citations if key == 'citations' else default

                    class MessageWithCitations:
                        def __init__(self, original_message, citations):
                            for attr in dir(original_message):
                                if not attr.startswith('_'):
                                    setattr(self, attr, getattr(original_message, attr))
                            self.context = CitationContext(citations)

                    reasoning_response.choices[0].message = MessageWithCitations(
                        reasoning_response.choices[0].message, citations
                    )

                # return final output and full response object
                return {
                    "output": reasoning_response.choices[0].message.content,
                    "full_output": reasoning_response
                }

        except Exception as e:
            # catch and log any exceptions during the query process
            logging.error(f"Error in query method: {e}")
            raise

    async def _query_with_enhanced_retrieval(self, query, filter_str=None):
        """Enhanced retrieval pipeline + query"""
        try:
            # initialize enhanced components
            self._initialize_enhanced_components()
            
            # step 1: get base retrieval results using original method
            if self.azure_search_compatible:
                base_results = await self._get_azure_search_results(query, filter_str)
            else:
                base_results = await self._get_reasoning_model_results(query, filter_str)
            
            # step 2: apply enhanced processing if enabled
            if base_results and self.retrieval_method == 'enhanced':
                processed_results = await self._apply_enhanced_processing(query, base_results)
                
                # step 3: re-create the response with enhanced context but maintain original format
                return await self._create_enhanced_response(query, processed_results)
            else:
                # fallback to original results
                return base_results

        except Exception as e:
            logging.error(f"Enhanced retrieval error: {e}")
            # fallback to original query method
            return await self._fallback_to_original_query(query, filter_str)

    async def _get_azure_search_results(self, query, filter_str=None):
        """Get results using Azure Search extension (original method)"""
        try:
            # define azure search parameters for document retrieval
            search_parameters = {
                "endpoint": f"https://{self.search_service}.search.windows.net",
                "index_name": self.index_name,
                "authentication": {"type": "system_assigned_managed_identity"},
                "top_n_documents": self.initial_retrieval_k  # use initial_retrieval_k instead of self.top_k
            }
            if filter_str:
                search_parameters["filter"] = filter_str

            # set up request to openai with search-backed retrieval context
            kwargs = {
                'model': self.deployment,
                'messages': [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query}
                ],
                'extra_body': {
                    "data_sources": [{"type": "azure_search", "parameters": search_parameters}]
                },
                'max_tokens': self.max_tokens,
                'temperature': self.temperature,
                'top_p': self.top_p,
                'seed': self.seed
            }

            # run the completion call using azure search as retrieval context
            response = await self.azure_openai_client.chat.completions.create(**kwargs)
            return {"output": response.choices[0].message.content, "full_output": response}
            
        except Exception as e:
            # log and re-raise if retrieval or completion fails
            logging.error(f"Azure Search results error: {e}")
            raise

    async def _get_reasoning_model_results(self, query, filter_str=None):
        """Get results using reasoning model RAG"""
        try:
            # set up client to use gpt-4o for retrieval via azure search
            retrieval_client = AsyncAzureOpenAI(
                azure_ad_token_provider=get_bearer_token_provider(
                    self.async_credential, "https://cognitiveservices.azure.com/.default"
                ),
                api_version='2024-12-01-preview',
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://cog-gryrq-aik-swce-poc.openai.azure.com")
            )

            # define azure search parameters for document retrieval
            search_parameters = {
                "endpoint": f"https://{self.search_service}.search.windows.net",
                "index_name": self.index_name,
                "authentication": {"type": "system_assigned_managed_identity"},
                "top_n_documents": self.initial_retrieval_k  # use initial_retrieval_k instead of self.top_k
            }
            if filter_str:
                search_parameters["filter"] = filter_str

            # prepare retrieval query using gpt-4o
            retrieval_kwargs = {
                'model': 'gpt-4-omni',
                'messages': [
                    {"role": "system", "content": "You are a helpful assistant. Retrieve relevant documents and provide them as context."},
                    {"role": "user", "content": query}
                ],
                'extra_body': {
                    "data_sources": [{"type": "azure_search", "parameters": search_parameters}]
                },
                'max_tokens': 4096,
                'temperature': 0.0
            }

            # run retrieval call and get document citations
            retrieval_response = await retrieval_client.chat.completions.create(**retrieval_kwargs)
            
            # extract citations from the gpt-4o response
            citations = []
            if hasattr(retrieval_response.choices[0].message, 'context'):
                citations = retrieval_response.choices[0].message.context.get('citations', [])

            # format citations into a readable context string
            if citations:
                docs = []
                for i, citation in enumerate(citations, 1):
                    title = citation.get('title', 'Untitled')
                    content = citation.get('content', '')
                    docs.append(f"[doc{i}]: {title}\n{content}")
                context = "\n-----\n".join(docs)
            else:
                context = "[]"

            # construct reasoning prompt using retrieved context + query
            reasoning_kwargs = {
                'model': self.deployment,
                'messages': [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"{context}\n\n{query}"}
                ]
            }

            # set generation parameters depending on model type
            if self.uses_max_completion_tokens:
                reasoning_kwargs['max_completion_tokens'] = self.max_tokens
            else:
                reasoning_kwargs['max_tokens'] = self.max_tokens
                reasoning_kwargs['temperature'] = self.temperature
                reasoning_kwargs['top_p'] = self.top_p
                reasoning_kwargs['seed'] = self.seed

            # run the reasoning model with the constructed prompt
            reasoning_response = await self.azure_openai_client.chat.completions.create(**reasoning_kwargs)

            # manually attach citations back to message for downstream evals
            if citations:
                class CitationContext:
                    def __init__(self, citations):
                        self.citations = citations
                    def get(self, key, default=None):
                        return self.citations if key == 'citations' else default

                class MessageWithCitations:
                    def __init__(self, original_message, citations):
                        for attr in dir(original_message):
                            if not attr.startswith('_'):
                                setattr(self, attr, getattr(original_message, attr))
                        self.context = CitationContext(citations)

                reasoning_response.choices[0].message = MessageWithCitations(
                    reasoning_response.choices[0].message, citations
                )

            # return final output and full response object
            return {
                "output": reasoning_response.choices[0].message.content,
                "full_output": reasoning_response
            }

        except Exception as e:
            # log and re-raise on any failure
            logging.error(f"Reasoning model results error: {e}")
            raise


    async def _apply_enhanced_processing(self, query, base_results):
        """Apply enhanced processing to base results"""
        try:
            # extract citations from the base model output
            citations = []
            if hasattr(base_results['full_output'].choices[0].message, 'context'):
                citations = base_results['full_output'].choices[0].message.context.get('citations', [])

            # if no citations found, skip enhanced processing
            if not citations:
                return base_results

            retrieval_results = []
            total = len(citations)
            for rank, citation in enumerate(citations):
                retrieval_results.append(RetrievalResult(
                    content=citation.get('content', ''),
                    title=citation.get('title', ''),
                    score = 1.0 - (rank / total),  # normalized rank-based score
                    metadata=citation
                ))


            # initialize processed results
            processed_results = retrieval_results

            # optional: expand query terms and adjust scores based on overlap
            if self.query_expansion and self._query_expander:
                expanded_queries = self._query_expander.expand_query(query)
                for result in processed_results:
                    for expanded_query in expanded_queries:
                        overlap = self._calculate_overlap(expanded_query, result.content)
                        result.score = max(result.score, overlap)

            # optional: rerank results using advanced reranker
            if self.advanced_reranking and self._advanced_reranker:
                processed_results = self._advanced_reranker.rerank(query, processed_results)

            # always truncate to final_k regardless of diversity selection (if enhanced - since this is the enhanced function)
            if len(processed_results) > self.final_k:
                processed_results = processed_results[:self.final_k]

            # optional: apply diversity-based selection to final results
            if self.diversity_selection and self._diversity_selector:
                processed_results = self._diversity_selector.select_diverse(processed_results, self.final_k)# use final_k instead of self.top_k

            return processed_results

        except Exception as e:
            # log and fall back to original results on error
            logging.error(f"Enhanced processing error: {e}")
            return base_results

    def _calculate_overlap(self, query, content):
        """Simple unigram overlap calculation for scoring"""
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        content_terms = set(re.findall(r'\b\w+\b', content.lower()))
        
        if not query_terms:
            return 0.0
        
        overlap = len(query_terms.intersection(content_terms))
        return overlap / len(query_terms)

    async def _create_enhanced_response(self, query, processed_results):
        """Create response with enhanced results but maintain original format"""
        try:
            # if enhanced processing was skipped, return base results directly
            if isinstance(processed_results, dict):
                return processed_results

            # convert enhanced RetrievalResult objects into citation dictionaries
            enhanced_citations = []
            for result in processed_results:
                enhanced_citations.append({
                    'title': result.title,
                    'content': result.content,
                    'confidence': getattr(result, 'final_score', getattr(result, 'rerank_score', result.score))
                })

            # build context string from enhanced citations
            if enhanced_citations:
                docs = []
                for i, citation in enumerate(enhanced_citations, 1):
                    title = citation.get('title', 'Untitled')
                    content = citation.get('content', '')
                    confidence = citation.get('confidence', '')
                    confidence_str = f" (confidence: {confidence:.3f})" if confidence else ""
                    docs.append(f"[doc{i}]: {title}{confidence_str}\n{content}")
                context = "\n-----\n".join(docs)
            else:
                context = "[]"

            # generate final model response using the enhanced context
            kwargs = {
                'model': self.deployment,
                'messages': [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"{context}\n\n{query}"}
                ]
            }

            # set token and sampling parameters depending on model support
            if self.uses_max_completion_tokens:
                kwargs['max_completion_tokens'] = self.max_tokens
            else:
                kwargs['max_tokens'] = self.max_tokens
                kwargs['temperature'] = self.temperature
                kwargs['top_p'] = self.top_p
                kwargs['seed'] = self.seed

            # run chat completion with enriched inputs
            response = await self.azure_openai_client.chat.completions.create(**kwargs)

            # attach citations to message for downstream evaluation or inspection
            if enhanced_citations:
                class CitationContext:
                    def __init__(self, citations):
                        self.citations = citations
                    def get(self, key, default=None):
                        return self.citations if key == 'citations' else default

                class MessageWithCitations:
                    def __init__(self, original_message, citations):
                        for attr in dir(original_message):
                            if not attr.startswith('_'):
                                setattr(self, attr, getattr(original_message, attr))
                        self.context = CitationContext(citations)

                response.choices[0].message = MessageWithCitations(
                    response.choices[0].message, enhanced_citations
                )

            # return the enhanced output and full raw response
            return {
                "output": response.choices[0].message.content,
                "full_output": response
            }

        except Exception as e:
            # log and propagate error during enhanced response generation
            logging.error(f"Enhanced response creation error: {e}")
            raise

    async def _fallback_to_original_query(self, query, filter_str=None):
        """Fallback to original query method"""
        try:
            # temporarily disable enhanced retrieval for fallback
            original_retrieval_method = self.retrieval_method
            self.retrieval_method = 'default'
            
            # call original query logic
            result = await self.query(query, 
                                     folder_name=filter_str.split("'")[1].split("/*")[0] if filter_str and "metadata_storage_path" in filter_str else None,
                                     filename=filter_str.split("'")[1] if filter_str and "metadata_storage_name" in filter_str else None)
            
            # restore original retrieval method
            self.retrieval_method = original_retrieval_method
            
            return result
            
        except Exception as e:
            logging.error(f"Fallback query error: {e}")
            raise

    async def query_simple(self, query):
        """ simple query function that only uses generation without RAG search """
        try:
            kwargs = {
                'model': self.deployment,
                'messages': [
                    {
                        "role": "system", 
                        "content": "You are an AI assistant that helps people find information. Please answer using only the information provided in the user's message. Do not include any other information from your own knowledge or any other sources."
                    },
                    {
                        "role": "user", 
                        "content": query
                    }
                ]
            }

            if self.uses_max_completion_tokens:
                kwargs['max_completion_tokens'] = self.max_tokens
            else:
                kwargs['max_tokens'] = self.max_tokens
                kwargs['temperature'] = self.temperature
                kwargs['top_p'] = self.top_p
                kwargs['seed'] = self.seed

            response = await self.azure_openai_client.chat.completions.create(**kwargs)

            return {
                "output": response.choices[0].message.content,
                "full_output": response
            }

        except Exception as e:
            logging.error(f"Error in query_simple: {e}")
            return {
                "output": f"Error processing query: {str(e)}",
                "full_output": None
            }
