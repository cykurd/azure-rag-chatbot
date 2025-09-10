# `Ask_HR_dynamic.py`
import os
import json
import logging
from datetime import datetime
import io
import asyncio
from azure.identity.aio import DefaultAzureCredential
from azure.identity import DefaultAzureCredential as DefaultAzureCredentialSync
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.indexes.models import SearchIndexer
from openai import AsyncAzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from urllib.parse import quote
import mimetypes
from dotenv import load_dotenv

load_dotenv()

class AskHR:
    def __init__(self, temperature=0.0, top_p=1.0, index_name='uscomm-ana-index', 
                 container_name='uscomm-ana', system_prompt_variant="current", 
                 deployment='gpt-4-omni', max_tokens=None, seed=42, top_k=5):
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt_variant = system_prompt_variant
        self.seed = seed
        self.top_k = top_k

        # Map model names to deployment names, API versions, and token limits
        # CRITICAL: The 'uses_max_completion_tokens' flag determines which parameter to use:
        # - True: Use max_completion_tokens (for o-series reasoning models)
        # - False: Use max_tokens (for all other models including GPT-4, GPT-4o, etc.)
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
                'api_version': '2025-04-01-preview',  # Required for o4-mini
                'max_tokens': 65536,
                'uses_max_completion_tokens': True,  # o-series models use max_completion_tokens
                'azure_search_compatible': False  # Azure Search doesn't support o-series params yet
            },
            'o3-mini': {
                'deployment': 'o3-mini',
                'api_version': '2025-03-01-preview',  # Required for o3-mini
                'max_tokens': 65536,
                'uses_max_completion_tokens': True,  # o-series models use max_completion_tokens
                'azure_search_compatible': False  # Azure Search doesn't support o-series params yet
            },
            'o3': {
                'deployment': 'o3',
                'api_version': '2025-04-01-preview',  # Required for o3
                'max_tokens': 100000,
                'uses_max_completion_tokens': True,  # o-series models use max_completion_tokens
                'azure_search_compatible': False  # Azure Search doesn't support o-series params yet
            },
            'o1': {
                'deployment': 'o1',
                'api_version': '2025-03-01-preview',  # Required for o1
                'max_tokens': 100000,
                'uses_max_completion_tokens': True,  # o-series models use max_completion_tokens
                'azure_search_compatible': False  # Azure Search doesn't support o-series params yet
            },
            'o1-mini': {
                'deployment': 'o1-mini',
                'api_version': '2025-03-01-preview',  # Required for o1-mini
                'max_tokens': 65536,
                'uses_max_completion_tokens': True,  # o-series models use max_completion_tokens
                'azure_search_compatible': False  # Azure Search doesn't support o-series params yet
            },
            'o1-preview': {
                'deployment': 'o1-preview',
                'api_version': '2025-03-01-preview',  # Required for o1-preview
                'max_tokens': 32768,
                'uses_max_completion_tokens': True,  # o-series models use max_completion_tokens
                'azure_search_compatible': False  # Azure Search doesn't support o-series params yet
            },
            'gpt-4.1': {
                'deployment': 'gpt-4.1',
                'api_version': '2024-12-01-preview',
                'max_tokens': 8192,
                'uses_max_completion_tokens': False,  # GPT-4.1 uses max_tokens, not max_completion_tokens
                'azure_search_compatible': True
            },
            'model-router': {
                'deployment': 'model-router',
                'api_version': '2024-12-01-preview',
                'max_tokens': 4096,
                'uses_max_completion_tokens': False,  # Router uses max_tokens
                'azure_search_compatible': True
            }
        }
        
        # Get the model configuration
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
        # Use provided max_tokens or model default
        self.max_tokens = max_tokens if max_tokens is not None else model_info['max_tokens']

        # setup credentials
        self.credential = DefaultAzureCredentialSync()
        self.async_credential = DefaultAzureCredential()

        # initialize openai client for generation with dynamic API version
        self.azure_openai_client = AsyncAzureOpenAI(
            azure_ad_token_provider=get_bearer_token_provider(
                self.async_credential, "https://cognitiveservices.azure.com/.default"
            ),
            api_version=self.api_version,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://cog-gryrq-aik-swce-poc.openai.azure.com")
        )

        # setup embeddings in case we need them with dynamic API version
        self.embedding_client = AsyncAzureOpenAI(
            azure_ad_token_provider=get_bearer_token_provider(
                self.async_credential, "https://cognitiveservices.azure.com/.default"
            ),
            api_version=self.api_version,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://cog-gryrq-aik-swce-poc.openai.azure.com")
        )

        # Rest of initialization remains the same...
        self.search_service = "cog-ktpce-aik-euno-poc"
        self.index_name = index_name
        
        # Initialize azure search client for retrieval
        self.search_client = SearchClient(
            endpoint=f"https://{self.search_service}.search.windows.net",
            index_name=self.index_name,
            credential=self.credential,
        )
        self.indexer_client = SearchIndexerClient(
            f"https://{self.search_service}.search.windows.net", self.credential
        )

        # setup blob
        self.container_name = container_name
        self.blob_service_client = BlobServiceClient(
            account_url=os.getenv("AZURE_STORAGE_ACCOUNT_URL", "https://stviovcaikeunopoc.blob.core.windows.net/"),
            credential=self.credential
        )

        # System prompt variants for A/B testing
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
            )
        }
        
        # Set the system prompt based on the variant
        self.system_prompt = self.system_prompts[system_prompt_variant]
        
    def get_indexer_name_for_index(self, index_name):
        """
        Map index names to their corresponding indexer names based on your naming convention
        """
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
        return indexer_mapping.get(index_name, f"{index_name}-indexer")

    async def upload_file(self, file_path, folder_name=None, is_overwrite=True):
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            blob_name = os.path.basename(file_path) if folder_name is None else f"{folder_name}/{os.path.basename(file_path)}"
            blob_client = container_client.get_blob_client(blob_name)
            if not is_overwrite and blob_client.exists():
                raise ValueError(f"Blob {blob_name} already exists.")
            with open(file_path, "rb") as file:
                blob_client.upload_blob(file, overwrite=is_overwrite)
            return True
        except Exception as e:
            logging.error(f"Error uploading file {file_path}: {e}")
            raise

    async def trigger_indexer(self):
        try:
            # Get the correct indexer name for this index
            indexer_name = self.get_indexer_name_for_index(self.index_name)
            
            if indexer_name in self.indexer_client.get_indexer_names():
                self.indexer_client.run_indexer(indexer_name)
            else:
                # If indexer doesn't exist, create it with proper naming
                # Note: This assumes your data sources and skillsets follow the pattern
                indexer_dict = {
                    "name": indexer_name,
                    "dataSourceName": f"{self.container_name}-data-source",  # uscomm-ana-data-source
                    "targetIndexName": self.index_name,  # Use actual index name
                    "skillsetName": f"{self.container_name}-general-skillset" if os.getenv("chunking", "true") == "true" else None
                }
                indexer = SearchIndexer.from_dict(indexer_dict)
                self.indexer_client.create_indexer(indexer)
                self.indexer_client.run_indexer(indexer_name)
        except Exception as e:
            logging.error(f"Error triggering indexer {indexer_name} for index {self.index_name}: {e}")
            raise

    async def query(self, query, folder_name=None, filename=None):
        """
        Simple query method that works for both Azure Search compatible and reasoning models
        """
        try:
            # Build filter string for retrieval constraints
            filter_str = None
            if folder_name or filename:
                filters = []
                if folder_name:
                    filters.append(f"metadata_storage_path eq '{folder_name}/*'")
                if filename:
                    filters.append(f"metadata_storage_name eq '{filename}'")
                filter_str = " and ".join(filters)

            if self.azure_search_compatible:
                # Use Azure Search extension for compatible models (GPT-4o, GPT-4.1, etc.)
                search_parameters = {
                    "endpoint": f"https://{self.search_service}.search.windows.net",
                    "index_name": self.index_name,
                    "authentication": {"type": "system_assigned_managed_identity"},
                    "top_n_documents": self.top_k
                }
                if filter_str:
                    search_parameters["filter"] = filter_str

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

                response = await self.azure_openai_client.chat.completions.create(**kwargs)
                return {"output": response.choices[0].message.content, "full_output": response}
            
            else:
                # Simple RAG for reasoning models (o-series)
                # Step 1: Use GPT-4o to retrieve documents via Azure Search
                retrieval_client = AsyncAzureOpenAI(
                    azure_ad_token_provider=get_bearer_token_provider(
                        self.async_credential, "https://cognitiveservices.azure.com/.default"
                    ),
                    api_version='2024-12-01-preview',  # GPT-4o API version
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://cog-gryrq-aik-swce-poc.openai.azure.com")
                )

                search_parameters = {
                    "endpoint": f"https://{self.search_service}.search.windows.net",
                    "index_name": self.index_name,
                    "authentication": {"type": "system_assigned_managed_identity"},
                    "top_n_documents": self.top_k
                }
                if filter_str:
                    search_parameters["filter"] = filter_str

                # Use GPT-4o for retrieval
                retrieval_kwargs = {
                    'model': 'gpt-4-omni',  # Always use GPT-4o for retrieval
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

                retrieval_response = await retrieval_client.chat.completions.create(**retrieval_kwargs)
                
                # Step 2: Extract citations from GPT-4o response
                citations = []
                if hasattr(retrieval_response.choices[0].message, 'context'):
                    citations = retrieval_response.choices[0].message.context.get('citations', [])

                # Step 3: Format documents as context
                if citations:
                    docs = []
                    for i, citation in enumerate(citations, 1):
                        title = citation.get('title', 'Untitled')
                        content = citation.get('content', '')
                        docs.append(f"[doc{i}]: {title}\n{content}")
                    context = "\n-----\n".join(docs)
                else:
                    context = "[]"  # No documents found

                # Step 4: Use reasoning model with retrieved context
                reasoning_kwargs = {
                    'model': self.deployment,  # o4-mini, o1, etc.
                    'messages': [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"{context}\n\n{query}"}
                    ]
                }

                # Add appropriate token parameters for reasoning models
                if self.uses_max_completion_tokens:
                    reasoning_kwargs['max_completion_tokens'] = self.max_tokens
                    # o-series models don't support temperature, top_p, seed
                else:
                    reasoning_kwargs['max_tokens'] = self.max_tokens
                    reasoning_kwargs['temperature'] = self.temperature
                    reasoning_kwargs['top_p'] = self.top_p
                    reasoning_kwargs['seed'] = self.seed

                reasoning_response = await self.azure_openai_client.chat.completions.create(**reasoning_kwargs)

                # Step 5: Manually add citations to response for evaluation consistency
                if citations:
                    class MockContext:
                        def __init__(self, citations):
                            self.citations = citations
                        def get(self, key, default=None):
                            return self.citations if key == 'citations' else default

                    class MockMessage:
                        def __init__(self, original_message, citations):
                            for attr in dir(original_message):
                                if not attr.startswith('_'):
                                    setattr(self, attr, getattr(original_message, attr))
                            self.context = MockContext(citations)

                    reasoning_response.choices[0].message = MockMessage(reasoning_response.choices[0].message, citations)

                return {
                    "output": reasoning_response.choices[0].message.content,
                    "full_output": reasoning_response
                }

        except Exception as e:
            logging.error(f"Error in query method: {e}")
            raise

    async def query_simple(self, query):
        """
        Simple query function that only uses generation without RAG search.
        Args:
            query (str): Query to ask.
        Returns:
            dict: Response with output and full_output keys.
        """
        try:
            # Prepare kwargs for API call
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

            # CRITICAL FIX: Use the correct parameter based on model type
            if self.uses_max_completion_tokens:
                # o-series models (o1, o3, o4, etc.) use max_completion_tokens
                kwargs['max_completion_tokens'] = self.max_tokens
                # o-series models don't support temperature, top_p, seed
            else:
                # All other models (GPT-4, GPT-4o, GPT-4.1, etc.) use max_tokens
                kwargs['max_tokens'] = self.max_tokens
                kwargs['temperature'] = self.temperature
                kwargs['top_p'] = self.top_p
                # Only add seed for non-o-series models
                kwargs['seed'] = self.seed

            response = await self.azure_openai_client.chat.completions.create(**kwargs)

            output = response.choices[0].message.content

            return {
                "output": output,
                "full_output": response
            }

        except Exception as e:
            logging.error(f"Error in query_simple: {e}")
            return {
                "output": f"Error processing query: {str(e)}",
                "full_output": None
            }
