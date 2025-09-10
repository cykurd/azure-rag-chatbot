import json
import logging
from datetime import datetime
import io
import os
import asyncio
from azure.identity.aio import DefaultAzureCredential
from azure.identity import DefaultAzureCredential as DefaultAzureCredentialSync
from azure.storage.blob import BlobServiceClient
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.indexes.models import SearchIndexer
from openai import AsyncAzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from urllib.parse import quote
import mimetypes
from dotenv import load_dotenv

load_dotenv()

class TevaGPT:
    def __init__(self, temperature=0.0, top_p=1.0, index_name=None, system_prompt_variant="current", 
                 top_k=5, max_tokens=None):
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt_variant = system_prompt_variant
        self.top_k = top_k
        self.deployment = 'gpt-4-omni'
        self.deployment_name = 'gpt-4-omni'

        # Initialize client with dynamic API version
        self.client = AsyncAzureOpenAI(
            api_version='2024-12-01-preview',
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://cog-gryrq-aik-swce-poc.openai.azure.com"),
            azure_deployment=self.deployment,
            azure_ad_token_provider=get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            )
        )
        
        self.search_service = os.getenv("AZURE_SEARCH_SERVICE", "cog-ktpce-aik-euno-poc")
        self.search_endpoint = f"https://{self.search_service}.search.windows.net"
        self.index_name = index_name or os.getenv("AZURE_AI_SEARCH_INDEX", "uscomm-ana-index")

        # System prompt variants for A/B testing
        self.system_prompts = {
            "current": (
                "You are an AI assistant that helps people find information. Please answer using only the information "
                "provided in this prompt, including previous chat history and retrieved document chunks. Do not include "
                "any other information from your own knowledge or any other sources. If the information is not available "
                "in these specified sources, respond with 'Information not available.' Additionally, outline which docs "
                "you retrieved verbatim."
            ),
            
            "relaxed_refusal": (
                "You are an AI assistant that helps people find information. Please answer using primarily the information "
                "provided in this prompt, including previous chat history and retrieved document chunks. If the answer "
                "cannot be found directly in the retrieved documents, attempt to infer a helpful response using any "
                "partial or related information available. Only if truly no content is relevant, respond with "
                "'Information not available.' Additionally, outline which docs you retrieved verbatim."
            ),
            
            "fallback_chaining": (
                "You are an AI assistant that helps people find information. Please answer using only the information "
                "provided in this prompt, including previous chat history and retrieved document chunks. Do not include "
                "any other information from your own knowledge or any other sources. If you cannot find the information, "
                "suggest one clarifying question or next step the user can take, such as rephrasing their query or "
                "expanding the scope. Additionally, outline which docs you retrieved verbatim."
            ),
            
            "confidence_aware": (
                "You are an AI assistant that helps people find information. Please answer using only the information "
                "provided in this prompt, including previous chat history and retrieved document chunks. Do not include "
                "any other information from your own knowledge or any other sources. If you are uncertain, clearly state "
                "any assumptions or explain why the answer may be incomplete, rather than refusing. Always prioritize "
                "providing the best-available response using retrieved context. Additionally, outline which docs you "
                "retrieved verbatim."
            )
        }
        
        # Set the system prompt based on the variant
        self.system_prompt = self.system_prompts[system_prompt_variant]

    async def process_and_query(self, file_path, query, folder_name=None, filename=None):
        """
        Process a file and query with RAG-based Azure Search integration, supporting folder and file filtering.
        """
        try:
            search_parameters = {
                "endpoint": self.search_endpoint,
                "index_name": self.index_name,
                "authentication": {"type": "system_assigned_managed_identity"},
                "top_n_documents": self.top_k
            }

            if folder_name or filename:
                filters = []
                if folder_name:
                    filters.append(f"metadata_storage_path eq '{quote(folder_name)}/*'")
                if filename:
                    filters.append(f"metadata_storage_name eq '{quote(filename)}'")
                search_parameters["filter"] = " and ".join(filters)

            # Prepare kwargs for API call
            kwargs = {
                'model': self.deployment_name,
                'messages': [
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                'extra_body': {
                    "data_sources": [
                        {
                            "type": "azure_search",
                            "parameters": search_parameters
                        }
                    ]
                }
            }

            kwargs['temperature'] = self.temperature
            kwargs['top_p'] = self.top_p

            response = await self.client.chat.completions.create(**kwargs)
            
            output = response.choices[0].message.content  
    
            return {  
                "output": output,  
                "full_output": response  
            }  
        except Exception as e:
            logging.error(f"Error in process_and_query: {str(e)}")
            return {
                "output": f"Error processing query: {str(e)}",
                "full_output": None
            }

# Updated TevaGPT query_simple method
    async def query_simple(self, query):
        """
        Simple query function that only uses generation without RAG search.
        """
        try:
            # Prepare kwargs for API call
            kwargs = {
                'model': self.deployment_name,
                'messages': [
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            }

            kwargs['temperature'] = self.temperature
            kwargs['top_p'] = self.top_p
            
            response = await self.client.chat.completions.create(**kwargs)
            
            output = response.choices[0].message.content
            
            return {
                "output": output,
                "full_output": response
            }
            
        except Exception as e:
            logging.error(f"Error in query: {str(e)}")
            return {
                "output": f"Error processing query: {str(e)}",
                "full_output": None
            }
