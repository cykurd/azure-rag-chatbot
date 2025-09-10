# `BaseModel.py`
import os
import json
import logging
from datetime import datetime
import io
import asyncio
from azure.identity.aio import DefaultAzureCredential
from azure.identity import DefaultAzureCredential as DefaultAzureCredentialSync
from openai import AsyncAzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv

load_dotenv()

class BaseModel:
    def __init__(self, model='gpt-4-omni', temperature=0.0, top_p=1.0):
        """
        Initialize BaseModel with configurable model selection.
        
        Args:
            model (str): Model deployment name ('gpt-4-omni', 'o4-mini', 'gpt-4.1', or 'model-router')
            temperature (float): Temperature setting for generation
            top_p (float): Top-p setting for generation
        """
        self.temperature = temperature
        self.top_p = top_p
        self.model = model
        
        # Map model names to deployment names and API versions
        self.model_config = {
            'gpt-4-omni': {
                'deployment': 'gpt-4-omni',
                'api_version': '2024-12-01-preview'
            },
            'o4-mini': {
                'deployment': 'o4-mini',
                'api_version': '2024-12-01-preview'
            },
            'gpt-4.1': {
                'deployment': 'gpt-4.1',
                'api_version': '2025-04-14'
            },
            'model-router': {
                'deployment': 'model-router',
                'api_version': '2025-05-19'
            }
        }
        
        # Get the model configuration
        model_info = self.model_config.get(model, {
            'deployment': model,
            'api_version': '2024-12-01-preview'  # Default API version
        })
        
        self.deployment_name = model_info['deployment']
        self.api_version = model_info['api_version']
        
        # Initialize Azure OpenAI client with appropriate API version
        self.client = AsyncAzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://cog-gryrq-aik-swce-poc.openai.azure.com"),
            azure_deployment=self.deployment_name,
            azure_ad_token_provider=get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            )
        )
        
        # Simple system prompt
        self.system_prompt = "You are a helpful assistant."
    
    async def query(self, query, messages=None):
        try:
            if messages is None:
                messages = []

            conversation_messages = [
                { 'role': 'system', 'content': self.system_prompt },
                *messages,
                { 'role': 'user', 'content': query }
            ]

            kwargs = {
                'model': self.deployment_name,
                'messages': conversation_messages
            }

            # Only add temperature and top_p for models that support them
            # o4-mini and some newer models may not support these parameters
            if self.model not in ['o4-mini']:
                kwargs['temperature'] = self.temperature
                kwargs['top_p'] = self.top_p

            response = await self.client.chat.completions.create(**kwargs)
            output = response.choices[0].message.content

            return {
                'output': output,
                'full_output': response
            }

        except Exception as e:
            logging.error(f"Error in BaseModel query: {str(e)}")
            return {
                'output': f"Error processing query: {str(e)}",
                'full_output': None
            }
        
    async def chat(self, messages):
        """
        Chat with the model using a list of messages.
        
        Args:
            messages (list): List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            dict: Response with output and full_output keys
        """
        try:
            # Build messages list starting with system prompt
            conversation_messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                }
            ]
            
            # Add provided messages
            conversation_messages.extend(messages)
            
            # Prepare kwargs for API call
            kwargs = {
                'model': self.deployment_name,
                'messages': conversation_messages
            }
            
            # Only add temperature and top_p for models that support them
            if self.model not in ['o4-mini']:
                kwargs['temperature'] = self.temperature
                kwargs['top_p'] = self.top_p
            
            # Make the API call
            response = await self.client.chat.completions.create(**kwargs)
            
            # Extract the response content
            output = response.choices[0].message.content
            
            # Return in consistent format
            return {
                "output": output,
                "full_output": response
            }
            
        except Exception as e:
            logging.error(f"Error in BaseModel chat: {str(e)}")
            return {
                "output": f"Error processing chat: {str(e)}",
                "full_output": None
            }
