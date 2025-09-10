import asyncio
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  
from nltk.tokenize import word_tokenize
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import logging
import pandas as pd
import requests
import re
import time
import timeit

logger = logging.getLogger(__name__)

class LLMRAGEvaluator:
    def __init__(self):
        self.client = AzureOpenAI(
            api_version="2025-01-01-preview",
            azure_deployment="gpt-4-omni",
            azure_endpoint="https://cog-gryrq-aik-swce-poc.openai.azure.com",
            azure_ad_token_provider=get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            )
        )
        logger.debug("Initialized LLMRAGEvaluator")

    def evaluate_sample(self, query, retrieved_docs, model_output, expected_answer):
        if not isinstance(retrieved_docs, list):
            retrieved_docs = [str(retrieved_docs)]
        else:
            retrieved_docs = [str(doc) for doc in retrieved_docs]

        prompt = f"""
        You are a fair, expert RAG evaluator. Given the following:

        Query: {query}
        Retrieved Docs: {retrieved_docs}
        Model Output: {model_output}
        Expected Answer: {expected_answer}

        Evaluate the following metrics:

        1. TaskSuccess: Does the model output successfully complete the task requested in the query? This is a strict binary measure - either the user gets what they need or they don't. Use ONLY these values:
        - 0 = Failed to complete the task OR provides incorrect/misleading information OR missing critical information needed
        - 10 = Successfully completes the task with correct information that fully addresses what the user requested
        
        For different task types, consider:
        - Math/calculations/logical reasoning: Must contain exact correct answer with precise values
        - Factual questions: Must contain accurate facts/numbers
        - Summarization: Must capture main ideas and key information (format/wording can vary from expected answer)
        - Explanations: Must provide correct and complete explanation
        
        2. Faithfulness: Is the model output consistent with the information in the retrieved documents? (0 = completely inconsistent, 10 = fully consistent)
        
        3. AnswerRelevancy: Does the model output directly and accurately address the query? (0 = irrelevant, 10 = fully relevant)
        
        4. ContextRecall: Do the retrieved documents cover all the information present in the expected answer? (0 = no coverage, 10 = full coverage)
        
        5. ContextPrecision: Do the retrieved documents contain the most relevant information needed to answer the query? (0 = irrelevant documents, 10 = highly relevant documents)

        Return ONLY a valid JSON object with these keys and values:
        - TaskSuccess: 0 or 10 only
        - TaskSuccessExplanation: brief explanation (1-2 sentences) of why you gave this TaskSuccess score
        - Faithfulness: 0-10
        - AnswerRelevancy: 0-10  
        - ContextRecall: 0-10
        - ContextPrecision: 0-10


        Do not include any explanation or formatting. Just return the JSON.
        """
        
        
        # Define the API call function for timing
        def api_call():
            return self.client.chat.completions.create(
                model="gpt-4-omni",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300
            )

        try:
            # Precise timing using timeit
            start_time = timeit.default_timer()
            result = api_call()
            end_time = timeit.default_timer()
            
            # Calculate precise API call time
            api_call_time = end_time - start_time

            content = result.choices[0].message.content.strip()
            start = content.find("{")
            if start == -1:
                raise ValueError(f"No JSON object found in LLMRAGEvaluator response: {content!r}")

            decoder = json.JSONDecoder()
            data, _ = decoder.raw_decode(content[start:])
            
            # Add timing and usage information
            data['EvaluationTime'] = api_call_time  # Precise local timing in seconds
            data['APITimestamp'] = result.created   # API server timestamp
            
            usage = result.usage
            total_tokens = usage.total_tokens if usage else 0
            data['TokenUsage'] = total_tokens
            
            # GPT-4o pricing (as of 2024): $5 per 1M input tokens, $15 per 1M output tokens
            # For simplicity, using blended rate of ~$10 per 1M tokens (adjust based on actual input/output split)
            if usage:
                input_tokens = usage.prompt_tokens or 0
                output_tokens = usage.completion_tokens or 0
                # More precise pricing: $5/1M input + $15/1M output
                cost_usd = (input_tokens * 5.0 / 1_000_000) + (output_tokens * 15.0 / 1_000_000)
                data['CostUSD'] = round(cost_usd, 6)
                data['InputTokens'] = input_tokens
                data['OutputTokens'] = output_tokens
            else:
                data['CostUSD'] = 0.0
                data['InputTokens'] = 0
                data['OutputTokens'] = 0
            
            return data

        except Exception as e:
            # Still measure time even on failure
            error_time = timeit.default_timer() - start_time if 'start_time' in locals() else 0.0
            
            logger.error(f"Evaluation error: {e}")
            return {
                "TaskSuccess": 0,
                "Faithfulness": 0,
                "AnswerRelevancy": 0,
                "ContextRecall": 0,
                "ContextPrecision": 0,
                "EvaluationTime": error_time,
                "APITimestamp": 0,
                "TokenUsage": 0,
                "InputTokens": 0,
                "OutputTokens": 0,
                "CostUSD": 0.0,
                "error": str(e)
            }
