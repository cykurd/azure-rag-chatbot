import copy
import json
import os
import requests
import logging
import uuid
import time
from dotenv import load_dotenv
from itsdangerous import URLSafeSerializer
import mimetypes 
import pyodbc
import tiktoken 
import asyncio
# CyrusGPT enhanced retrieval imports
from backend.CyrusGPT import CyrusGPT

from apscheduler.schedulers.asyncio import AsyncIOScheduler  
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.cron import CronTrigger
from quart import (
    Blueprint,
    Quart,
    jsonify,
    make_response,
    request,
    send_from_directory,
    render_template,
    session,
    send_file
)
import io
from openai import AsyncAzureOpenAI
from openai.types.beta.assistant_stream_event import (
    ThreadRunRequiresAction, ThreadMessageDelta, ThreadRunCompleted, ThreadRunStepDelta,
    ThreadRunFailed, ThreadRunCancelling, ThreadRunCancelled, ThreadRunExpired, ThreadRunStepFailed,
    ThreadRunStepCancelled)
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from azure.identity import DefaultAzureCredential as DefaultAzureCredentialSync
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes import SearchIndexerClient
from azure.storage.blob import BlobServiceClient
from azure.search.documents.indexes.models import SearchIndex, SearchIndexer, SearchIndexerDataSourceConnection, SearchIndexerSkillset
from azure.core.exceptions import ResourceNotFoundError
from backend.auth.auth_utils import get_authenticated_user_details
from backend.history.sqldbservice import SqlConversationClient
from sqlalchemy import create_engine  
from sqlalchemy import URL 
from urllib.parse import quote_plus 
from backend.utils import format_as_ndjson, format_stream_response, generateFilterString, generateFileFilterStringGeneric, fetchUserGroupNames, parse_multi_columns, format_non_streaming_response, send_log_event_to_eventhub
from backend.utils import EventHandler
from urllib.parse import quote 
from functools import lru_cache 
from sqlalchemy import create_engine, Column, ForeignKey, Index, String, DateTime, Text  
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER  
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, relationship  
from sqlalchemy.sql import func 
from datetime import datetime 
bp = Blueprint("routes", __name__, static_folder="static", template_folder="static")

# Using DefaultAzureCredential to authenticate with the system identity manager
credential = DefaultAzureCredentialSync()
#logging.debug(credential.get_token("https://storage.azure.com/.default"))

# Configuration details for Azure SQL database    
AZURE_SQL_SERVER = os.environ.get("AZURE_SQL_SERVER") 
AZURE_SQL_DATABASE = os.environ.get("AZURE_SQL_DATABASE")
AZURE_SQL_USER = os.environ.get("AZURE_SQL_USER")
AZURE_SQL_PASSWORD = os.environ.get("AZURE_SQL_PASSWORD")
AZURE_SQL_ENABLE_FEEDBACK = os.environ.get("AZURE_SQL_ENABLE_FEEDBACK", "true")

Base = declarative_base()  
  
# Replace with your actual Azure SQL Database connection string  
params = quote_plus(  
    (  
        f"Driver={{ODBC Driver 18 for SQL Server}};"  
        f"Server=tcp:{AZURE_SQL_SERVER},1433;"  
        f"Database={AZURE_SQL_DATABASE};"  
        f"Uid={AZURE_SQL_USER};Pwd={AZURE_SQL_PASSWORD};"  
        f"Encrypt=yes;"  
        f"TrustServerCertificate=no;"  
        f"Connection Timeout=30;" 
        f"charset=UTF-8;"  
    )  
)   
connection_url = f"mssql+pyodbc:///?odbc_connect={params}"

# tiktoken local imports
tiktoken_cache_dir = "./backend/tiktoken_cache/"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

# Azure pricing List
# Read pricing data from local JSON file  
with open("./backend/azure_price_list/priceList.json", 'r') as file:  
    pricing_data = json.load(file) 

# UI configuration (optional)
UI_TITLE = os.environ.get("UI_TITLE") or "TevaGPT"
UI_LOGO = os.environ.get("UI_LOGO")
UI_CHAT_LOGO = os.environ.get("UI_CHAT_LOGO")
UI_CHAT_TITLE = os.environ.get("UI_CHAT_TITLE") or "Start chatting"
UI_CHAT_DESCRIPTION = os.environ.get("UI_CHAT_DESCRIPTION") or "This chatbot is configured to answer your questions"
UI_FAVICON = os.environ.get("UI_FAVICON") or "/teva-favi.png"
UI_SHOW_SHARE_BUTTON = os.environ.get("UI_SHOW_SHARE_BUTTON", "true").lower() == "true"
SOME_KEY = os.environ.get("SOME_KEY", "haergaerwetjaertwec")
serializer = URLSafeSerializer(SOME_KEY)
async def start_scheduler(scheduler):  
    if not scheduler.running:  
        scheduler.start()  
  
def create_app():  
    app = Quart(__name__)  
    app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024  # 300 MB limit (adjust as needed)  
    
    # Configure the job store and the scheduler  
    jobstores = {  
        'default': SQLAlchemyJobStore(url='sqlite:///jobs.sqlite')  
    }  
    scheduler = AsyncIOScheduler(jobstores=jobstores)  
    app.extensions['apscheduler'] = scheduler  
  
    # Define the job trigger  
    job_trigger = CronTrigger(hour=0, minute=0)  
  
    # Empty the job store before starting the app  
    scheduler.remove_all_jobs()  
  
    # Add the job after clearing the existing ones  
    scheduler.add_job(func=execute_stored_procedure, trigger=job_trigger)  
  
    app.register_blueprint(bp)  
  
    @app.before_serving  
    async def before_serving():  
        # Start the scheduler before the server starts serving  
        await start_scheduler(scheduler)  
  
    return app
 
def execute_stored_procedure(): 
    start_datetime = datetime.now() 
    # Set up your database connection string  
    connection_string = (  
                    f"Driver={{ODBC Driver 17 for SQL Server}};"  
                    f"Server=tcp:{AZURE_SQL_SERVER},1433;"  
                    f"Database={AZURE_SQL_DATABASE};"  
                    f"Uid={AZURE_SQL_USER};"    
                    f"Pwd={AZURE_SQL_PASSWORD};"  
                    f"Encrypt=yes;"  
                    f"TrustServerCertificate=no;"  
                    f"Connection Timeout=30;"  
                    # f"Authentication=ActiveDirectoryIntegrated;"   
                ) 
    days_to_keep = os.environ.get("CHAT_HISTORY_EXPIRY_DAYS", 30)           
    # Connect to the database  
    with pyodbc.connect(connection_string) as conn:  
        cursor = conn.cursor()  
          
        # Call the stored procedure  
        try:  
            cursor.execute("EXEC DeleteOldConversationsAndMessages @DaysToKeep=?", days_to_keep)  
            cursor.commit()
            #log event to eventhub
            send_log_event_to_eventhub(credential, "INFO", f"Auto Deletion Cron Job successful", "execute_stored_procedure", None, start_datetime=start_datetime,status="Success",action_type="App Action",action_description="Executing the SP for Auto deletion")
        
        except Exception as e: 
            #log event to eventhub
            send_log_event_to_eventhub(credential, "ERROR", f"Exception in Auto Deletion Cron Job:{e}", "execute_stored_procedure", None, start_datetime=start_datetime,status="Failed",action_type="App Action",action_description="Executing the SP for Auto deletion")
            print(f"An error occurred: {e}")  
        finally:  
            cursor.close()  
   
@bp.route("/")
async def index():
    return await render_template("index.html", title=UI_TITLE, favicon=UI_FAVICON)

@bp.route("/teva-favi.png")
async def favicon():
    return await bp.send_static_file("teva-favi.png")

@bp.route("/assets/<path:path>")
async def assets(path):
    return await send_from_directory("static/assets", path)

load_dotenv()

# Debug settings
DEBUG = os.environ.get("DEBUG", "false")
if DEBUG.lower() == "true":
    logging.basicConfig(level=logging.DEBUG)

USER_AGENT = "GitHubSampleWebApp/AsyncAzureOpenAI/1.0.0"
INDEX_GROUP_CONFIG = os.environ.get("INDEX_GROUP_CONFIG", None)

CUSTOM_TUNING_CONFIG = json.loads(os.environ.get("CUSTOM_TUNING_CONFIG", "{}"))
SYSTEM_PROMPT_CONFIG = json.loads(os.environ.get("SYSTEM_PROMPT_CONFIG", "{}"))
USECASE_DESC = json.loads(os.environ.get("USECASE_DESC", '{"rnd":"Chat with R&D SOPs", "cdc": "Chat with CDC Data","gsc":"Chat with GSC SOPs", "us-contracts":"US Contracting & Formulary Compliance", "dset": "Chat with DSET Demo Data", "im-portfolio": "IM Portfolio SWOT Analysis", "im-sales":"IM Sales Calls Analysis", "gia":"Search & Query GIA Audit Content"}'))

# On Your Data Settings
DATASOURCE_TYPE = os.environ.get("DATASOURCE_TYPE")
SEARCH_TOP_K = os.environ.get("SEARCH_TOP_K", 100)
SEARCH_STRICTNESS = os.environ.get("SEARCH_STRICTNESS", 3)
SEARCH_ENABLE_IN_DOMAIN = os.environ.get("SEARCH_ENABLE_IN_DOMAIN", "true")

# ACS Integration Settings
AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE", None)
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX", None)
AZURE_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY", None)
AZURE_SEARCH_USE_SEMANTIC_SEARCH = os.environ.get("AZURE_SEARCH_USE_SEMANTIC_SEARCH", "false")
AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG = os.environ.get("AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG", "default")
AZURE_SEARCH_TOP_K = os.environ.get("AZURE_SEARCH_TOP_K", SEARCH_TOP_K)
AZURE_SEARCH_ENABLE_IN_DOMAIN = os.environ.get("AZURE_SEARCH_ENABLE_IN_DOMAIN", SEARCH_ENABLE_IN_DOMAIN)
AZURE_SEARCH_CONTENT_COLUMNS = os.environ.get("AZURE_SEARCH_CONTENT_COLUMNS")
AZURE_SEARCH_FILENAME_COLUMN = os.environ.get("AZURE_SEARCH_FILENAME_COLUMN", "chunk_id")
AZURE_SEARCH_TITLE_COLUMN = os.environ.get("AZURE_SEARCH_TITLE_COLUMN", "title")
AZURE_SEARCH_URL_COLUMN = os.environ.get("AZURE_SEARCH_URL_COLUMN", "filepath")
AZURE_SEARCH_VECTOR_COLUMNS = os.environ.get("AZURE_SEARCH_VECTOR_COLUMNS")
AZURE_SEARCH_QUERY_TYPE = os.environ.get("AZURE_SEARCH_QUERY_TYPE", "vectorSemanticHybrid")
AZURE_SEARCH_PERMITTED_GROUPS_COLUMN = os.environ.get("AZURE_SEARCH_PERMITTED_GROUPS_COLUMN")
AZURE_SEARCH_STRICTNESS = os.environ.get("AZURE_SEARCH_STRICTNESS", SEARCH_STRICTNESS)
AZURE_SEARCH_PAGE_CHUNK_SIZE = os.environ.get("AZURE_SEARCH_PAGE_CHUNK_SIZE", 7000)
AZURE_SEARCH_PAGE_OVERLAP_SIZE = os.environ.get("AZURE_SEARCH_PAGE_OVERLAP_SIZE", 200)

# AOAI Integration Settings
AZURE_OPENAI_RESOURCE = os.environ.get("AZURE_OPENAI_RESOURCE")
AZURE_OPENAI_MODEL = os.environ.get("AZURE_OPENAI_MODEL")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_TEMPERATURE = os.environ.get("AZURE_OPENAI_TEMPERATURE", 0)
AZURE_OPENAI_TOP_P = os.environ.get("AZURE_OPENAI_TOP_P", 1.0)
AZURE_OPENAI_MAX_TOKENS = os.environ.get("AZURE_OPENAI_MAX_TOKENS", 3000)
AZURE_OPENAI_SEED = os.environ.get("AZURE_OPENAI_SEED", None)
AZURE_OPENAI_STOP_SEQUENCE = os.environ.get("AZURE_OPENAI_STOP_SEQUENCE")
AZURE_OPENAI_SYSTEM_MESSAGE = os.environ.get("AZURE_OPENAI_SYSTEM_MESSAGE", "Please answer using only the information provided in this prompt, including previous chat history and retrieved document chunks. Do not include any other information from your own knowledge or any other sources. If the information is not available in these specified sources, respond with 'Information not available.")
AZURE_OPENAI_PREVIEW_API_VERSION = os.environ.get("AZURE_OPENAI_PREVIEW_API_VERSION", "2023-12-01-preview")
AZURE_OPENAI_STREAM = os.environ.get("AZURE_OPENAI_STREAM", "true")
AZURE_OPENAI_MODEL_NAME = os.environ.get("AZURE_OPENAI_MODEL_NAME", "gpt-35-turbo-16k") # Name of the model, e.g. 'gpt-35-turbo-16k' or 'gpt-4'
AZURE_OPENAI_EMBEDDING_ENDPOINT = os.environ.get("AZURE_OPENAI_EMBEDDING_ENDPOINT")
AZURE_OPENAI_EMBEDDING_KEY = os.environ.get("AZURE_OPENAI_EMBEDDING_KEY")
AZURE_OPENAI_EMBEDDING_NAME = os.environ.get("AZURE_OPENAI_EMBEDDING_NAME", "")

# CyrusGPT Enhanced Retrieval Config
CYRUSGPT_ENABLED = os.environ.get("CYRUSGPT_ENABLED", "false").lower() == "true"
CYRUSGPT_TEMPERATURE = float(os.environ.get("CYRUSGPT_TEMPERATURE", "0.0"))
CYRUSGPT_TOP_P = float(os.environ.get("CYRUSGPT_TOP_P", "0.9"))
CYRUSGPT_MAX_TOKENS = int(os.environ.get("CYRUSGPT_MAX_TOKENS", "4096"))
CYRUSGPT_SEED = int(os.environ.get("CYRUSGPT_SEED", "42"))
CYRUSGPT_TOP_K = int(os.environ.get("CYRUSGPT_TOP_K", "25"))
CYRUSGPT_DEPLOYMENT = os.environ.get("CYRUSGPT_DEPLOYMENT", "gpt-4-omni")
CYRUSGPT_INDEX_NAME = os.environ.get("CYRUSGPT_INDEX_NAME", "uscomm-ana-index")
CYRUSGPT_CONTAINER_NAME = os.environ.get("CYRUSGPT_CONTAINER_NAME", "uscomm-ana")
CYRUSGPT_SYSTEM_PROMPT_VARIANT = os.environ.get("CYRUSGPT_SYSTEM_PROMPT_VARIANT", "current")
CYRUSGPT_RETRIEVAL_METHOD = os.environ.get("CYRUSGPT_RETRIEVAL_METHOD", "enhanced")
CYRUSGPT_QUERY_EXPANSION = os.environ.get("CYRUSGPT_QUERY_EXPANSION", "true").lower() == "true"
CYRUSGPT_HYBRID_SEARCH = os.environ.get("CYRUSGPT_HYBRID_SEARCH", "true").lower() == "true"
CYRUSGPT_ADVANCED_RERANKING = os.environ.get("CYRUSGPT_ADVANCED_RERANKING", "true").lower() == "true"
CYRUSGPT_DIVERSITY_SELECTION = os.environ.get("CYRUSGPT_DIVERSITY_SELECTION", "true").lower() == "true"
# CYRUSGPT_ENABLED_USECASES='{"uscomm-ana": true, "rnd": false, "cdc": true, "gsc": false}'
CYRUSGPT_ENABLED_USECASES = json.loads(os.environ.get("CYRUSGPT_ENABLED_USECASES", "{}"))
# CYRUSGPT_USECASE_CONFIG='{
#   "uscomm-ana": {
#     "deployment": "gpt-4-omni",
#     "temperature": 0.0,
#     "retrieval_method": "enhanced",
#     "query_expansion": true,
#     "advanced_reranking": true,
#     "diversity_selection": true
#   },
#   "cdc": {
#     "deployment": "o4-mini", 
#     "temperature": 0.1,
#     "retrieval_method": "default",
#     "query_expansion": false,
#     "advanced_reranking": false,
#     "diversity_selection": false
#   }
# }'
CYRUSGPT_USECASE_CONFIG = json.loads(os.environ.get("CYRUSGPT_USECASE_CONFIG", "{}"))

# CosmosDB Mongo vcore vector db Settings
AZURE_COSMOSDB_MONGO_VCORE_CONNECTION_STRING = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_CONNECTION_STRING")  #This has to be secure string
AZURE_COSMOSDB_MONGO_VCORE_DATABASE = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_DATABASE")
AZURE_COSMOSDB_MONGO_VCORE_CONTAINER = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_CONTAINER")
AZURE_COSMOSDB_MONGO_VCORE_INDEX = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_INDEX")
AZURE_COSMOSDB_MONGO_VCORE_TOP_K = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_TOP_K", AZURE_SEARCH_TOP_K)
AZURE_COSMOSDB_MONGO_VCORE_STRICTNESS = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_STRICTNESS", AZURE_SEARCH_STRICTNESS)  
AZURE_COSMOSDB_MONGO_VCORE_ENABLE_IN_DOMAIN = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_ENABLE_IN_DOMAIN", AZURE_SEARCH_ENABLE_IN_DOMAIN)
AZURE_COSMOSDB_MONGO_VCORE_CONTENT_COLUMNS = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_CONTENT_COLUMNS", "")
AZURE_COSMOSDB_MONGO_VCORE_FILENAME_COLUMN = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_FILENAME_COLUMN")
AZURE_COSMOSDB_MONGO_VCORE_TITLE_COLUMN = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_TITLE_COLUMN")
AZURE_COSMOSDB_MONGO_VCORE_URL_COLUMN = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_URL_COLUMN")
AZURE_COSMOSDB_MONGO_VCORE_VECTOR_COLUMNS = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_VECTOR_COLUMNS")

SHOULD_STREAM = True if AZURE_OPENAI_STREAM.lower() == "true" else False

# Chat History CosmosDB Integration Settings
AZURE_COSMOSDB_DATABASE = os.environ.get("AZURE_COSMOSDB_DATABASE")
AZURE_COSMOSDB_ACCOUNT = os.environ.get("AZURE_COSMOSDB_ACCOUNT")
AZURE_COSMOSDB_CONVERSATIONS_CONTAINER = os.environ.get("AZURE_COSMOSDB_CONVERSATIONS_CONTAINER")
AZURE_COSMOSDB_ACCOUNT_KEY = os.environ.get("AZURE_COSMOSDB_ACCOUNT_KEY")
AZURE_COSMOSDB_ENABLE_FEEDBACK = os.environ.get("AZURE_COSMOSDB_ENABLE_FEEDBACK", "false").lower() == "true"

# Elasticsearch Integration Settings
ELASTICSEARCH_ENDPOINT = os.environ.get("ELASTICSEARCH_ENDPOINT")
ELASTICSEARCH_ENCODED_API_KEY = os.environ.get("ELASTICSEARCH_ENCODED_API_KEY")
ELASTICSEARCH_INDEX = os.environ.get("ELASTICSEARCH_INDEX")
ELASTICSEARCH_QUERY_TYPE = os.environ.get("ELASTICSEARCH_QUERY_TYPE", "simple")
ELASTICSEARCH_TOP_K = os.environ.get("ELASTICSEARCH_TOP_K", SEARCH_TOP_K)
ELASTICSEARCH_ENABLE_IN_DOMAIN = os.environ.get("ELASTICSEARCH_ENABLE_IN_DOMAIN", SEARCH_ENABLE_IN_DOMAIN)
ELASTICSEARCH_CONTENT_COLUMNS = os.environ.get("ELASTICSEARCH_CONTENT_COLUMNS")
ELASTICSEARCH_FILENAME_COLUMN = os.environ.get("ELASTICSEARCH_FILENAME_COLUMN")
ELASTICSEARCH_TITLE_COLUMN = os.environ.get("ELASTICSEARCH_TITLE_COLUMN")
ELASTICSEARCH_URL_COLUMN = os.environ.get("ELASTICSEARCH_URL_COLUMN")
ELASTICSEARCH_VECTOR_COLUMNS = os.environ.get("ELASTICSEARCH_VECTOR_COLUMNS")
ELASTICSEARCH_STRICTNESS = os.environ.get("ELASTICSEARCH_STRICTNESS", SEARCH_STRICTNESS)
ELASTICSEARCH_EMBEDDING_MODEL_ID = os.environ.get("ELASTICSEARCH_EMBEDDING_MODEL_ID")

# Pinecone Integration Settings
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
PINECONE_TOP_K = os.environ.get("PINECONE_TOP_K", SEARCH_TOP_K)
PINECONE_STRICTNESS = os.environ.get("PINECONE_STRICTNESS", SEARCH_STRICTNESS)  
PINECONE_ENABLE_IN_DOMAIN = os.environ.get("PINECONE_ENABLE_IN_DOMAIN", SEARCH_ENABLE_IN_DOMAIN)
PINECONE_CONTENT_COLUMNS = os.environ.get("PINECONE_CONTENT_COLUMNS", "")
PINECONE_FILENAME_COLUMN = os.environ.get("PINECONE_FILENAME_COLUMN")
PINECONE_TITLE_COLUMN = os.environ.get("PINECONE_TITLE_COLUMN")
PINECONE_URL_COLUMN = os.environ.get("PINECONE_URL_COLUMN")
PINECONE_VECTOR_COLUMNS = os.environ.get("PINECONE_VECTOR_COLUMNS")

# Azure AI MLIndex Integration Settings - for use with MLIndex data assets created in Azure AI Studio
AZURE_MLINDEX_NAME = os.environ.get("AZURE_MLINDEX_NAME")
AZURE_MLINDEX_VERSION = os.environ.get("AZURE_MLINDEX_VERSION")
AZURE_ML_PROJECT_RESOURCE_ID = os.environ.get("AZURE_ML_PROJECT_RESOURCE_ID") # /subscriptions/{sub ID}/resourceGroups/{rg name}/providers/Microsoft.MachineLearningServices/workspaces/{AML project name}
AZURE_MLINDEX_TOP_K = os.environ.get("AZURE_MLINDEX_TOP_K", SEARCH_TOP_K)
AZURE_MLINDEX_STRICTNESS = os.environ.get("AZURE_MLINDEX_STRICTNESS", SEARCH_STRICTNESS)  
AZURE_MLINDEX_ENABLE_IN_DOMAIN = os.environ.get("AZURE_MLINDEX_ENABLE_IN_DOMAIN", SEARCH_ENABLE_IN_DOMAIN)
AZURE_MLINDEX_CONTENT_COLUMNS = os.environ.get("AZURE_MLINDEX_CONTENT_COLUMNS", "")
AZURE_MLINDEX_FILENAME_COLUMN = os.environ.get("AZURE_MLINDEX_FILENAME_COLUMN")
AZURE_MLINDEX_TITLE_COLUMN = os.environ.get("AZURE_MLINDEX_TITLE_COLUMN")
AZURE_MLINDEX_URL_COLUMN = os.environ.get("AZURE_MLINDEX_URL_COLUMN")
AZURE_MLINDEX_VECTOR_COLUMNS = os.environ.get("AZURE_MLINDEX_VECTOR_COLUMNS")
AZURE_MLINDEX_QUERY_TYPE = os.environ.get("AZURE_MLINDEX_QUERY_TYPE")


# Frontend Settings via Environment Variables
AUTH_ENABLED = os.environ.get("AUTH_ENABLED", "true").lower() == "true"
# CHAT_HISTORY_ENABLED = False 
frontend_settings = { 
    "auth_enabled": AUTH_ENABLED, 
    "feedback_enabled": AZURE_SQL_ENABLE_FEEDBACK and AZURE_SQL_DATABASE and AZURE_SQL_SERVER,
    "ui": {
        "title": UI_TITLE,
        "logo": UI_LOGO,
        "chat_logo": UI_CHAT_LOGO or UI_LOGO,
        "chat_title": UI_CHAT_TITLE,
        "chat_description": UI_CHAT_DESCRIPTION,
        "show_share_button": UI_SHOW_SHARE_BUTTON
    }
}

#Load System prompts
with open('systemPrompt.json', 'r') as config_file:  
    system_messages = json.load(config_file) 

# Constants for dummy file  
DUMMY_FILE_NAME = '.keep'  
DUMMY_FILE_CONTENT = b'' 

def should_use_data():
    global DATASOURCE_TYPE
    if AZURE_SEARCH_SERVICE and AZURE_SEARCH_INDEX:
        DATASOURCE_TYPE = "AzureCognitiveSearch"
        logging.debug("Using Azure Cognitive Search")
        return True
    
    if AZURE_COSMOSDB_MONGO_VCORE_DATABASE and AZURE_COSMOSDB_MONGO_VCORE_CONTAINER and AZURE_COSMOSDB_MONGO_VCORE_INDEX and AZURE_COSMOSDB_MONGO_VCORE_CONNECTION_STRING:
        DATASOURCE_TYPE = "AzureCosmosDB"
        logging.debug("Using Azure CosmosDB Mongo vcore")
        return True
    
    if ELASTICSEARCH_ENDPOINT and ELASTICSEARCH_ENCODED_API_KEY and ELASTICSEARCH_INDEX:
        DATASOURCE_TYPE = "Elasticsearch"
        logging.debug("Using Elasticsearch")
        return True
    
    if PINECONE_ENVIRONMENT and PINECONE_API_KEY and PINECONE_INDEX_NAME:
        DATASOURCE_TYPE = "Pinecone"
        logging.debug("Using Pinecone")
        return True
    
    if AZURE_MLINDEX_NAME and AZURE_MLINDEX_VERSION and AZURE_ML_PROJECT_RESOURCE_ID:
        DATASOURCE_TYPE = "AzureMLIndex"
        logging.debug("Using Azure ML Index")
        return True

    return False

SHOULD_USE_DATA = should_use_data()

# Initialize Azure OpenAI Client
def init_openai_client(index_to_use, param_dict, use_data=SHOULD_USE_DATA):
    azure_openai_client = None
    try:
        # Endpoint
        if not AZURE_OPENAI_ENDPOINT and not AZURE_OPENAI_RESOURCE:
            raise Exception("AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_RESOURCE is required")
        
        endpoint = AZURE_OPENAI_ENDPOINT if AZURE_OPENAI_ENDPOINT else f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com/"
        
        # Authentication
        aoai_api_key = AZURE_OPENAI_KEY
        ad_token_provider = None
        if not aoai_api_key:
            logging.debug("No AZURE_OPENAI_KEY found, using Azure AD auth")
            ad_token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
        
        # Deployment
        deployment = AZURE_OPENAI_MODEL

        if param_dict and param_dict.get("AZURE_OPENAI_MODEL"):
            deployment = param_dict.get("AZURE_OPENAI_MODEL")

        if not deployment:
            raise Exception("AZURE_OPENAI_MODEL is required")

        # Default Headers
        default_headers = {
            'x-ms-useragent': USER_AGENT
        }

        if use_data and str(index_to_use) != 'None':
            base_url = f"{str(endpoint).rstrip('/')}/openai/deployments/{deployment}/extensions"
            azure_openai_client = AsyncAzureOpenAI(
                base_url=str(base_url),
                api_version=AZURE_OPENAI_PREVIEW_API_VERSION,
                api_key=aoai_api_key,
                azure_ad_token_provider=ad_token_provider,
                default_headers=default_headers,
            )
        else:
            azure_openai_client = AsyncAzureOpenAI(
                api_version=AZURE_OPENAI_PREVIEW_API_VERSION,
                api_key=aoai_api_key,
                azure_ad_token_provider=ad_token_provider,
                default_headers=default_headers,
                azure_endpoint=endpoint
            )
        return azure_openai_client
    except Exception as e:
        logging.exception("Exception in Azure OpenAI initialization", e)
        # #log event to eventhub
        # send_log_event_to_eventhub(credential, "ERROR", f"Exception in Azure OpenAI initialization:{e}, 500", "init_openai_client", index_to_use)
        azure_openai_client = None
        raise e


def init_sql_client():  
    sql_connection = None 
    
    if AZURE_SQL_SERVER and AZURE_SQL_DATABASE:  
        try:      
            #logging.debug("I am in sql")
            # Create tables if they don't exist  
            sql_connection = SqlConversationClient( connection_url = connection_url)

        except Exception as e:  
            logging.exception("Exception in Azure SQL Server initialization", e)
            #log event to eventhub
            #send_log_event_to_eventhub(credential, "ERROR", f"Exception in Azure SQL Server initialization:{e}", "init_sql_client", None)
            sql_connection = None  
            raise e  
    else:  
        logging.debug("Azure SQL Server not configured")  
  
    return sql_connection  


def get_configured_data_source(index_to_use, param_dict, filter_expression):
    data_source = {}
    query_type = "simple"
    azure_index = None #AZURE_SEARCH_INDEX
    custom_top_k = None
    custom_strictness = None
    custom_config = CUSTOM_TUNING_CONFIG.get(index_to_use[:-6], None)
    if custom_config:
        custom_top_k = custom_config.get("top_k", None)
        custom_strictness = custom_config.get("strictness", None)
    if DATASOURCE_TYPE == "AzureCognitiveSearch":

        if index_to_use is not None:
            azure_index = index_to_use

        # Set query type
        if AZURE_SEARCH_QUERY_TYPE:
            query_type = AZURE_SEARCH_QUERY_TYPE
        elif AZURE_SEARCH_USE_SEMANTIC_SEARCH.lower() == "true" and AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG:
            query_type = "semantic"

        # Set filter
        filter = None
        
        userToken = None
        if AZURE_SEARCH_PERMITTED_GROUPS_COLUMN:
            userToken = request.headers.get('X-Ms-Token-Aad-Id-Token', 'eyJ0eXAiOiJKV1QiLCJub25jZSI6IjFvUVlyb2pIbm96OGpEX0Y1ZDMzbWZnbjJ6SVNFWi1JY0hlQmJFbmRtQkUiLCJhbGciOiJSUzI1NiIsIng1dCI6IkpZaEFjVFBNWl9MWDZEQmxPV1E3SG4wTmVYRSIsImtpZCI6IkpZaEFjVFBNWl9MWDZEQmxPV1E3SG4wTmVYRSJ9.eyJhdWQiOiJodHRwczovL2dyYXBoLm1pY3Jvc29mdC5jb20iLCJpc3MiOiJodHRwczovL3N0cy53aW5kb3dzLm5ldC8zZjk5MWE3Yi1lYTkzLTQxNjktYjI4Yy1jMzZmZjNlNWIwZDEvIiwiaWF0IjoxNzUzODgzMTQzLCJuYmYiOjE3NTM4ODMxNDMsImV4cCI6MTc1Mzg4NzY5NCwiYWNjdCI6MCwiYWNyIjoiMSIsImFjcnMiOlsicDEiLCJ1cm46dXNlcjpyZWdpc3RlcnNlY3VyaXR5aW5mbyJdLCJhaW8iOiJBWFFBaS84WkFBQUFWLzNFVVczUFV4aiszUFpoaEpHOWNlV3pKWC94OW9IL2cvcFBLVGJzTHRsR1JWN3dHVmtINU03MVBtZnY5MTcrUksvaGtDVlgxajhJZmZWSUN5MVR5SmZ5ZHRkRTkrWUJ3V3ZLUnlyRmJ3TlIvcWFBeXNXcHVZa05YRW94Q3o1YnVoaURjRGRIeDdHNy9nR1lnb1AxdWc9PSIsImFtciI6WyJwd2QiLCJyc2EiLCJtZmEiXSwiYXBwX2Rpc3BsYXluYW1lIjoiTWljcm9zb2Z0IEF6dXJlIENMSSIsImFwcGlkIjoiMDRiMDc3OTUtOGRkYi00NjFhLWJiZWUtMDJmOWUxYmY3YjQ2IiwiYXBwaWRhY3IiOiIwIiwiY2Fwb2xpZHNfbGF0ZWJpbmQiOlsiYzYyZWY3MDYtYmZkZi00OTVkLThhYjEtZGU0ZDljNmI4ZTY1Il0sImRldmljZWlkIjoiMzc0YjVlNTItMGI2ZS00ZjE5LTllMTgtZTQ1ZDAyZGNmNjVlIiwiZmFtaWx5X25hbWUiOiJLdXJkIiwiZ2l2ZW5fbmFtZSI6IkN5cnVzIiwiaWR0eXAiOiJ1c2VyIiwiaW5fY29ycCI6InRydWUiLCJpcGFkZHIiOiIxNjIuMjUxLjEwNC4xMCIsIm5hbWUiOiJDeXJ1cyBLdXJkIiwib2lkIjoiMWVhNjA2YjEtY2IyMy00OTcxLWJlMzAtNGJiMWVkNTk0YjM5Iiwib25wcmVtX3NpZCI6IlMtMS01LTIxLTI5ODgyNjg5OTMtMjQ2NDM3NTA5Ni0zNzI1MzI2NDYtNTY1MTg0IiwicGxhdGYiOiIzIiwicHVpZCI6IjEwMDMyMDA0N0Y3NDJFODkiLCJyaCI6IjEuQVlJQWV4cVpQNVBxYVVHeWpNTnY4LVd3MFFNQUFBQUFBQUFBd0FBQUFBQUFBQUNxQUJLQ0FBLiIsInNjcCI6IkFwcGxpY2F0aW9uLlJlYWRXcml0ZS5BbGwgQXBwUm9sZUFzc2lnbm1lbnQuUmVhZFdyaXRlLkFsbCBBdWRpdExvZy5SZWFkLkFsbCBEZWxlZ2F0ZWRQZXJtaXNzaW9uR3JhbnQuUmVhZFdyaXRlLkFsbCBEaXJlY3RvcnkuQWNjZXNzQXNVc2VyLkFsbCBlbWFpbCBHcm91cC5SZWFkV3JpdGUuQWxsIG9wZW5pZCBwcm9maWxlIFVzZXIuUmVhZC5BbGwgVXNlci5SZWFkV3JpdGUuQWxsIiwic2lkIjoiMDA2YjRjZDktOWYxYy04YmMyLTc0MWMtZDRhYWQ1ZWU0NzM5Iiwic2lnbmluX3N0YXRlIjpbImR2Y19tbmdkIiwiZHZjX2RtamQiLCJpbmtub3dubnR3ayIsImttc2kiXSwic3ViIjoiSjVKTmxMN0tZUDFtdld5ckFUd281aXpiTURGaFNKRVpzUGwybjV1bC1PdyIsInRlbmFudF9yZWdpb25fc2NvcGUiOiJFVSIsInRpZCI6IjNmOTkxYTdiLWVhOTMtNDE2OS1iMjhjLWMzNmZmM2U1YjBkMSIsInVuaXF1ZV9uYW1lIjoiQ3lydXMuS3VyZEB0ZXZhcGhhcm0uY29tIiwidXBuIjoiQ3lydXMuS3VyZEB0ZXZhcGhhcm0uY29tIiwidXRpIjoibTdjUTQ5Y3lnVVNTTFl3bUlwVm1BQSIsInZlciI6IjEuMCIsIndpZHMiOlsiYjc5ZmJmNGQtM2VmOS00Njg5LTgxNDMtNzZiMTk0ZTg1NTA5Il0sInhtc19jYyI6WyJDUDEiXSwieG1zX2Z0ZCI6ImFYNVgxTmU1UElYNjNRWVlSN25XSHdOaDFfc2tGc1FObkdnVUxYd0Z5WWtCWm5KaGJtTmxZeTFrYzIxeiIsInhtc19pZHJlbCI6IjEyIDEiLCJ4bXNfc3NtIjoiMSIsInhtc19zdCI6eyJzdWIiOiJvQ204VDBOT0I4TGJwMm80bEZfcHJnRXJvaVI4RDNCM2ZaUEVNbXlkNy1jIn0sInhtc190Y2R0IjoxMzk2ODkzNDYwfQ.oPVzUxXSo7Dk2Rld570JC7gGVjvPq9C0pf_5klaQprDoTDfcqFkwAFPuPBhduCj2TmkUpXbTX5rY_u1Bf-FIiPkHdz6WdTpWGCkFBbAYyVxwKystKDGc_RLyPSzctoz02yzJTwhNZA-8Y3_T3a4rh5EYOGoto6QOMp_magh-i0ijtuL2nLQ3nNP62_mPHWNGIg6vl9xtNqfCOnS-PxIybt3oXSewFHyjhWfxt26TgmNwonjfSTb20GNoBeOEUt-fentWpN4vkOdguTlTwrKu85R1Ea2PHzLjFojWnnrmLFq-cp1HSDZD83os0lz-kxo15ym3ebQpi8IveeHlOXAbow')
            logging.debug(f"USER TOKEN is {'present' if userToken else 'not present'}")

            filter = generateFilterString(userToken)
            logging.debug(f"FILTER: {filter}")

        # Set authentication
        authentication = {}
        if AZURE_SEARCH_KEY:
            authentication = {
                "type": "APIKey",
                "key": AZURE_SEARCH_KEY,
                "apiKey": AZURE_SEARCH_KEY
            }
        else:
            # If key is not provided, assume AOAI resource identity has been granted access to the search service
            authentication = {
                "type": "SystemAssignedManagedIdentity"
            }

        data_source = {
                "type": "AzureCognitiveSearch",
                "parameters": {
                    "endpoint": f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
                    "authentication": authentication,
                    "indexName": azure_index,
                    "fieldsMapping": {
                        "contentFields": parse_multi_columns(AZURE_SEARCH_CONTENT_COLUMNS) if AZURE_SEARCH_CONTENT_COLUMNS else ["content"],
                        "titleField": AZURE_SEARCH_TITLE_COLUMN if AZURE_SEARCH_TITLE_COLUMN else None,
                        "urlField": AZURE_SEARCH_URL_COLUMN if AZURE_SEARCH_URL_COLUMN else None,
                        "filepathField": AZURE_SEARCH_FILENAME_COLUMN if AZURE_SEARCH_FILENAME_COLUMN else None,
                        "vectorFields": parse_multi_columns(AZURE_SEARCH_VECTOR_COLUMNS) if AZURE_SEARCH_VECTOR_COLUMNS else []
                    },
                    "inScope": True if AZURE_SEARCH_ENABLE_IN_DOMAIN.lower() == "true" else False,
                    "topNDocuments": custom_top_k if custom_top_k else (int(AZURE_SEARCH_TOP_K) if AZURE_SEARCH_TOP_K else int(SEARCH_TOP_K)),
                    "queryType": query_type,
                    "semanticConfiguration": AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG if AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG else "",
                    "roleInformation": AZURE_OPENAI_SYSTEM_MESSAGE,
                    "filter": filter_expression,
                    "strictness": custom_strictness if custom_strictness else int(AZURE_SEARCH_STRICTNESS) if AZURE_SEARCH_STRICTNESS else int(SEARCH_STRICTNESS)
                }
            }

        if param_dict.get("AZURE_SEARCH_TOP_K"):
            data_source["parameters"]["topNDocuments"] = float(param_dict.get("AZURE_SEARCH_TOP_K"))
        if param_dict.get("AZURE_SEARCH_STRICTNESS"):
            data_source["parameters"]["strictness"] = float(param_dict.get("AZURE_SEARCH_STRICTNESS"))
        if param_dict.get("AZURE_SEARCH_QUERY_TYPE"):
            data_source["parameters"]["queryType"] = param_dict.get("AZURE_SEARCH_QUERY_TYPE")
            query_type = param_dict.get("AZURE_SEARCH_QUERY_TYPE")
        if param_dict.get("AZURE_OPENAI_SYSTEM_MESSAGE"):
            data_source["parameters"]["roleInformation"] = param_dict.get("AZURE_OPENAI_SYSTEM_MESSAGE")

    elif DATASOURCE_TYPE == "AzureCosmosDB":
        query_type = "vector"

        data_source = {
                "type": "AzureCosmosDB",
                "parameters": {
                    "authentication": {
                        "type": "ConnectionString",
                        "connectionString": AZURE_COSMOSDB_MONGO_VCORE_CONNECTION_STRING
                    },
                    "indexName": AZURE_COSMOSDB_MONGO_VCORE_INDEX,
                    "databaseName": AZURE_COSMOSDB_MONGO_VCORE_DATABASE,
                    "containerName": AZURE_COSMOSDB_MONGO_VCORE_CONTAINER,                    
                    "fieldsMapping": {
                        "contentFields": parse_multi_columns(AZURE_COSMOSDB_MONGO_VCORE_CONTENT_COLUMNS) if AZURE_COSMOSDB_MONGO_VCORE_CONTENT_COLUMNS else [],
                        "titleField": AZURE_COSMOSDB_MONGO_VCORE_TITLE_COLUMN if AZURE_COSMOSDB_MONGO_VCORE_TITLE_COLUMN else None,
                        "urlField": AZURE_COSMOSDB_MONGO_VCORE_URL_COLUMN if AZURE_COSMOSDB_MONGO_VCORE_URL_COLUMN else None,
                        "filepathField": AZURE_COSMOSDB_MONGO_VCORE_FILENAME_COLUMN if AZURE_COSMOSDB_MONGO_VCORE_FILENAME_COLUMN else None,
                        "vectorFields": parse_multi_columns(AZURE_COSMOSDB_MONGO_VCORE_VECTOR_COLUMNS) if AZURE_COSMOSDB_MONGO_VCORE_VECTOR_COLUMNS else []
                    },
                    "inScope": True if AZURE_COSMOSDB_MONGO_VCORE_ENABLE_IN_DOMAIN.lower() == "true" else False,
                    "topNDocuments": int(AZURE_COSMOSDB_MONGO_VCORE_TOP_K) if AZURE_COSMOSDB_MONGO_VCORE_TOP_K else int(SEARCH_TOP_K),
                    "strictness": int(AZURE_COSMOSDB_MONGO_VCORE_STRICTNESS) if AZURE_COSMOSDB_MONGO_VCORE_STRICTNESS else int(SEARCH_STRICTNESS),
                    "queryType": query_type,
                    "roleInformation": AZURE_OPENAI_SYSTEM_MESSAGE
                }
            }
    elif DATASOURCE_TYPE == "Elasticsearch":
        if ELASTICSEARCH_QUERY_TYPE:
            query_type = ELASTICSEARCH_QUERY_TYPE

        data_source = {
            "type": "Elasticsearch",
            "parameters": {
                "endpoint": ELASTICSEARCH_ENDPOINT,
                "authentication": {
                    "type": "EncodedAPIKey",
                    "encodedApiKey": ELASTICSEARCH_ENCODED_API_KEY
                },
                "indexName": ELASTICSEARCH_INDEX,
                "fieldsMapping": {
                    "contentFields": parse_multi_columns(ELASTICSEARCH_CONTENT_COLUMNS) if ELASTICSEARCH_CONTENT_COLUMNS else [],
                    "titleField": ELASTICSEARCH_TITLE_COLUMN if ELASTICSEARCH_TITLE_COLUMN else None,
                    "urlField": ELASTICSEARCH_URL_COLUMN if ELASTICSEARCH_URL_COLUMN else None,
                    "filepathField": ELASTICSEARCH_FILENAME_COLUMN if ELASTICSEARCH_FILENAME_COLUMN else None,
                    "vectorFields": parse_multi_columns(ELASTICSEARCH_VECTOR_COLUMNS) if ELASTICSEARCH_VECTOR_COLUMNS else []
                },
                "inScope": True if ELASTICSEARCH_ENABLE_IN_DOMAIN.lower() == "true" else False,
                "topNDocuments": int(ELASTICSEARCH_TOP_K) if ELASTICSEARCH_TOP_K else int(SEARCH_TOP_K),
                "queryType": query_type,
                "roleInformation": AZURE_OPENAI_SYSTEM_MESSAGE,
                "strictness": int(ELASTICSEARCH_STRICTNESS) if ELASTICSEARCH_STRICTNESS else int(SEARCH_STRICTNESS)
            }
        }
    elif DATASOURCE_TYPE == "AzureMLIndex":
        if AZURE_MLINDEX_QUERY_TYPE:
            query_type = AZURE_MLINDEX_QUERY_TYPE

        data_source = {
            "type": "AzureMLIndex",
            "parameters": {
                "name": AZURE_MLINDEX_NAME,
                "version": AZURE_MLINDEX_VERSION,
                "projectResourceId": AZURE_ML_PROJECT_RESOURCE_ID,
                "fieldsMapping": {
                    "contentFields": parse_multi_columns(AZURE_MLINDEX_CONTENT_COLUMNS) if AZURE_MLINDEX_CONTENT_COLUMNS else [],
                    "titleField": AZURE_MLINDEX_TITLE_COLUMN if AZURE_MLINDEX_TITLE_COLUMN else None,
                    "urlField": AZURE_MLINDEX_URL_COLUMN if AZURE_MLINDEX_URL_COLUMN else None,
                    "filepathField": AZURE_MLINDEX_FILENAME_COLUMN if AZURE_MLINDEX_FILENAME_COLUMN else None,
                    "vectorFields": parse_multi_columns(AZURE_MLINDEX_VECTOR_COLUMNS) if AZURE_MLINDEX_VECTOR_COLUMNS else []
                },
                "inScope": True if AZURE_MLINDEX_ENABLE_IN_DOMAIN.lower() == "true" else False,
                "topNDocuments": int(AZURE_MLINDEX_TOP_K) if AZURE_MLINDEX_TOP_K else int(SEARCH_TOP_K),
                "queryType": query_type,
                "roleInformation": AZURE_OPENAI_SYSTEM_MESSAGE,
                "strictness": int(AZURE_MLINDEX_STRICTNESS) if AZURE_MLINDEX_STRICTNESS else int(SEARCH_STRICTNESS)
            }
        }
    elif DATASOURCE_TYPE == "Pinecone":
        query_type = "vector"

        data_source = {
            "type": "Pinecone",
            "parameters": {
                "environment": PINECONE_ENVIRONMENT,
                "authentication": {
                    "type": "APIKey",
                    "key": PINECONE_API_KEY
                },
                "indexName": PINECONE_INDEX_NAME,
                "fieldsMapping": {
                    "contentFields": parse_multi_columns(PINECONE_CONTENT_COLUMNS) if PINECONE_CONTENT_COLUMNS else [],
                    "titleField": PINECONE_TITLE_COLUMN if PINECONE_TITLE_COLUMN else None,
                    "urlField": PINECONE_URL_COLUMN if PINECONE_URL_COLUMN else None,
                    "filepathField": PINECONE_FILENAME_COLUMN if PINECONE_FILENAME_COLUMN else None,
                    "vectorFields": parse_multi_columns(PINECONE_VECTOR_COLUMNS) if PINECONE_VECTOR_COLUMNS else []
                },
                "inScope": True if PINECONE_ENABLE_IN_DOMAIN.lower() == "true" else False,
                "topNDocuments": int(PINECONE_TOP_K) if PINECONE_TOP_K else int(SEARCH_TOP_K),
                "strictness": int(PINECONE_STRICTNESS) if PINECONE_STRICTNESS else int(SEARCH_STRICTNESS),
                "queryType": query_type,
                "roleInformation": AZURE_OPENAI_SYSTEM_MESSAGE,
            }
        }
    else:
        raise Exception(f"DATASOURCE_TYPE is not configured or unknown: {DATASOURCE_TYPE}")

    if "vector" in query_type.lower() and DATASOURCE_TYPE != "AzureMLIndex":
        embeddingDependency = {}
        if AZURE_OPENAI_EMBEDDING_NAME:
            embeddingDependency = {
                "type": "DeploymentName",
                "deploymentName": AZURE_OPENAI_EMBEDDING_NAME
            }
        elif AZURE_OPENAI_EMBEDDING_ENDPOINT and AZURE_OPENAI_EMBEDDING_KEY:
            embeddingDependency = {
                "type": "Endpoint",
                "endpoint": AZURE_OPENAI_EMBEDDING_ENDPOINT,
                "authentication": {
                    "type": "APIKey",
                    "key": AZURE_OPENAI_EMBEDDING_KEY
                }
            }
        elif DATASOURCE_TYPE == "Elasticsearch" and ELASTICSEARCH_EMBEDDING_MODEL_ID:
            embeddingDependency = {
                "type": "ModelId",
                "modelId": ELASTICSEARCH_EMBEDDING_MODEL_ID
            }
        else:
            raise Exception(f"Vector query type ({query_type}) is selected for data source type {DATASOURCE_TYPE} but no embedding dependency is configured")
        data_source["parameters"]["embeddingDependency"] = embeddingDependency
    
    return data_source

def prepare_model_args(param_dict, request_body, index_to_use, current_user, container_obj, start_datetime):
    request_messages = request_body.get("messages", [])
    filter_files = request_body.get("file_filter", [])
    all_files = request_body.get("all_files", [])
    length_of_topics = request_body.get("length_of_topics", 0)
    
    file_filter = generateFileFilterStringGeneric(filter_files, int(length_of_topics), all_files)
    messages = []
    
    if str(index_to_use) == 'None':
        messages = [
            {
                "role": "system",
                "content": AZURE_OPENAI_SYSTEM_MESSAGE
            }
        ]

    for message in request_messages:
        if message:
            messages.append({
                "role": message["role"] ,
                "content": message["content"]
            })

    model_args = {
        "messages": messages,
        "temperature": float(AZURE_OPENAI_TEMPERATURE),
        "max_tokens": int(AZURE_OPENAI_MAX_TOKENS),
        "top_p": float(AZURE_OPENAI_TOP_P),
        "stop": parse_multi_columns(AZURE_OPENAI_STOP_SEQUENCE) if AZURE_OPENAI_STOP_SEQUENCE else None,
        "stream": SHOULD_STREAM,
        "model": AZURE_OPENAI_MODEL_NAME,
    }

    if SHOULD_USE_DATA and str(index_to_use) != 'None' :

        custom_temperature = AZURE_OPENAI_TEMPERATURE
        custom_config = CUSTOM_TUNING_CONFIG.get(index_to_use, None)
        if custom_config:
            custom_temperature = custom_config.get("temperature", AZURE_OPENAI_TEMPERATURE)
            custom_max_tokens = custom_config.get("max_tokens", int(AZURE_OPENAI_MAX_TOKENS))
            custom_seed = custom_config.get("seed", None)

            model_args["temperature"] = float(custom_temperature)
            model_args["max_tokens"] = int(custom_max_tokens)
            if custom_seed:
                model_args["seed"] = int(custom_seed)

        data_sources = get_configured_data_source(index_to_use, param_dict, file_filter)

        

        if len(param_dict) > 0:
            if param_dict.get("AZURE_OPENAI_TEMPERATURE"):
                model_args["temperature"] = float(param_dict.get("AZURE_OPENAI_TEMPERATURE"))
            if param_dict.get("AZURE_OPENAI_MAX_TOKENS"):
                model_args["max_tokens"] = int(param_dict.get("AZURE_OPENAI_MAX_TOKENS"))
            if param_dict.get("AZURE_OPENAI_TOP_P"):
                model_args["top_p"] = float(param_dict.get("AZURE_OPENAI_TOP_P"))
            if param_dict.get("custom_seed"):
                model_args["seed"] = float(param_dict.get("custom_seed"))
            if param_dict.get("AZURE_OPENAI_TOP_P"):
                model_args["top_p"] = float(param_dict.get("AZURE_OPENAI_TOP_P"))
            
        model_args["extra_body"] = {
            "dataSources": [data_sources]
        }
    model_args_clean = copy.deepcopy(model_args)
    if model_args_clean.get("extra_body"):
        secret_params = ["key", "connectionString", "embeddingKey", "encodedApiKey", "apiKey"]
        for secret_param in secret_params:
            if model_args_clean["extra_body"]["dataSources"][0]["parameters"].get(secret_param):
                model_args_clean["extra_body"]["dataSources"][0]["parameters"][secret_param] = "*****"
        authentication = model_args_clean["extra_body"]["dataSources"][0]["parameters"].get("authentication", {})
        for field in authentication:
            if field in secret_params:
                model_args_clean["extra_body"]["dataSources"][0]["parameters"]["authentication"][field] = "*****"
        embeddingDependency = model_args_clean["extra_body"]["dataSources"][0]["parameters"].get("embeddingDependency", {})
        if "authentication" in embeddingDependency:
            for field in embeddingDependency["authentication"]:
                if field in secret_params:
                    model_args_clean["extra_body"]["dataSources"][0]["parameters"]["embeddingDependency"]["authentication"][field] = "*****"
        
    #logging.debug(f"REQUEST BODY: {json.dumps(model_args_clean, indent=4)}")

    #log event to eventhub
    send_log_event_to_eventhub(credential, "INFO", f"CREATING REQUEST TO OPENAI", "prepare_model_args", index_to_use.replace("-index",""),current_user,container_obj,start_datetime,"Success","App Action","Preparing an OpenAI request")
    
    return model_args

async def send_chat_request(cookies, request, index_to_use, current_user, container_obj, start_datetime):
    
    param_dict = make_param_dict_from_cookie(cookies)
    
    model_args = prepare_model_args(param_dict, request, index_to_use,current_user,container_obj,start_datetime)

    try:
        azure_openai_client = init_openai_client(index_to_use, param_dict)
        
        response = await azure_openai_client.chat.completions.create(**model_args)
        #print(response)
    except Exception as e:
        logging.exception("Exception in send_chat_request")
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in send_chat_request:{e},500", "send_chat_request", index_to_use.replace("-index",""), current_user ,container_obj, start_datetime,"Failed","App Action","Sending an OpenAI request")
        raise e

    return response

def num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18"):
    """Return the number of tokens used by a list of messages."""
    #try:
       
    #    encoding = tiktoken.encoding_for_model(model)
    #except KeyError:
    #    print("Warning: model not found. Using o200k_base encoding.")
    encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06"
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0125.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125")
    elif "gpt-4o-mini" in model:
        print("Warning: gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-mini-2024-07-18.")
        return num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18")
    elif "gpt-4o" in model:
        print("Warning: gpt-4o and gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-2024-08-06.")
        return num_tokens_from_messages(messages, model="gpt-4o-2024-08-06")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )
    num_tokens = 0

    for content_key in ['user_content', 'assistant_content', 'tool_citations_content', 'tool_intent']:  
        content_array = messages.get(content_key, [])  
        if not isinstance(content_array, list):  
            raise ValueError(f"The value for {content_key} must be a list.")  
  
        for content_string in content_array:  # Iterate over each string in the array  
            num_tokens += len(encoding.encode(content_string))  
  
    # Add tokens for any priming or structure if needed  
    num_tokens += 3  # Example: every reply is primed with <|start|>assistant<|message|>  
    return num_tokens 

def extract_content(response_data):  
    messages = response_data  
    user_content = []  
    assistant_content = []  
    tool_citations_content = []  
    tool_intent = []  
  
    for message in messages:  
        role = message.get('role')  
        content = message.get('content')  
  
        if role == 'user':  
            user_content.append(content)  
        elif role == 'assistant':  
            assistant_content.append(content)  
        elif role == 'tool':  
            # Extracting citations content  
            content_json = json.loads(content)  
            citations = content_json.get('citations', [])  
            for citation in citations:  
                if citation.get('content'):  
                    tool_citations_content.append(citation.get('content'))  
            # Extracting intent  
            if 'intent' in content_json:  
                # Directly parse the intent value as JSON and add it to the list  
                tool_intent_json = json.loads(content_json.get('intent'))  
                tool_intent.extend(tool_intent_json)  
  
    return {  
        'user_content': user_content,  
        'assistant_content': assistant_content,  
        'tool_citations_content': tool_citations_content,  
        'tool_intent': tool_intent  
    }

async def complete_chat_request(cookies, request_body, index_to_use,current_user,container_obj, start_datetime):
    response = await send_chat_request(cookies, request_body, index_to_use,current_user,container_obj,start_datetime)
    history_metadata = request_body.get("history_metadata", {})

    return format_non_streaming_response(response, history_metadata)

async def stream_chat_request(cookies, request_body, index_to_use, current_user, container_obj, start_datetime, max_retries=1, retry_delay=2):
    retry_count = 0  
    while retry_count <= max_retries:  
        try:  
            # Make the request and get the response async generator  
            response = await send_chat_request(  
                cookies, request_body, index_to_use, current_user, container_obj, start_datetime  
            )  
  
            buffer = ""  # Initialize an empty buffer to accumulate assistant responses  
            async for chunk in response:  
                # Format the chunk  
                formatted_chunk = format_stream_response(chunk, request_body.get("history_metadata", {}))  
  
                # Extract and accumulate assistant content only  
                messages = formatted_chunk.get("choices", [{}])[0].get("messages", [])  
                for message in messages:  
                    if message.get("role") == "assistant":  
                        buffer += message.get("content", "")  # Accumulate assistant content  
                        # print("Accumulated Buffer:", buffer)  
  
                       # Validate the accumulated assistant response content  
                        if invalid_phrase_in_buffer(buffer) and retry_count < max_retries:  
                            print(  
                                f"Invalid content detected in assistant response. Retrying... Attempt {retry_count + 1}/{max_retries}"  
                            ) 
                            buffer = ""  # Clear the buffer before retrying 
                            raise ValueError("Invalid content detected")  # Trigger retry logic    
  
                # Yield the current chunk (even if the buffer hasn't been fully validated yet)  
                yield formatted_chunk  
  
            # If the generator completes successfully, exit the retry loop  
            return  # Exit successfully  
  
        except Exception as e:  
            print(f"Error occurred during request: {e}")  
            retry_count += 1  
  
            if retry_count <= max_retries:  
                print(f"Retrying request... Attempt {retry_count}/{max_retries}")
                #log event to eventhub
                send_log_event_to_eventhub(credential, "INFO", f"Chat request Retried successfully, 200", "stream_chat_request", index_to_use.replace("-index",""), current_user, container_obj, start_datetime,"Success","App Action","Conversation/Prompting Retry Request")
                await asyncio.sleep(retry_delay)  
                
            else:  
                print("Max retries exceeded. Returning the last response as is.")  
                return
  
  
def invalid_phrase_in_buffer(buffer):  
    """  
    Function to check invalid phrases in OpenAI response across multiple languages.  
    """  
    # Dictionary of invalid phrases in different languages  
    invalid_phrases = {  
        "en": [  
            "Sorry, we couldn't find the information in MyTeva. Please try refining your query, using the retry button, or opening a support ticket for further assistance.",  
            "The requested information is not found in the retrieved data. Please try another query or topic."  
        ],  
        "he": [  
            "מצטערים, לא הצלחנו למצוא את המידע ב-MyTeva. אנא נסה להריץ מחדש את השאילתה באמצעות לחצן הניסיון מחדש, לחדד אותה או לפתוח כרטיס תמיכה לקבלת סיוע נוסף.",  
            "המידע המבוקש לא נמצא בנתונים שהתקבלו. אנא נסה שאילתה או נושא אחר."  
        ],  
        "de": [  
            "Entschuldigung, wir konnten die Informationen in MyTeva nicht finden. Bitte versuchen Sie, Ihre Anfrage zu verfeinern, verwenden Sie die Wiederholen-Schaltfläche oder öffnen Sie ein Support-Ticket für weitere Unterstützung.",  
            "Die angeforderten Informationen wurden in den abgerufenen Daten nicht gefunden. Bitte versuchen Sie eine andere Abfrage oder ein anderes Thema."  
        ]  
        # Add more languages as needed  
    }  
      
    # Normalize buffer input to lowercase for comparison  
    buffer_lower = buffer.lower()  
      
    # Check if any invalid phrase exists in the buffer across all languages  
    for phrases in invalid_phrases.values():  
        if any(phrase.lower() in buffer_lower for phrase in phrases):  
            return True  
      
    return False


async def conversation_internal(request_body, cookies, user, container_obj, start_datetime):
    index_to_use = None
    try:
        if SHOULD_STREAM:
            if cookies.get('index_to_use') is None:
                index_to_use = serializer.loads(cookies.get('index_container_to_use', None))
                print("I was fetched first time: " + str(index_to_use))
                
                # Check if CyrusGPT is enabled and this is an appropriate use case
                if CYRUSGPT_ENABLED and should_use_cyrusgpt(index_to_use, request_body):
                    # Use CyrusGPT for enhanced retrieval
                    result_list = await handle_cyrusgpt_request(
                        request_body, index_to_use, user, container_obj, start_datetime
                    )
                else:
                    # Use existing logic
                    result_list = []
                    async for chunk in stream_chat_request(
                        cookies,
                        request_body,
                        f'{index_to_use}-index',
                        user,
                        get_adgroup_by_usecase(container_obj, index_to_use),
                        start_datetime
                    ):
                        result_list.append(chunk)

                # Pass the collected result list to format_as_ndjson 
                response = await make_response(format_as_ndjson(result_list))
                response.timeout = None
                response.mimetype = "application/json-lines"
                
            else:
                index_container_to_use = serializer.loads(cookies.get('index_container_to_use'))
                index_to_use = serializer.loads(cookies.get('index_to_use'))
                
                # Check if CyrusGPT is enabled and this is an appropriate use case
                if CYRUSGPT_ENABLED and should_use_cyrusgpt(index_to_use, request_body):
                    # Use CyrusGPT for enhanced retrieval
                    result_list = await handle_cyrusgpt_request(
                        request_body, index_to_use, user, container_obj, start_datetime
                    )
                else:
                    # Use existing logic
                    result_list = []
                    async for chunk in stream_chat_request(
                        cookies,
                        request_body,
                        f'{index_to_use}-index',
                        user,
                        get_adgroup_by_usecase(container_obj, index_to_use),
                        start_datetime
                    ):
                        result_list.append(chunk)

                # Pass the collected result list to format_as_ndjson 
                response = await make_response(format_as_ndjson(result_list))
                response.timeout = None
                response.mimetype = "application/json-lines"
            
            return response
        else:
            index_to_use = serializer.loads(cookies.get('index_container_to_use', None))
            
            # Check if CyrusGPT is enabled and this is an appropriate use case
            if CYRUSGPT_ENABLED and should_use_cyrusgpt(index_to_use, request_body):
                # Use CyrusGPT for enhanced retrieval (non-streaming)
                result = await handle_cyrusgpt_request_non_streaming(
                    request_body, index_to_use, user, container_obj, start_datetime
                )
            else:
                # Use existing logic
                result = await complete_chat_request(request_body, index_to_use, user, get_adgroup_by_usecase(container_obj, index_to_use), start_datetime)
            
            return jsonify(result)
    
    except Exception as ex:
        logging.exception(ex)
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in Conversation :{ex}, 500", "conversation_internal", index_to_use.replace("-index","") if index_to_use else "unknown", user, get_adgroup_by_usecase(container_obj, index_to_use) if index_to_use else None, start_datetime, "Failed", "User Action", "Conversation/Prompting Request")
        return jsonify({"error": str(ex)}), 500

def should_use_cyrusgpt(index_to_use, request_body):
    """
    Determine whether to use CyrusGPT based on the use case and configuration.
    """
    # First check if CyrusGPT is globally enabled
    if not CYRUSGPT_ENABLED:
        return False
    
    if not index_to_use:
        return False
    
    # Extract the use case name from index_to_use
    # Remove '-index' suffix if present
    use_case = index_to_use.replace('-index', '')
    
    # Check if this specific use case has CyrusGPT enabled
    if use_case in CYRUSGPT_ENABLED_USECASES:
        is_enabled = CYRUSGPT_ENABLED_USECASES[use_case]
        logging.debug(f"CyrusGPT for use case '{use_case}': {'enabled' if is_enabled else 'disabled'}")
        return is_enabled
    
    # Fallback to the original logic if not explicitly configured
    cyrusgpt_enabled_containers = [
        'uscomm-ana', 'rnd', 'cdc', 'gsc', 'us-contracts', 
        'dset', 'im-portfolio', 'im-sales', 'gia'
    ]
    
    # Check if any of the enabled containers match
    is_enabled = any(container in use_case for container in cyrusgpt_enabled_containers)
    logging.debug(f"CyrusGPT for use case '{use_case}' (fallback logic): {'enabled' if is_enabled else 'disabled'}")
    
    return is_enabled

def get_cyrusgpt_config_for_usecase(use_case):
    """
    Get CyrusGPT configuration specific to a use case, with fallbacks to global config.
    """
    # Remove '-index' suffix if present
    clean_use_case = use_case.replace('-index', '')
    
    # Start with global configuration
    config = {
        'temperature': CYRUSGPT_TEMPERATURE,
        'top_p': CYRUSGPT_TOP_P,
        'deployment': CYRUSGPT_DEPLOYMENT,
        'max_tokens': CYRUSGPT_MAX_TOKENS,
        'seed': CYRUSGPT_SEED,
        'top_k': CYRUSGPT_TOP_K,
        'system_prompt_variant': CYRUSGPT_SYSTEM_PROMPT_VARIANT,
        'retrieval_method': CYRUSGPT_RETRIEVAL_METHOD,
        'query_expansion': CYRUSGPT_QUERY_EXPANSION,
        'hybrid_search': CYRUSGPT_HYBRID_SEARCH,
        'advanced_reranking': CYRUSGPT_ADVANCED_RERANKING,
        'diversity_selection': CYRUSGPT_DIVERSITY_SELECTION
    }
    
    # Override with use case-specific configuration if available
    if clean_use_case in CYRUSGPT_USECASE_CONFIG:
        use_case_config = CYRUSGPT_USECASE_CONFIG[clean_use_case]
        config.update(use_case_config)
        logging.debug(f"Using custom CyrusGPT config for use case '{clean_use_case}': {use_case_config}")
    
    return config

async def handle_cyrusgpt_request(request_body, index_to_use, user, container_obj, start_datetime):
    """
    Handle requests using CyrusGPT with enhanced retrieval capabilities.
    Returns a list of formatted chunks for streaming responses.
    """
    try:
        # Get use case-specific configuration
        use_case_config = get_cyrusgpt_config_for_usecase(index_to_use)
        
        # Initialize CyrusGPT with use case-specific configuration
        cyrus_gpt = CyrusGPT(
            temperature=use_case_config['temperature'],
            top_p=use_case_config['top_p'],
            index_name=f'{index_to_use}-index',
            container_name=index_to_use,
            system_prompt_variant=use_case_config['system_prompt_variant'],
            deployment=use_case_config['deployment'],
            max_tokens=use_case_config['max_tokens'],
            seed=use_case_config['seed'],
            top_k=use_case_config['top_k'],
            retrieval_method=use_case_config['retrieval_method'],
            query_expansion=use_case_config['query_expansion'],
            hybrid_search=use_case_config['hybrid_search'],
            advanced_reranking=use_case_config['advanced_reranking'],
            diversity_selection=use_case_config['diversity_selection']
        )
        
        # Extract the user query from the request
        messages = request_body.get("messages", [])
        if not messages or messages[-1]['role'] != 'user':
            raise ValueError("No user message found in request")
        
        user_query = messages[-1]['content']
        
        # Get response from CyrusGPT
        cyrus_response = await cyrus_gpt.query(user_query)
        
        # Convert CyrusGPT response to the expected format for streaming
        result_list = []
        if cyrus_response and cyrus_response.get('output'):
            # Create a formatted chunk that matches the existing format
            formatted_chunk = {
                "id": str(uuid.uuid4()),
                "object": "chat.completion.chunk",
                "choices": [{
                    "messages": [{
                        "role": "assistant",
                        "content": cyrus_response['output']
                    }]
                }],
                "history_metadata": request_body.get("history_metadata", {}),
                "cyrusgpt_config": {
                    "use_case": index_to_use,
                    "retrieval_method": use_case_config['retrieval_method'],
                    "deployment": use_case_config['deployment']
                }
            }
            result_list.append(formatted_chunk)
        
        #log event to eventhub
        send_log_event_to_eventhub(
            credential, "INFO", 
            f"CyrusGPT request completed successfully with config: {use_case_config['retrieval_method']}", 
            "handle_cyrusgpt_request", 
            index_to_use, user, 
            get_adgroup_by_usecase(container_obj, index_to_use), 
            start_datetime, "Success", "User Action", 
            "CyrusGPT Enhanced Retrieval Request"
        )
        
        return result_list
        
    except Exception as e:
        logging.exception(f"Error in CyrusGPT request: {e}")
        #log event to eventhub
        send_log_event_to_eventhub(
            credential, "ERROR", 
            f"Exception in CyrusGPT request: {e}", 
            "handle_cyrusgpt_request", 
            index_to_use, user, 
            get_adgroup_by_usecase(container_obj, index_to_use), 
            start_datetime, "Failed", "User Action", 
            "CyrusGPT Enhanced Retrieval Request"
        )
        # Fallback to empty result list
        return []
        
async def handle_cyrusgpt_request_non_streaming(request_body, index_to_use, user, container_obj, start_datetime):
    """
    Handle non-streaming requests using CyrusGPT with enhanced retrieval capabilities.
    Returns a result dictionary in the expected format.
    """
    try:
        # Initialize CyrusGPT with configuration
        cyrus_gpt = CyrusGPT(
            temperature=CYRUSGPT_TEMPERATURE,
            top_p=CYRUSGPT_TOP_P,
            index_name=f'{index_to_use}-index',
            container_name=index_to_use,
            system_prompt_variant=CYRUSGPT_SYSTEM_PROMPT_VARIANT,
            deployment=CYRUSGPT_DEPLOYMENT,
            max_tokens=CYRUSGPT_MAX_TOKENS,
            seed=CYRUSGPT_SEED,
            top_k=CYRUSGPT_TOP_K,
            retrieval_method=CYRUSGPT_RETRIEVAL_METHOD,
            query_expansion=CYRUSGPT_QUERY_EXPANSION,
            hybrid_search=CYRUSGPT_HYBRID_SEARCH,
            advanced_reranking=CYRUSGPT_ADVANCED_RERANKING,
            diversity_selection=CYRUSGPT_DIVERSITY_SELECTION
        )
        
        # Extract the user query from the request
        messages = request_body.get("messages", [])
        if not messages or messages[-1]['role'] != 'user':
            raise ValueError("No user message found in request")
        
        user_query = messages[-1]['content']
        
        # Get response from CyrusGPT
        cyrus_response = await cyrus_gpt.query(user_query)
        
        # Convert CyrusGPT response to the expected format
        result = {
            "output": cyrus_response.get('output', ''),
            "full_output": cyrus_response.get('full_output')
        }
        
        #log event to eventhub
        send_log_event_to_eventhub(
            credential, "INFO", 
            f"CyrusGPT non-streaming request completed successfully", 
            "handle_cyrusgpt_request_non_streaming", 
            index_to_use, user, 
            get_adgroup_by_usecase(container_obj, index_to_use), 
            start_datetime, "Success", "User Action", 
            "CyrusGPT Enhanced Retrieval Request"
        )
        
        return result
        
    except Exception as e:
        logging.exception(f"Error in CyrusGPT non-streaming request: {e}")
        #log event to eventhub
        send_log_event_to_eventhub(
            credential, "ERROR", 
            f"Exception in CyrusGPT non-streaming request: {e}", 
            "handle_cyrusgpt_request_non_streaming", 
            index_to_use, user, 
            get_adgroup_by_usecase(container_obj, index_to_use), 
            start_datetime, "Failed", "User Action", 
            "CyrusGPT Enhanced Retrieval Request"
        )
        # Return empty result as fallback
        return {"output": "", "full_output": None}

def get_adgroup_by_usecase(container_obj, search_value):
    try:
        for key, value in container_obj.items():  
            if value == search_value:  
                return key  
    except Exception as e:
        return None 

def get_container_obj(request, current_user):  
    userToken = request.headers.get('X-Ms-Token-Aad-Id-Token', 'eyJ0eXAiOiJKV1QiLCJub25jZSI6IjFvUVlyb2pIbm96OGpEX0Y1ZDMzbWZnbjJ6SVNFWi1JY0hlQmJFbmRtQkUiLCJhbGciOiJSUzI1NiIsIng1dCI6IkpZaEFjVFBNWl9MWDZEQmxPV1E3SG4wTmVYRSIsImtpZCI6IkpZaEFjVFBNWl9MWDZEQmxPV1E3SG4wTmVYRSJ9.eyJhdWQiOiJodHRwczovL2dyYXBoLm1pY3Jvc29mdC5jb20iLCJpc3MiOiJodHRwczovL3N0cy53aW5kb3dzLm5ldC8zZjk5MWE3Yi1lYTkzLTQxNjktYjI4Yy1jMzZmZjNlNWIwZDEvIiwiaWF0IjoxNzUzODgzMTQzLCJuYmYiOjE3NTM4ODMxNDMsImV4cCI6MTc1Mzg4NzY5NCwiYWNjdCI6MCwiYWNyIjoiMSIsImFjcnMiOlsicDEiLCJ1cm46dXNlcjpyZWdpc3RlcnNlY3VyaXR5aW5mbyJdLCJhaW8iOiJBWFFBaS84WkFBQUFWLzNFVVczUFV4aiszUFpoaEpHOWNlV3pKWC94OW9IL2cvcFBLVGJzTHRsR1JWN3dHVmtINU03MVBtZnY5MTcrUksvaGtDVlgxajhJZmZWSUN5MVR5SmZ5ZHRkRTkrWUJ3V3ZLUnlyRmJ3TlIvcWFBeXNXcHVZa05YRW94Q3o1YnVoaURjRGRIeDdHNy9nR1lnb1AxdWc9PSIsImFtciI6WyJwd2QiLCJyc2EiLCJtZmEiXSwiYXBwX2Rpc3BsYXluYW1lIjoiTWljcm9zb2Z0IEF6dXJlIENMSSIsImFwcGlkIjoiMDRiMDc3OTUtOGRkYi00NjFhLWJiZWUtMDJmOWUxYmY3YjQ2IiwiYXBwaWRhY3IiOiIwIiwiY2Fwb2xpZHNfbGF0ZWJpbmQiOlsiYzYyZWY3MDYtYmZkZi00OTVkLThhYjEtZGU0ZDljNmI4ZTY1Il0sImRldmljZWlkIjoiMzc0YjVlNTItMGI2ZS00ZjE5LTllMTgtZTQ1ZDAyZGNmNjVlIiwiZmFtaWx5X25hbWUiOiJLdXJkIiwiZ2l2ZW5fbmFtZSI6IkN5cnVzIiwiaWR0eXAiOiJ1c2VyIiwiaW5fY29ycCI6InRydWUiLCJpcGFkZHIiOiIxNjIuMjUxLjEwNC4xMCIsIm5hbWUiOiJDeXJ1cyBLdXJkIiwib2lkIjoiMWVhNjA2YjEtY2IyMy00OTcxLWJlMzAtNGJiMWVkNTk0YjM5Iiwib25wcmVtX3NpZCI6IlMtMS01LTIxLTI5ODgyNjg5OTMtMjQ2NDM3NTA5Ni0zNzI1MzI2NDYtNTY1MTg0IiwicGxhdGYiOiIzIiwicHVpZCI6IjEwMDMyMDA0N0Y3NDJFODkiLCJyaCI6IjEuQVlJQWV4cVpQNVBxYVVHeWpNTnY4LVd3MFFNQUFBQUFBQUFBd0FBQUFBQUFBQUNxQUJLQ0FBLiIsInNjcCI6IkFwcGxpY2F0aW9uLlJlYWRXcml0ZS5BbGwgQXBwUm9sZUFzc2lnbm1lbnQuUmVhZFdyaXRlLkFsbCBBdWRpdExvZy5SZWFkLkFsbCBEZWxlZ2F0ZWRQZXJtaXNzaW9uR3JhbnQuUmVhZFdyaXRlLkFsbCBEaXJlY3RvcnkuQWNjZXNzQXNVc2VyLkFsbCBlbWFpbCBHcm91cC5SZWFkV3JpdGUuQWxsIG9wZW5pZCBwcm9maWxlIFVzZXIuUmVhZC5BbGwgVXNlci5SZWFkV3JpdGUuQWxsIiwic2lkIjoiMDA2YjRjZDktOWYxYy04YmMyLTc0MWMtZDRhYWQ1ZWU0NzM5Iiwic2lnbmluX3N0YXRlIjpbImR2Y19tbmdkIiwiZHZjX2RtamQiLCJpbmtub3dubnR3ayIsImttc2kiXSwic3ViIjoiSjVKTmxMN0tZUDFtdld5ckFUd281aXpiTURGaFNKRVpzUGwybjV1bC1PdyIsInRlbmFudF9yZWdpb25fc2NvcGUiOiJFVSIsInRpZCI6IjNmOTkxYTdiLWVhOTMtNDE2OS1iMjhjLWMzNmZmM2U1YjBkMSIsInVuaXF1ZV9uYW1lIjoiQ3lydXMuS3VyZEB0ZXZhcGhhcm0uY29tIiwidXBuIjoiQ3lydXMuS3VyZEB0ZXZhcGhhcm0uY29tIiwidXRpIjoibTdjUTQ5Y3lnVVNTTFl3bUlwVm1BQSIsInZlciI6IjEuMCIsIndpZHMiOlsiYjc5ZmJmNGQtM2VmOS00Njg5LTgxNDMtNzZiMTk0ZTg1NTA5Il0sInhtc19jYyI6WyJDUDEiXSwieG1zX2Z0ZCI6ImFYNVgxTmU1UElYNjNRWVlSN25XSHdOaDFfc2tGc1FObkdnVUxYd0Z5WWtCWm5KaGJtTmxZeTFrYzIxeiIsInhtc19pZHJlbCI6IjEyIDEiLCJ4bXNfc3NtIjoiMSIsInhtc19zdCI6eyJzdWIiOiJvQ204VDBOT0I4TGJwMm80bEZfcHJnRXJvaVI4RDNCM2ZaUEVNbXlkNy1jIn0sInhtc190Y2R0IjoxMzk2ODkzNDYwfQ.oPVzUxXSo7Dk2Rld570JC7gGVjvPq9C0pf_5klaQprDoTDfcqFkwAFPuPBhduCj2TmkUpXbTX5rY_u1Bf-FIiPkHdz6WdTpWGCkFBbAYyVxwKystKDGc_RLyPSzctoz02yzJTwhNZA-8Y3_T3a4rh5EYOGoto6QOMp_magh-i0ijtuL2nLQ3nNP62_mPHWNGIg6vl9xtNqfCOnS-PxIybt3oXSewFHyjhWfxt26TgmNwonjfSTb20GNoBeOEUt-fentWpN4vkOdguTlTwrKu85R1Ea2PHzLjFojWnnrmLFq-cp1HSDZD83os0lz-kxo15ym3ebQpi8IveeHlOXAbow')

    # Fetch the list of group objects the user belongs to  
    groups = []
    grps_id_list = cached_fetch_user_group_names(userToken)
    
    # Extract the group IDs from the list of group objects  
    groups = [obj['id'] for obj in grps_id_list]  
      
    # Map the current user to an index synchronously with the group IDs  
    container_obj = map_user_to_index_sync(current_user, groups)  
      
    return container_obj  

def map_user_to_index_sync(current_user, user_groups):
    
    config = {}
    if INDEX_GROUP_CONFIG is not None:
        config = json.loads(INDEX_GROUP_CONFIG)
    else:
        config = {}
    containers = []
    container_obj = {}
    for ad_group in user_groups:
        if config.get(ad_group) is not None:
            container_obj[ad_group] = config.get(ad_group)[0]
            #containers = containers  + config.get(ad_group)
    
    if len(container_obj) >=2 and container_obj.get("11f37fb4-0db0-4fcf-a1af-27f10c39a60f") is not None:
        container_obj.pop("11f37fb4-0db0-4fcf-a1af-27f10c39a60f")
    
    # remove after testing is complete
    if len(container_obj) > 0:
        return container_obj
    else:
        return {}


async def get_user_groups(email):
    
    other_token_provider =  get_bearer_token_provider(
    DefaultAzureCredential(), "https://graph.microsoft.com/.default")
    token = await other_token_provider()
    headers = {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json"
    }

    user_url = f"https://graph.microsoft.com/v1.0/users/{email}"
    response = requests.get(user_url, headers=headers)

    if response.status_code == 200:
        user_id = response.json()["id"]
        groups_url = f"https://graph.microsoft.com/v1.0/users/{user_id}/memberOf"
        groups_response = requests.get(groups_url, headers=headers)
        if groups_response.status_code == 200:
            groups = groups_response.json()["value"]
            return [group["displayName"] for group in groups]
        else:
            print("Failed to fetch user's groups:", groups_response.text)
            return "Null"
    else:
        print("Failed to fetch user details:", response.text)
        return "Null"

@bp.route("/conversation", methods=["POST"])
async def conversation():
    start_datetime = datetime.now()
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415
    request_json = await request.get_json()
    current_user = get_authenticated_user_details(request_headers=request.headers)
    # Get containber obj i.e usecase, ad group mapping
    container_obj = get_container_obj(request,current_user)
        
    print(current_user['user_name'])
    return await conversation_internal(request_json, request.cookies,current_user['user_name'],container_obj,start_datetime)

@bp.route("/cyrusgpt/query", methods=["POST"])
async def cyrusgpt_query():
    """
    Direct API endpoint for CyrusGPT queries with enhanced retrieval.
    """
    start_datetime = datetime.now()
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415
    
    if not CYRUSGPT_ENABLED:
        return jsonify({"error": "CyrusGPT is not enabled"}), 400
    
    request_json = await request.get_json()
    current_user = get_authenticated_user_details(request_headers=request.headers)
    container_obj = get_container_obj(request, current_user)
    
    try:
        # Get query parameters
        query = request_json.get("query", "")
        index_name = request_json.get("index_name", CYRUSGPT_INDEX_NAME)
        container_name = request_json.get("container_name", CYRUSGPT_CONTAINER_NAME)
        
        if not query:
            return jsonify({"error": "query parameter is required"}), 400
        
        # Initialize CyrusGPT
        cyrus_gpt = CyrusGPT(
            temperature=request_json.get("temperature", CYRUSGPT_TEMPERATURE),
            top_p=request_json.get("top_p", CYRUSGPT_TOP_P),
            index_name=index_name,
            container_name=container_name,
            system_prompt_variant=request_json.get("system_prompt_variant", CYRUSGPT_SYSTEM_PROMPT_VARIANT),
            deployment=request_json.get("deployment", CYRUSGPT_DEPLOYMENT),
            max_tokens=request_json.get("max_tokens", CYRUSGPT_MAX_TOKENS),
            seed=request_json.get("seed", CYRUSGPT_SEED),
            top_k=request_json.get("top_k", CYRUSGPT_TOP_K),
            retrieval_method=request_json.get("retrieval_method", CYRUSGPT_RETRIEVAL_METHOD),
            query_expansion=request_json.get("query_expansion", CYRUSGPT_QUERY_EXPANSION),
            hybrid_search=request_json.get("hybrid_search", CYRUSGPT_HYBRID_SEARCH),
            advanced_reranking=request_json.get("advanced_reranking", CYRUSGPT_ADVANCED_RERANKING),
            diversity_selection=request_json.get("diversity_selection", CYRUSGPT_DIVERSITY_SELECTION)
        )
        
        # Execute query
        response = await cyrus_gpt.query(
            query,
            folder_name=request_json.get("folder_name"),
            filename=request_json.get("filename")
        )
        
        #log event to eventhub
        send_log_event_to_eventhub(
            credential, "INFO", 
            f"Direct CyrusGPT query completed successfully", 
            "cyrusgpt_query", 
            container_name, current_user['user_name'], 
            get_adgroup_by_usecase(container_obj, container_name), 
            start_datetime, "Success", "User Action", 
            "Direct CyrusGPT API Query"
        )
        
        return jsonify(response), 200
        
    except Exception as e:
        logging.exception("Exception in /cyrusgpt/query")
        #log event to eventhub
        send_log_event_to_eventhub(
            credential, "ERROR", 
            f"Exception in /cyrusgpt/query: {e}", 
            "cyrusgpt_query", 
            container_name if 'container_name' in locals() else "unknown", 
            current_user['user_name'], 
            get_adgroup_by_usecase(container_obj, container_name) if 'container_name' in locals() else None, 
            start_datetime, "Failed", "User Action", 
            "Direct CyrusGPT API Query"
        )
        return jsonify({"error": str(e)}), 500

@bp.route("/cyrusgpt/upload", methods=["POST"])
async def cyrusgpt_upload():
    """
    Upload files to CyrusGPT blob storage and trigger indexing.
    """
    start_datetime = datetime.now()
    if not CYRUSGPT_ENABLED:
        return jsonify({"error": "CyrusGPT is not enabled"}), 400
    
    current_user = get_authenticated_user_details(request_headers=request.headers)
    container_obj = get_container_obj(request, current_user)
    
    try:
        files = await request.files
        uploaded_files = files.getlist('files')
        folder_name = request.args.get('folder_name')
        is_overwrite = request.args.get('is_overwrite', 'true').lower() == 'true'
        container_name = request.args.get('container_name', CYRUSGPT_CONTAINER_NAME)
        
        if not uploaded_files:
            return jsonify({"error": "No files provided"}), 400
        
        # Initialize CyrusGPT
        cyrus_gpt = CyrusGPT(container_name=container_name)
        
        # Upload files
        uploaded_file_paths = []
        for file in uploaded_files:
            # Save file temporarily
            temp_file_path = f"/tmp/{file.filename}"
            await file.save(temp_file_path)
            
            # Upload to blob storage
            success = await cyrus_gpt.upload_file(
                temp_file_path, 
                folder_name=folder_name, 
                is_overwrite=is_overwrite
            )
            
            if success:
                uploaded_file_paths.append(temp_file_path)
            
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        
        # Trigger indexer if files were uploaded
        if uploaded_file_paths:
            await cyrus_gpt.trigger_indexer()
        
        #log event to eventhub
        send_log_event_to_eventhub(
            credential, "INFO", 
            f"CyrusGPT file upload completed: {len(uploaded_file_paths)} files", 
            "cyrusgpt_upload", 
            container_name, current_user['user_name'], 
            get_adgroup_by_usecase(container_obj, container_name), 
            start_datetime, "Success", "User Action", 
            "CyrusGPT File Upload"
        )
        
        return jsonify({
            "message": f"Successfully uploaded {len(uploaded_file_paths)} files",
            "uploaded_files": [os.path.basename(path) for path in uploaded_file_paths]
        }), 201
        
    except Exception as e:
        logging.exception("Exception in /cyrusgpt/upload")
        #log event to eventhub
        send_log_event_to_eventhub(
            credential, "ERROR", 
            f"Exception in /cyrusgpt/upload: {e}", 
            "cyrusgpt_upload", 
            container_name if 'container_name' in locals() else "unknown", 
            current_user['user_name'], 
            get_adgroup_by_usecase(container_obj, container_name) if 'container_name' in locals() else None, 
            start_datetime, "Failed", "User Action", 
            "CyrusGPT File Upload"
        )
        return jsonify({"error": str(e)}), 500

@bp.route("/cyrusgpt/config", methods=["GET"])
def cyrusgpt_config():
    """
    Get current CyrusGPT configuration.
    """
    if not CYRUSGPT_ENABLED:
        return jsonify({"error": "CyrusGPT is not enabled"}), 400
    
    config = {
        "enabled": CYRUSGPT_ENABLED,
        "temperature": CYRUSGPT_TEMPERATURE,
        "top_p": CYRUSGPT_TOP_P,
        "max_tokens": CYRUSGPT_MAX_TOKENS,
        "seed": CYRUSGPT_SEED,
        "top_k": CYRUSGPT_TOP_K,
        "deployment": CYRUSGPT_DEPLOYMENT,
        "index_name": CYRUSGPT_INDEX_NAME,
        "container_name": CYRUSGPT_CONTAINER_NAME,
        "system_prompt_variant": CYRUSGPT_SYSTEM_PROMPT_VARIANT,
        "retrieval_method": CYRUSGPT_RETRIEVAL_METHOD,
        "query_expansion": CYRUSGPT_QUERY_EXPANSION,
        "hybrid_search": CYRUSGPT_HYBRID_SEARCH,
        "advanced_reranking": CYRUSGPT_ADVANCED_RERANKING,
        "diversity_selection": CYRUSGPT_DIVERSITY_SELECTION
    }
    
    return jsonify(config), 200

async def stream_assistant_request(client, assistant_id, thread_id,history_metadata):
    response_stream = await client.beta.threads.runs.create(
    assistant_id="asst_22Ato2XdpsixwE7h3HoReGVa",
    stream=True,
    thread_id=thread_id,
    )
    
    async def generate():
            async for completionChunk in response_stream:
                yield process_event(completionChunk,history_metadata)

    return generate()

def process_event(event,history_metadata, **kwargs):
        response_obj = {
            "id": event.data.id,
            # "model": null,
            # "created": null,
            "object": event.data.object,
            "choices": [{
                "messages": []
            }],
            "history_metadata": history_metadata[0],
            # "usage": chatCompletion.usage
            
            }
        if isinstance(event, ThreadMessageDelta):
            
            data = event.data.delta.content
            for text in data:
                messageObj = {
                    "role": "assistant",
                    "context": text.text.value,
                }
                response_obj["choices"][0]["messages"].append(messageObj)
                return response_obj
                # print(text.text.value, end='', flush=True)
        
        elif isinstance(event, ThreadRunStepDelta):
            tool_calls = event.data.delta.step_details.tool_calls
            for code in tool_calls:
                messageObj = {
                    "role": "tool_call",
                    "context": code.code_interpreter.input,
                }
                response_obj["choices"][0]["messages"].append(messageObj)
                return response_obj
        elif any(isinstance(event, cls) for cls in [ThreadRunFailed, ThreadRunCancelling, ThreadRunCancelled,
                                                    ThreadRunExpired, ThreadRunStepFailed, ThreadRunStepCancelled]):
            raise Exception("Run failed")

        elif isinstance(event, ThreadRunCompleted):
            print("\nRun completed")
            return response_obj
        
        
 
        # Handle other event types like ThreadRunQueued, ThreadRunStepInProgress, ThreadRunInProgress
        else:
            print("\nRun in progress")
            return response_obj

@bp.route("/create-message", methods=["POST"])
async def create_message():
    start_datetime = datetime.now()
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415
    request_json = await request.get_json()
    current_user = get_authenticated_user_details(request_headers=request.headers)
    # Get containber obj i.e usecase, ad group mapping
    container_obj = get_container_obj(request,current_user)
        
    print(current_user['user_name'])
    return await create_message_internal(request_json, request.cookies,current_user['user_name'],container_obj,start_datetime)

async def create_message_internal(request_json, cookies, user, container_obj, start_datetime):
    index_container_to_use = None 
    try:

        request_messages = request_json.get("messages", {})
        history_metadata = request_json.get("history_metadata", {}),
        
        if cookies.get('index_container_to_use', None) is not None: 
            index_container_to_use = serializer.loads(cookies.get('index_container_to_use'))
        file_list = request_json.get("file_filter", [])
        base_filename_list = {}
        for file in file_list:
            if "." in file:
                file_path_parts = file.split('\/')
                length_of_path = len(file_path_parts)
                base_filename_list[file_path_parts[length_of_path-1]] = file.replace("/","",1)
        ad_token_provider = get_bearer_token_provider(DefaultAzureCredentialSync(), "https://cognitiveservices.azure.com/.default")
        logging.debug("base_filename_dict" + str(base_filename_list))
        client = AsyncAzureOpenAI(
            azure_ad_token_provider=ad_token_provider, 
            api_version="2024-08-01-preview",
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        assistant_files = await get_assistant_files(index_container_to_use,user,container_obj)
        assistant_files = await assistant_files.get_json()
        logging.debug(assistant_files)
        assistant_only_file_names_dict = {}


        attaching_file_id_dict = {}

        for obj in assistant_files:
            assistant_only_file_names_dict[obj["filename"]] = obj["id"] 
        logging.debug(assistant_only_file_names_dict)
        new_file_list = []
        # filter files that are new and needs uploading

        for key, value in base_filename_list.items():
            if value not in assistant_only_file_names_dict.keys():
                new_file_list.append(base_filename_list[key])
            else:
                attaching_file_id_dict[value] = assistant_only_file_names_dict[value]

        #upload new files to assistant
        logging.debug(new_file_list)
        final_file_id_dict = {}
        if len(new_file_list) > 0 :
            response, status = await upload_blob_to_openai(new_file_list, index_container_to_use,user,container_obj)
            res_files = await response.get_json()
            logging.debug(res_files)
            file_ids = res_files['file_ids']  
            if file_ids:
                final_file_id_dict = {**file_ids , **attaching_file_id_dict}  
            logging.debug("file_ids_new" + str(file_ids) )
        else:
            final_file_id_dict = attaching_file_id_dict
        logging.debug("final_id" + str(final_file_id_dict))
        thread_id = request_json.get('thread_id', None)
        assistant_id = request_json['assistant_id']
        ad_token_provider = get_bearer_token_provider(DefaultAzureCredentialSync(), "https://cognitiveservices.azure.com/.default")

        final_file_id_list = list(final_file_id_dict.values())

        if thread_id is None:
            thread = await client.beta.threads.create(
            tool_resources={"code_interpreter":{"file_ids": final_file_id_list}}
            )
            thread_id = thread.id
            print(thread.id)
        else:
            thread = await client.beta.threads.update(thread_id,
                tool_resources={"code_interpreter":{"file_ids": final_file_id_list}}
            )
        logging.debug(request_messages["content"])
        message = await client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=request_messages["content"]
            )
            
        result = await stream_assistant_request(client, assistant_id,thread_id,history_metadata)
        response = await make_response(format_as_ndjson(result))
        response.timeout = None
        response.mimetype = "application/json-lines"
        return response
    except Exception as e:
        logging.exception(e)
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in Create message Internal :{e}, 500", "create_message_internal", index_container_to_use, user, get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","User Action","Conversation/Prompting Request with Assistant")
        return jsonify({"error": str(e)}), 500

@bp.route("/create-thread", methods=["POST"])
async def create_thread():
    start_datetime = datetime.now()
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415
    index_container_to_use = None
    ad_token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    current_user = get_authenticated_user_details(request_headers=request.headers)
    # Get containber obj i.e usecase, ad group mapping
    container_obj = get_container_obj(request,current_user)
    if request.cookies.get('index_container_to_use', None) is not None: 
        index_container_to_use = serializer.loads(request.cookies.get('index_container_to_use'))
    #recieve details as part of the POST json to map this thread to a user and its chat history
    client = AsyncAzureOpenAI(
            azure_ad_token_provider=ad_token_provider, 
            api_version="2024-08-01-preview",
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    
    # append the id from previous step
    thread = await client.beta.threads.create(
        tool_resources={"code_interpreter":{"file_ids": []}}
    )

    logging.debug(thread)
    # Add code to map this thread id to a chat history session
    #log event to eventhub
    send_log_event_to_eventhub(credential, "INFO", f"Created Thread Successfully, thread_id:  {thread.id}, 200", "create_thread", index_container_to_use, current_user['user_name'], get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Success","App Action","Creating thread to chat with Assistant")            
    return jsonify({'thread_id': thread.id}), 200 

async def get_assistant_files(container_to_use,user,container_obj):
    start_datetime = datetime.now()
    try:
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        upload_url = f"{azure_endpoint}/openai/files?api-version=2024-10-21"
        azure_ad_token_provider = get_bearer_token_provider(
                DefaultAzureCredentialSync(), "https://cognitiveservices.azure.com/.default")
        token_json = await azure_ad_token_provider()
        headers = {
                    'Authorization': f"Bearer {token_json}"  
                }

        # Make the POST request to upload the file
        response = requests.get(upload_url, headers=headers)

        # Check if the upload was successful
        if response.status_code == 200:
            response_data = response.json()
        return jsonify(response_data['data'])
    except Exception as e:
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in fetching file from Assistant:{e}, 500", "get_assistant_files", container_to_use, user, get_adgroup_by_usecase(container_obj,container_to_use),start_datetime,"Failed","App Action","Fetching blobs from Openai")
        
        print(f"Error getting files from openai resources")
        return None

async def upload_blob_to_openai(file_list, container_to_use,user,container_obj):
    start_datetime = datetime.now()
    uploaded_files = {}
    try:
        # Define the upload URL
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        upload_url = f"{azure_endpoint}/openai/files?api-version=2024-10-21"
        azure_ad_token_provider = get_bearer_token_provider(
                DefaultAzureCredentialSync(), "https://cognitiveservices.azure.com/.default")
        token_json = await azure_ad_token_provider()
        blob_service_client = BlobServiceClient(account_url="https://stviovcaikeunopoc.blob.core.windows.net/", credential=credential)
        container_client = blob_service_client.get_container_client(container_to_use)  
        for file_name in file_list:
        # Get the blob client for the specified blob  
            try:
                blob_client = container_client.get_blob_client(file_name)
                # Download the blob content as a stream
                blob_stream = blob_client.download_blob().readall()

                # Define the form data
                files = {
                    'file': (file_name, blob_stream),
                    'purpose': (None, 'assistants')
                }
                
                headers = {
                    'Authorization': f"Bearer {token_json}"
                }
                
                # Make the POST request to upload the file
                response = requests.post(upload_url, files=files, headers=headers)

                # Check if the upload was successful
                if response.status_code == 200:
                    response_data = response.json()
                    uploaded_files[file_name] = response_data.get('id')
                    
                else:
                    #log event to eventhub
                    send_log_event_to_eventhub(credential, "ERROR", f"Failed to upload {file_name}. Status code: {response.status_code}, Response: {response.text}", "upload_blob_to_openai", container_to_use, user, get_adgroup_by_usecase(container_obj,container_to_use),start_datetime,"Failed","App Action","Uploading blob to openai")
        
                    print(f"Failed to upload {file_name}. Status code: {response.status_code}, Response: {response.text}")
                    return jsonify({"error": f"Failed to upload {file_name}. Status code: {response.status_code}, Response: {response.text}" }), 500

            except Exception as e:
                #log event to eventhub
                send_log_event_to_eventhub(credential, "ERROR", f"Error uploading blob {file_name}: {e}, 500", "upload_blob_to_openai", container_to_use, user, get_adgroup_by_usecase(container_obj,container_to_use),start_datetime,"Failed","App Action","Uploading blob to openai")   
                print(f"Error uploading blob {file_name}: {e}")
                return None
        #log event to eventhub
        send_log_event_to_eventhub(credential, "INFO", f"Upload Successful, file_ids:  {uploaded_files}, 200", "upload_blob_to_openai", container_to_use, user, get_adgroup_by_usecase(container_obj,container_to_use),start_datetime,"Success","App Action","Uploading blob to openai")          
        return jsonify({"Success": "Upload Successful", "file_ids":  uploaded_files}), 200
    except Exception as e:
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in Uploading blob to openai :{e}, 500", "upload_blob_to_openai", container_to_use, user, get_adgroup_by_usecase(container_obj,container_to_use),start_datetime,"Failed","App Action","Uploading blob to openai")
        return jsonify({"error": e}), 500

@bp.route("/toggle-chat", methods=["POST"])
async def update_mode():
    start_datetime = datetime.now()
    request_body = await request.get_json()
    index_container_to_use = None
    current_user = get_authenticated_user_details(request_headers=request.headers)
    # Get containber obj i.e usecase, ad group mapping
    container_obj = get_container_obj(request,current_user)
    if request.cookies.get('index_container_to_use', None) is not None: 
        index_container_to_use = serializer.loads(request.cookies.get('index_container_to_use'))
    app_mode = request_body.get("app_mode", None)
    assistant_config = request_body.get("assistant_config", None)
    assistant_id = None
    if app_mode is not None: 
        response_json = {'success': f'App mode set to {app_mode}'}
        
        if app_mode.lower() == 'tabular':
            if request.cookies.get('ASSISTANT_ID', None) is None:
                ad_token_provider = get_bearer_token_provider(DefaultAzureCredentialSync(), "https://cognitiveservices.azure.com/.default")

                client = AsyncAzureOpenAI(
                    azure_ad_token_provider=ad_token_provider, 
                    api_version="2024-08-01-preview",
                    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
                )

                if assistant_config is not None:
                    deployment = os.getenv("AZURE_OPENAI_MODEL")
                    instructions = "You are a helpful AI assistant who helps user find data from their tabular files like csv or excel"
                    if assistant_config.get("instructions") is not None:
                        instructions = assistant_config.get("instructions")
                        assistant = client.beta.assistants.create(
                                    instructions=instructions,
                                    model=deployment, # replace with model deployment name. 
                                    tools=[{"type": "code_interpreter"}]
                                    )
                        response_json['assistant_id'] = assistant.id
                else:
                    response_json['assistant_id'] = "asst_22Ato2XdpsixwE7h3HoReGVa"       
            else:
                response_json['assistant_id'] = "asst_22Ato2XdpsixwE7h3HoReGVa"
        elif app_mode.lower() == 'textual':
            assistant_id = None
            response_json['assistant_id'] = None

        #Create the response object and also set cookies
        response = await make_response(jsonify(response_json))
        response.set_cookie('APP_MODE',  f'{app_mode}', secure=True, max_age=60*60*6)
        if response_json['assistant_id'] is not None:  
            response.set_cookie('ASSISTANT_ID', str(response_json['assistant_id']), secure=True, max_age=60*60*6)
        #log event to eventhub
        send_log_event_to_eventhub(credential, "INFO", f"App Mode updated Successfully to {app_mode}, 200", "update_mode", index_container_to_use, current_user['user_name'], get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Success","User Action","Updating App mode")            
    
        return response, 200
    else:
        response = await make_response(jsonify({'error': f'please define the APP_MODE value'}))
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Please define the APP_MODE value, 400", "update_mode", index_container_to_use, current_user['user_name'], get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","User Action","Updating App mode")            
        return response, 400

@bp.route("/count-tokens", methods=["POST"])
async def counting_tokens():
    start_datetime = datetime.now()
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415
    full_response = await request.get_json()
    current_user = get_authenticated_user_details(request_headers=request.headers)  
    # Get containber obj i.e usecase, ad group mapping
    container_obj = get_container_obj(request,current_user)
    index_container_to_use = None 
    if request.cookies.get('index_container_to_use', None) is not None: 
        index_container_to_use = serializer.loads(request.cookies.get('index_container_to_use'))

    request_content = extract_content(full_response['messages'])
    response_content = extract_content(full_response['response_messages'])

    # Append 'tool_citations_content' from response_content to request_content  
    request_content['tool_citations_content'].extend(response_content['tool_citations_content'])  

    assistant_content_dict = {'assistant_content': response_content['assistant_content'],'tool_intent': response_content['tool_intent']}  
    try:

        completion_usage_tokens = num_tokens_from_messages(assistant_content_dict, AZURE_OPENAI_MODEL_NAME)
        prompt_usage_tokens = num_tokens_from_messages(request_content, AZURE_OPENAI_MODEL_NAME)
    
        # total_tokens = completion_usage_tokens + prompt_usage_tokens
        input_unit_cost , input_unit_measure = get_azure_openai_price("input",AZURE_OPENAI_MODEL)
        output_unit_cost, output_unit_measure = get_azure_openai_price("output",AZURE_OPENAI_MODEL)
        input_cost = input_unit_cost * prompt_usage_tokens / 1000
        output_cost = output_unit_cost * completion_usage_tokens / 1000
        total_cost = input_cost + output_cost
        # print(f"{prompt_usage_tokens} prompt tokens counted by num_tokens_from_messages().")
        # print(f"{completion_usage_tokens} completion tokens counted by num_tokens_from_messages().")
        # print(f"{total_tokens} total tokens counted by num_tokens_from_messages().")
        # print(f"{input_cost} USD is the input cost, {output_cost} USD is the output cost and {total_cost} USD is the total cost")

        usage_details ={
            "PromptTokens": prompt_usage_tokens,
            "PromptTokenUnitPrice": input_unit_cost,
            "PromptCost":input_cost,
            "CompletionTokens": completion_usage_tokens,
            "CompletionTokenUnitPrice": output_unit_cost,
            "CompletionCost":output_cost,
            "TotalCost": total_cost,
            "UnitOfMeasure": f"per {input_unit_measure} tokens",
            "CurrencyCode":"USD",
            "Model": AZURE_OPENAI_MODEL_NAME,
            "ModelName": AZURE_OPENAI_MODEL,
        }
        send_log_event_to_eventhub(credential, "INFO", "Consumption Calculation" , "counting_tokens", index_container_to_use,current_user['user_name'],get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Success","User Action","Consumption Calculation for an OpenAI Request and Response",api_usage= usage_details)

        return jsonify({"message": "success"}), 200
    except Exception as e:
        logging.exception("Exception in /count-tokens")
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /count-tokens :{e}, 500", "counting_tokens", index_container_to_use,current_user['user_name'],get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","User Action","Consumption Calculation for an OpenAI Request and Response")
        return jsonify({"error": str(e)}), 500 

def get_azure_openai_price(token_type, open_ai_model, region='swedencentral'):  
    # Map models and token types to meter names  
    meter_name_map = {  
        'gpt-4-omni-dev-webapp': {  
            'input': 'gpt-4o-0806-Inp-glbl Tokens',  
            'output': 'gpt-4o-0806-Outp-glbl Tokens'  
        },
        'gpt-4-omni-prod-webapp': {  
            'input': 'gpt-4o-0806-Inp-glbl Tokens',  
            'output': 'gpt-4o-0806-Outp-glbl Tokens'  
        },   
        'gpt-4-omni': {  
            'input': 'gpt 4o 0513 Input regional Tokens',  
            'output': 'gpt 4o 0513 Output regional Tokens'  
        },  
        'gpt-4-turbo-1106-preview': {  
            'input': 'gpt-4-turbo-128K Input-regional Tokens',  
            'output': 'gpt-4-turbo-128K Output-regional Tokens'  
        }  
    }  
  
    # Construct the meter name based on the model and token type  
    meter_name = meter_name_map.get(open_ai_model, {'input': 'gpt-4o-0806-Inp-glbl Tokens',  
            'output': 'gpt-4o-0806-Outp-glbl Tokens'}).get(token_type)  
    if not meter_name:  
        raise ValueError("Invalid model or token type provided.")  
  
    # # Define the API endpoint and parameters  
    # url = "https://prices.azure.com/api/retail/prices"  
    # params = {  
    #     "$filter": (  
    #         f"serviceName eq 'Cognitive Services' "  
    #         f"and productName eq 'Azure OpenAI' "  
    #         f"and armRegionName eq '{region}' "  
    #         f"and meterName eq '{meter_name}'"  
    #     )  
    # }  
  
    # # Make the request to the Azure Retail Prices API  
    # response = requests.get(url, params=params)  
    # if response.status_code == 200:  
    #     pricing_data = response.json()  
    #     # Assuming there's only one price for the meter name  
    #     if pricing_data['Items']:  
    #         unit_price = pricing_data['Items'][0]['unitPrice']
    #         unit_measure = pricing_data['Items'][0]['unitOfMeasure'] 
    #         return unit_price, unit_measure
    #     else:  
    #         raise ValueError("No pricing data found for the specified parameters.")  
    # else:  
    #     raise ConnectionError(f"Failed to retrieve pricing data: {response.status_code}")  
     
    # Filter the pricing data based on region and meter name  
    matching_items = [  
        item for item in pricing_data['Items']  
        if item['armRegionName'].lower() == region.lower() and item['meterName'] == meter_name  
    ]  
  
    # Assuming there's only one price for the meter name  
    if matching_items:  
        unit_price = matching_items[0]['unitPrice']  
        unit_measure = matching_items[0]['unitOfMeasure']  
        return unit_price, unit_measure  
    else:  
        raise ValueError("No pricing data found for the specified parameters.")

@bp.route("/conversation-test", methods=["POST"])
async def conversation_test():
    start_datetime = datetime.now()
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415
    request_json = await request.get_json()
    
    current_user = get_authenticated_user_details(request_headers=request.headers)    
    print(current_user['user_name'])
    return await conversation_internal(request_json, current_user['user_name'], request.cookies,current_user['user_name'],None,start_datetime)

@bp.route("/frontend_settings", methods=["GET"])  
def get_frontend_settings():
    start_datetime = datetime.now()
    container_obj ={}
    current_user = None
    index_container_to_use = None
    try:
        current_user = get_authenticated_user_details(request_headers=request.headers)  
        # Get containber obj i.e usecase, ad group mapping
        container_obj = get_container_obj(request,current_user) 
        if request.cookies.get('index_container_to_use', None) is not None:
            index_container_to_use = serializer.loads(request.cookies.get('index_container_to_use'))

        return jsonify(frontend_settings), 200
    except Exception as e:
        logging.exception("Exception in /frontend_settings")
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /frontend_settings :{e},500", "get_frontend_settings", index_container_to_use, current_user['user_name'], get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","App Action","Fetch frontend settings like applogo,auth status while loading the app")
        return jsonify({"error": str(e)}), 500  

## Conversation History API ## 
@bp.route("/history/generate", methods=["POST"])
async def add_conversation():
    start_datetime = datetime.now()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']
    user_email = authenticated_user['user_name']
    container_obj={}
    index_container_to_use = None
    title = None
    app_mode = None
    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get('conversation_id', None)
    if request.cookies.get('index_container_to_use', None) is not None:
        index_container_to_use = serializer.loads(request.cookies.get('index_container_to_use'))
    
    app_mode = request.cookies.get('APP_MODE', 'Textual')   
    # make sure Sql is configured
    sql_conversation_client = init_sql_client()
    try:
        # Get containber obj i.e usecase, ad group mapping
        container_obj = get_container_obj(request,authenticated_user)
        
        if not sql_conversation_client:
            raise Exception("SqlDB is not configured or not working")

        # check for the conversation_id, if the conversation is not set, we will create a new one
        history_metadata = {}
        if not conversation_id:
            title = await generate_title(request_json["messages"],index_container_to_use,user_email,container_obj)
            conversation_dict = sql_conversation_client.create_conversation(user_id=user_id, title=title, user_email=user_email)
            conversation_id = conversation_dict['id']
            history_metadata['title'] = title
            history_metadata['date'] = conversation_dict['createdAt']
            
        ## Format the incoming message object in the "chat/completions" messages format
        ## then write it to the conversation history in Sql
        messages = request_json["messages"]
        if len(messages) > 0 and messages[-1]['role'] == "user":
            createdMessageValue = sql_conversation_client.create_message(
                uuid=str(uuid.uuid4()),
                conversation_id=conversation_id,
                user_id=user_id,
                user_email = user_email,
                input_message=messages[-1]
            )
            
            if createdMessageValue == "Conversation not found":
                raise Exception("Conversation not found for the given conversation ID: " + conversation_id + ".")
        else:
            raise Exception("No user message found")
        
        
        
        # Submit request to Chat Completions for response
        request_body = await request.get_json()
        history_metadata['conversation_id'] = conversation_id
        request_body['history_metadata'] = history_metadata
        if app_mode.lower() == "textual":
            return await conversation_internal(request_body, request.cookies,user_email,container_obj,start_datetime)
        else:
            if len(messages) > 0 and messages[-1]['role'] == "user":
                request_body["messages"] = messages[-1]
            return await create_message_internal(request_body, request.cookies,user_email,container_obj,start_datetime)
     
    except Exception as e:
        logging.exception("Exception in /history/generate")
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /history/generate for the conversation '{title}' with id {conversation_id}:{e}, 500", "add_conversation", index_container_to_use,user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","User Action","Conversation/Prompting when chat history is enabled")
        return jsonify({"error": str(e)}), 500
    finally:
        sql_conversation_client.close()

@bp.route("/history/update", methods=["POST"])
async def update_conversation():
    start_datetime = datetime.now()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']
    user_email = authenticated_user['user_name']
    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get('conversation_id', None)
    container_obj ={}
    title = request_json.get('conversation_title', None)
    index_container_to_use = None
    # make sure Sql is configured
    sql_conversation_client = init_sql_client()
    try:
        # Get containber obj i.e usecase, ad group mapping
        container_obj = get_container_obj(request,authenticated_user)
        if request.cookies.get('index_container_to_use', None) is not None:  
            index_container_to_use = serializer.loads(request.cookies.get('index_container_to_use'))
        
        if not sql_conversation_client:
            raise Exception("Sql DB is not configured or not working")

        # check for the conversation_id, if the conversation is not set, we will create a new one
        if not conversation_id:
            raise Exception("No conversation_id found")
            
        ## Format the incoming message object in the "chat/completions" messages format
        ## then write it to the conversation history in Sql DB
        messages = request_json["messages"]
        if len(messages) > 0 and messages[-1]['role'] == "assistant":
            if len(messages) > 1 and messages[-2].get('role', None) == "tool":
                # write the tool message first
                sql_conversation_client.create_message(
                    uuid=str(uuid.uuid4()),
                    conversation_id=conversation_id,
                    user_id=user_id,
                    user_email = user_email,
                    input_message=messages[-2]
                )
            # write the assistant message
            sql_conversation_client.create_message(
                uuid=messages[-1]['id'],
                conversation_id=conversation_id,
                user_id=user_id,
                user_email = user_email,
                input_message=messages[-1]
            )
        else:
            raise Exception("No bot messages found")
        
        # Submit request to Chat Completions for response
        sql_conversation_client.close()
        response = {'success': True}
        #log event to eventhub
        send_log_event_to_eventhub(credential, "INFO", f"Conversation and messages updated successfully for the conversation '{title}' and conversation_id: {conversation_id}", "update_conversation", index_container_to_use, user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Success","User Action","Updating Conversation when chat history is enabled")
        
        return jsonify(response), 200
       
    except Exception as e:
        logging.exception("Exception in /history/update")
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /history/update for the conversation '{title}' with id: {conversation_id} :{e}", "update_conversation", index_container_to_use, user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","User Action","Updating Conversation when chat history is enabled")
        return jsonify({"error": str(e)}), 500
    finally:
        sql_conversation_client.close()

@bp.route("/history/message_feedback", methods=["POST"])
async def update_message():
    start_datetime = datetime.now()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']
    user_email = authenticated_user['user_name']
    sql_conversation_client = init_sql_client()
    container_obj ={}
    index_container_to_use = None
    ## check request for message_id
    request_json = await request.get_json()
    message_id = request_json.get('message_id', None)
    message_feedback = request_json.get("message_feedback", None)
    feedback_description = request_json.get("feedback_description", "") 
    chat_history_status = request.cookies.get('CHAT_HISTORY_ENABLED','false')
    try:
        # Get containber obj i.e usecase, ad group mapping
        container_obj = get_container_obj(request,authenticated_user)
        if request.cookies.get('index_container_to_use', None) is not None:  
            index_container_to_use = serializer.loads(request.cookies.get('index_container_to_use'))
        
        feedback ={
                "Feedback": message_feedback,
                "FeedbackDescription": feedback_description
        }
        if chat_history_status.lower() == 'true':    
            if not message_id:
                return jsonify({"error": "message_id is required"}), 400
            
            if not message_feedback:
                return jsonify({"error": "message_feedback is required"}), 400
            
            ## update the message in db
            updated_message = sql_conversation_client.update_message_feedback(message_id, user_id, message_feedback, feedback_description)
            if updated_message:
                
                send_log_event_to_eventhub(credential, "INFO", f"Message Feedback successfully updated", "update_message",index_container_to_use, user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Success","User Action","Giving message feedback", feedback=feedback)
                return jsonify({"message": f"Successfully updated message with feedback {message_feedback}", "message_id": message_id}), 200

            else:
                send_log_event_to_eventhub(credential, "ERROR", f"Unable to update message {message_id}. It either does not exist or the user does not have access to it.,404", "update_message",index_container_to_use, user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","User Action","Giving message feedback")

                return jsonify({"error": f"Unable to update message {message_id}. It either does not exist or the user does not have access to it."}), 404
        else :  
            print("chat disabled") 
            if not message_id:
                return jsonify({"error": "message_id is required"}), 400         
            if not message_feedback:
                return jsonify({"error": "message_feedback is required"}), 400

            send_log_event_to_eventhub(credential, "INFO", f"Message Feedback successfully updated", "update_message",index_container_to_use, user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Success","User Action","Giving message feedback", feedback=feedback)
            return jsonify({"message": f"Successfully updated message with feedback {message_feedback}", "message_id": message_id}), 200
    except Exception as e:
        logging.exception("Exception in /history/message_feedback")
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /history/message_feedback : {e},500", "update_message",index_container_to_use, user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","User Action","Giving message feedback")
        return jsonify({"error": str(e)}), 500
    finally:
        sql_conversation_client.close()

@bp.route("/history/delete", methods=["DELETE"])
async def delete_conversation():
    start_datetime = datetime.now()
    ## get the user id from the request headers
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']
    user_email = authenticated_user['user_name']
    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get('conversation_id', None)
    container_obj ={}
    title = request_json.get('conversation_title', None)
    index_container_to_use = None
    ## make sure DB is configured
    sql_conversation_client = init_sql_client()
    try:
        # Get containber obj i.e usecase, ad group mapping
        container_obj = get_container_obj(request,authenticated_user)
        if request.cookies.get('index_container_to_use', None) is not None:  
            index_container_to_use = serializer.loads(request.cookies.get('index_container_to_use')) 
        if not conversation_id:
            return jsonify({"error": "conversation_id is required"}), 400
    
        if not sql_conversation_client:
            raise Exception("Sql DB is not configured or not working")
        ## delete the conversation messages from db first
        deleted_messages = sql_conversation_client.delete_messages(conversation_id, user_id)

        ## Now delete the conversation 
        deleted_conversation = sql_conversation_client.delete_conversation(conversation_id ,user_id)

        if deleted_conversation:
            #log event to eventhub
            send_log_event_to_eventhub(credential, "INFO", f"Successfully deleted conversation and messages with the title '{title}' and conversation_id: {conversation_id}", "delete_conversation", index_container_to_use,user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Success","User Action","Deleting the conversation from the chat history for the user")
            return jsonify({"message": "Successfully deleted conversation and messages", "conversation_id": conversation_id}), 200
        else :
            raise Exception("Cannot delete conversation")    
    except Exception as e:
        logging.exception("Exception in /history/delete")
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /history/delete for '{title}' with id: {conversation_id}:{e},500", "delete_conversation", index_container_to_use,user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","User Action","Deleting the conversation from the chat history for the user")
        return jsonify({"error": str(e)}), 500
    finally:
        sql_conversation_client.close()

@bp.route("/history/list", methods=["GET"])
async def list_conversations():
    start_datetime = datetime.now()
    offset = request.args.get("offset", 0)
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']
    user_email = authenticated_user['user_name']
    container_obj ={}
    index_container_to_use = None
    ## make sure DB is configured
    sql_conversation_client = init_sql_client()
    try:
        # Get containber obj i.e usecase, ad group mapping
        container_obj = get_container_obj(request,authenticated_user)
        if request.cookies.get('index_container_to_use', None) is not None: 
            index_container_to_use = serializer.loads(request.cookies.get('index_container_to_use'))
        
        if not sql_conversation_client:
            raise Exception("SqlDb is not configured or not working")

        ## get the conversations from db
        conversations = sql_conversation_client.get_conversations(user_id, offset=offset, limit=25)

        if not isinstance(conversations, list):
            #log event to eventhub
            send_log_event_to_eventhub(credential, "ERROR", f"No conversations for {user_id} were found,404", "list_conversations", index_container_to_use, user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","App Action","Fetching the list of chat history for the user")
            return jsonify({"error": f"No conversations for {user_id} were found"}), 404
        
        ## return the conversation ids
        #log event to eventhub
        send_log_event_to_eventhub(credential, "INFO", f"List of all the chat history sent successfully for the User", "list_conversations", index_container_to_use, user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Success","App Action","Fetching the list of chat history for the user")
        
        return jsonify(conversations), 200
    except Exception as e:
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /history/list :{e},500", "list_conversations", index_container_to_use, user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","App Action","Fetching the list of chat history for the user")
        return jsonify({"error": str(e)}), 500
    finally:
        sql_conversation_client.close()

@bp.route("/history/read", methods=["POST"])
async def get_conversation():
    start_datetime = datetime.now()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']
    user_email = authenticated_user['user_name']

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get('conversation_id', None)
    title = request_json.get('conversation_title', None)
    if not conversation_id:
        return jsonify({"error": "conversation_id is required"}), 400
    container_obj ={}
    index_container_to_use = None
    ## make sure db is configured
    sql_conversation_client = init_sql_client()
    try:
        # Get containber obj i.e usecase, ad group mapping
        container_obj = get_container_obj(request,authenticated_user)
        if request.cookies.get('index_container_to_use', None) is not None: 
            index_container_to_use = serializer.loads(request.cookies.get('index_container_to_use'))
        
        if not sql_conversation_client:
            raise Exception("Sql Db is not configured or not working")

        ## get the conversation object and the related messages from sql
        conversation = sql_conversation_client.get_conversation(conversation_id=conversation_id, user_id=user_id)
        ## return the conversation id and the messages in the bot frontend format
        if not conversation:
            return jsonify({"error": f"Conversation {conversation_id} was not found. It either does not exist or the logged in user does not have access to it."}), 404
        
        # get the messages for the conversation from sql
        conversation_messages = sql_conversation_client.get_messages(conversation_id, user_id)
        
        ## format the messages in the bot frontend format
        messages = [  
        {  
            'id': msg.id,  
            'role': msg.role,  
            'content': msg.content,  
            'createdAt': msg.createdAt,  
            'feedback': getattr(msg, 'feedback', None)  # Assuming 'feedback' is an optional attribute  
        }  
        for msg in conversation_messages  # conversation_messages should be the result of get_messages  
        ]  
        return jsonify({"conversation_id": conversation_id, "messages": messages}), 200
    except Exception as e:
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /history/read for the conversation '{title}' with id: {conversation_id}:{e},500", "get_conversation", index_container_to_use,user_email, get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","App Action","Fetching the chat history conversations messages for the user")
        return jsonify({"error":str(e)}),500
    finally:
        sql_conversation_client.close()

@bp.route("/history/rename", methods=["POST"])
async def rename_conversation():
    start_datetime = datetime.now()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']
    user_email = authenticated_user['user_name']
    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get('conversation_id', None)
    old_title = None
    if not conversation_id:
        return jsonify({"error": "conversation_id is required"}), 400
    container_obj ={}
    index_container_to_use = None
    ## make sure db is configured
    sql_conversation_client = init_sql_client()
    try:
        # Get containber obj i.e usecase, ad group mapping
        container_obj = get_container_obj(request,authenticated_user)
        if request.cookies.get('index_container_to_use', None) is not None:  
            index_container_to_use = serializer.loads(request.cookies.get('index_container_to_use'))
        
        if not sql_conversation_client:
            raise Exception("Sql DB is not configured or not working")
        
        ## get the conversation from db
        conversation = sql_conversation_client.get_conversation(user_id=user_id, conversation_id=conversation_id)
        
        if not conversation:
            return jsonify({"error": f"Conversation {conversation_id} was not found. It either does not exist or the logged in user does not have access to it."}), 404
        old_title = conversation.title
        ## update the title
        title = request_json.get("title", None)
        if not title:
            return jsonify({"error": "title is required"}), 400
        conversation.title = title
        updated_conversation = sql_conversation_client.upsert_conversation(conversation)

        
        #log event to eventhub
        send_log_event_to_eventhub(credential, "INFO", f"Chat History title renamed successfully from :{old_title} to :{title}", "rename_conversation", index_container_to_use,user_email, get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Success","User Action","Renaming the title of the chat history")
        
        return jsonify(updated_conversation), 200
    except Exception as e:
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /history/rename for the conversation '{old_title}' with id {conversation_id} :{e},500", "rename_conversation", index_container_to_use,user_email, get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","User Action","Renaming the title of the chat history")
        return jsonify({"error":str(e)}),500
    finally:
        sql_conversation_client.close() 

@bp.route("/history/delete_all", methods=["DELETE"])
async def delete_all_conversations():
    start_datetime = datetime.now()
    ## get the user id from the request headers
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']
    user_email = authenticated_user['user_name']
    # get conversations for user
    container_obj ={}
    index_container_to_use = None
    ## make sure sql is configured
    sql_conversation_client = init_sql_client()
    try:
        # Get containber obj i.e usecase, ad group mapping
        container_obj = get_container_obj(request,authenticated_user)
        if request.cookies.get('index_container_to_use', None) is not None:  
            index_container_to_use = serializer.loads(request.cookies.get('index_container_to_use'))
        
        if not sql_conversation_client:
            raise Exception("Sql DB is not configured or not working")

        conversations = sql_conversation_client.get_conversations(user_id, offset=0, limit=None)
        if not conversations:
            #log event to eventhub
            send_log_event_to_eventhub(credential, "ERROR", f"No conversations for {user_id} were found,404", "delete_all_conversations", index_container_to_use,user_email, get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed")
        
            return jsonify({"error": f"No conversations for {user_id} were found"}), 404
        
        # delete each conversation
        for conversation in conversations:
            ## delete the conversation messages from db first
            deleted_messages = sql_conversation_client.delete_messages(conversation['id'], user_id)

            ## Now delete the conversation 
            deleted_conversation = sql_conversation_client.delete_conversation(conversation['id'], user_id)
        
        
        #log event to eventhub
        send_log_event_to_eventhub(credential, "INFO", f"Successfully deleted all the conversations and messages for user {user_email}", "delete_all_conversations", index_container_to_use,user_email, get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Success","User Action","Deleting the entire chat history for the user")
        return jsonify({"message": f"Successfully deleted all the conversation and messages for user {user_id}"}), 200
    
    except Exception as e:
        logging.exception("Exception in /history/delete_all")
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /history/delete_all :{e},500", "delete_all_conversations", index_container_to_use,user_email, get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","User Action","Deleting the entire chat history for the user")
        return jsonify({"error": str(e)}), 500
    finally:
        sql_conversation_client.close()

@bp.route("/history/clear", methods=["POST"])
async def clear_messages():
    start_datetime = datetime.now()
    ## get the user id from the request headers
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']
    user_email = authenticated_user['user_name']
    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get('conversation_id', None)

    container_obj ={}
    index_container_to_use = None
    try:
        # Get containber obj i.e usecase, ad group mapping
        container_obj = get_container_obj(request,authenticated_user)
        if request.cookies.get('index_container_to_use', None) is not None:  
            index_container_to_use = serializer.loads(request.cookies.get('index_container_to_use')) 
        if not conversation_id:
            return jsonify({"error": "conversation_id is required"}), 400
        
        ## make sure db is configured
        sql_conversation_client = init_sql_client()
        if not sql_conversation_client:
            raise Exception("Sql DB is not configured or not working")

        ## delete the conversation messages from sql
        deleted_messages = sql_conversation_client.delete_messages(conversation_id, user_id)

        return jsonify({"message": "Successfully deleted messages in conversation", "conversation_id": conversation_id}), 200
    except Exception as e:
        logging.exception("Exception in /history/clear_messages")
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /history/clear_messages :{e},500", "clear_messages", index_container_to_use,user_email, get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","User Action","Clearing the chat history messages for the conversation")
        return jsonify({"error": str(e)}), 500

@bp.route("/history/ensure", methods=["GET"])  
async def ensure_sql():  
    start_datetime = datetime.now()
    # Check if the necessary configuration for Azure SQL Server is provided  
    current_user = get_authenticated_user_details(request_headers=request.headers)
    token = credential.get_token("https://database.windows.net/.default").token 
    CHAT_HISTORY_ENABLED = request.cookies.get('CHAT_HISTORY_ENABLED', 'false')  
    # print(current_user['user_name'])
    index_container_to_use = None
    # Get containber obj i.e usecase, ad group mapping
    container_obj = get_container_obj(request,current_user) 
    if request.cookies.get('index_container_to_use', None) is not None:
        index_container_to_use = serializer.loads(request.cookies.get('index_container_to_use')) 
    

    if not (AZURE_SQL_SERVER and AZURE_SQL_DATABASE):  
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Azure SQL Server is not configured,404", "history_ensure", index_container_to_use,current_user['user_name'],get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","App Action","Ensuring the chat history status")
        return jsonify({"error": "Azure SQL Server is not configured"}), 404  

    try:  
        connection_string = (  
                    f"Driver={{ODBC Driver 17 for SQL Server}};"  
                    f"Server=tcp:{AZURE_SQL_SERVER},1433;"  
                    f"Database={AZURE_SQL_DATABASE};"  
                    f"Uid={AZURE_SQL_USER};"    
                    f"Pwd={AZURE_SQL_PASSWORD};"  
                    f"Encrypt=yes;"  
                    f"TrustServerCertificate=no;"  
                    f"Connection Timeout=30;"  
                    # f"Authentication=ActiveDirectoryIntegrated;"   
                )  
        # Initialize SQL client  
        sql_connection = pyodbc.connect(connection_string) 
         
        # Execute a simple query to check if the connection is alive  
        with sql_connection.cursor() as cursor:  
            cursor.execute("SELECT 1")  
            result = cursor.fetchone()  
            if not result:  
                #log event to eventhub
                send_log_event_to_eventhub(credential, "ERROR", f"Azure SQL Server is not working,400", "history_ensure", index_container_to_use,current_user['user_name'],get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","App Action","Ensuring the chat history status")
                return jsonify({"error": "Azure SQL Server is not working"}), 400  
          
        # Close the SQL connection  
        sql_connection.close()
        if CHAT_HISTORY_ENABLED.lower() == 'false': 
            #log event to eventhub
            send_log_event_to_eventhub(credential, "INFO", f"Azure SQL Server is configured and working and Chat History is Disabled", "history_ensure", index_container_to_use,current_user['user_name'],get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Success","App Action","Ensuring the chat history status")
         
            return jsonify({"message": "Azure SQL Server is configured and working and Chat History is Disabled"}), 200
        #log event to eventhub
        send_log_event_to_eventhub(credential, "INFO", f"Azure SQL Server is configured and working and Chat History is Enabled", "history_ensure", index_container_to_use,current_user['user_name'],get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Success","App Action","Ensuring the chat history status")
        
        return jsonify({"message": "Azure SQL Server is configured and working and Chat History is Enabled"}), 200     
    except pyodbc.Error as e:  
        logging.exception("Exception in /history/ensure")
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /history/ensure :{e},500", "history_ensure", index_container_to_use,current_user['user_name'],get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","App Action","Ensuring the chat history status")
        sql_exception = str(e)  
        if "Login failed for user" in sql_exception:  
            return jsonify({"error": sql_exception}), 401  
        else:  
            return jsonify({"error": "Azure SQL Server is not working"}), 500  

#Prompt Catalog APIs
@bp.route("/prompts/create", methods=["POST"])  
async def create_prompt():  
    start_datetime = datetime.now()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']
    user_email = authenticated_user['user_name']
    container_obj={}
    index_container_to_use = None
    title = None
    # Parse request data  
    request_json = await request.get_json()  
    title = request_json.get('title')  
    description = request_json.get('description')  
    tags = request_json.get('tags', [])  # Expect tags to be a list of tag names  
    # Check for required fields  
    if not title or not description:  
        error_message = "Title and description are required fields."  
        logging.error(error_message) 
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /prompts/create for the '{title}':{error_message}, 400", "create_prompt", index_container_to_use,user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","User Action","Creating a new Prompt")
         
        return jsonify({"error": error_message}), 400 
    if request.cookies.get('index_container_to_use', None) is not None:
        index_container_to_use = serializer.loads(request.cookies.get('index_container_to_use'))
  
    # Initialize SQL client  
    sql_client = init_sql_client()  
    try:  
        # Create the prompt  
        prompt_dict = sql_client.create_prompt(  
            title=title,  
            description=description,  
            user_id=user_id,  
            user_email=user_email,  
            tag_names=tags  
        )    
        #log event to eventhub
        send_log_event_to_eventhub(credential, "INFO", f"Prompt '{title}' created successfully", "create_prompt", index_container_to_use,user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Success","User Action","Creating a new Prompt")  
        
        return jsonify(prompt_dict), 201  
    except Exception as e:  
        logging.exception("Exception in /prompts/create")  
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /prompts/create for the '{title}':{e}, 500", "create_prompt", index_container_to_use,user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","User Action","Creating a new Prompt")  
        return jsonify({"error": str(e)}), 500  
    finally:  
        # Ensure that the SQL client is closed after the operation  
        sql_client.close()  

@bp.route("/prompts/update", methods=["POST"])  
async def update_prompt():  
    start_datetime = datetime.now()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']
    user_email = authenticated_user['user_name']
    container_obj={}
    index_container_to_use = None
    title = None
    # Parse request data  
    request_json = await request.get_json()  
    prompt_id = request_json.get('prompt_id')  
    title = request_json.get('title')  
    description = request_json.get('description')  
    tags = request_json.get('tags', None)  # New tags  
    # Check for required fields  
    if not title or not description:  
        error_message = "Title and description are required fields."  
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /prompts/update for the '{title}':{error_message}, 400", "update_prompt", index_container_to_use,user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","User Action","Updating a Prompt")
         
        return jsonify({"error": error_message}), 400  
    
    if request.cookies.get('index_container_to_use', None) is not None:
        index_container_to_use = serializer.loads(request.cookies.get('index_container_to_use'))
  
    # Initialize SQL client  
    sql_client = init_sql_client()  
    try:  
        # Create the prompt  
        prompt_dict = sql_client.update_prompt(  
        prompt_id=prompt_id,  
        title=title,  
        description=description,  
        tag_names=tags  
         )    
        #log event to eventhub
        send_log_event_to_eventhub(credential, "INFO", f"Prompt '{title}' updated successfully", "update_prompt", index_container_to_use,user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Success","User Action","Updating a Prompt")  
        
        return jsonify(prompt_dict), 200  
    except Exception as e:    
       #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /prompts/update for the '{title}':{e}, 500", "update_prompt", index_container_to_use,user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","User Action","Updating a Prompt")
        return jsonify({"error": str(e)}), 500  
    finally:  
        # Ensure that the SQL client is closed after the operation  
        sql_client.close()  

@bp.route("/prompts/delete", methods=["DELETE"])  
async def delete_prompt():  
    start_datetime = datetime.now()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']
    user_email = authenticated_user['user_name']
    container_obj={}
    index_container_to_use = None
    title = None
    # Parse request data  
    request_json = await request.get_json()  
    prompt_id = request_json.get('prompt_id')  
    title = request_json.get('title')

    if request.cookies.get('index_container_to_use', None) is not None:
        index_container_to_use = serializer.loads(request.cookies.get('index_container_to_use'))
  
    # Initialize SQL client  
    sql_client = init_sql_client()  
    try:  
        # Create the prompt  
        success = sql_client.delete_prompt(prompt_id=prompt_id)  
                
        if success:
            #log event to eventhub
            send_log_event_to_eventhub(credential, "INFO", f"Prompt '{title}' deleted successfully", "delete_prompt", index_container_to_use,user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Success","User Action","Deleting a Prompt")
            return jsonify({"success": success}), 200
        else:
            #log event to eventhub
            send_log_event_to_eventhub(credential, "ERROR", f"Prompt '{title}' not found", "delete_prompt", index_container_to_use,user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","User Action","Deleting a Prompt")  
            return jsonify({"error": f"Prompt '{title}' not found"}), 404  

    except Exception as e:    
       #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /prompts/delete for the '{title}':{error_message}, 500", "delete_prompt", index_container_to_use,user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","User Action","Deleting a Prompt")
        return jsonify({"error": str(e)}), 500  
    finally:  
        # Ensure that the SQL client is closed after the operation  
        sql_client.close()  

@bp.route("/prompts/delete_all", methods=["DELETE"])  
async def delete_all_prompts():  
    start_datetime = datetime.now()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']
    user_email = authenticated_user['user_name']
    container_obj={}
    index_container_to_use = None
    title = None
    # Parse request data  
    request_json = await request.get_json()  
    prompt_id = request_json.get('prompt_id')  
    title = request_json.get('title')

    if request.cookies.get('index_container_to_use', None) is not None:
        index_container_to_use = serializer.loads(request.cookies.get('index_container_to_use'))
  
    # Initialize SQL client  
    sql_client = init_sql_client()  
    try: 
        
        success = sql_client.get_prompts_by_user(user_id, offset=0, limit=None)
        if not success:
            #log event to eventhub
            send_log_event_to_eventhub(credential, "ERROR", f"Prompts not found", "delete_all_prompts", index_container_to_use,user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","User Action","Deleting all the Prompts")  
            return jsonify({"error": f"Prompts not found"}), 404

        # delete each prompt
        for s in success:
            ## Now delete the prompt 
            deleted_prompt = sql_client.delete_prompt(prompt_id=s['prompt_id']) 
         
            
        #log event to eventhub
        send_log_event_to_eventhub(credential, "INFO", f"Prompts deleted successfully", "delete_all_prompts", index_container_to_use,user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Success","User Action","Deleting all the Prompts")
        return jsonify({"success": success}), 200 

    except Exception as e:    
       #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /prompts/delete for the '{title}':{e}, 500", "delete_all_prompts", index_container_to_use,user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","User Action","Deleting all the Prompts")
        return jsonify({"error": str(e)}), 500  
    finally:  
        # Ensure that the SQL client is closed after the operation  
        sql_client.close()  

@bp.route("/prompts/list", methods=["GET"])  
async def list_prompts():  
    start_datetime = datetime.now()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']
    user_email = authenticated_user['user_name']
    container_obj={}
    index_container_to_use = None
    #parse the request  
    limit = request.args.get("limit", 10)  
    offset = request.args.get("offset", 0)  
    sort_order = request.args.get("sort_order", "DESC")  
    if request.cookies.get('index_container_to_use', None) is not None:
        index_container_to_use = serializer.loads(request.cookies.get('index_container_to_use'))
  
    # Initialize SQL client  
    sql_client = init_sql_client()  
    try:   
        prompts_list = sql_client.get_prompts_by_user(  
            user_id=user_id,  
            limit=limit,  
            offset=offset,  
            sort_order=sort_order  
        )  
        if prompts_list:
            #log event to eventhub
            send_log_event_to_eventhub(credential, "INFO", f"Prompts fetched successfully", "list_prompt", index_container_to_use,user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Success","App Action","Fetching the list of Prompts")
            return jsonify(prompts_list), 200
        else:
            #log event to eventhub
            send_log_event_to_eventhub(credential, "ERROR", f"Prompts not found, 404", "list_prompt", index_container_to_use,user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","App Action","Fetching the list of Prompts")  
            return jsonify({"error": f"Prompts not found"}), 404  

    except Exception as e:    
       #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /prompts/list :{error_message}, 500", "list_prompt", index_container_to_use,user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","App Action","Fetching the list of Prompts")
        return jsonify({"error": str(e)}), 500  
    finally:  
        # Ensure that the SQL client is closed after the operation  
        sql_client.close() 

@bp.route("/prompts/tags/list", methods=["GET"])  
async def list_tags():  
    start_datetime = datetime.now()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']
    user_email = authenticated_user['user_name']
    container_obj={}
    index_container_to_use = None
      
    if request.cookies.get('index_container_to_use', None) is not None:
        index_container_to_use = serializer.loads(request.cookies.get('index_container_to_use'))
  
    # Initialize SQL client  
    sql_client = init_sql_client()  
    try:   
        tag_list = sql_client.get_tags()  
        if tag_list:
            #log event to eventhub
            send_log_event_to_eventhub(credential, "INFO", f"Tags fetched successfully", "list_tags", index_container_to_use,user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Success","App Action","Fetching the list of tags")
            return jsonify(tag_list), 200
        else:
            #log event to eventhub
            send_log_event_to_eventhub(credential, "ERROR", f"Tags not found, 404", "list_tags", index_container_to_use,user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","App Action","Fetching the list of tags")  
            return jsonify({"error": f"Prompts not found"}), 404  

    except Exception as e:    
       #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /prompts/tags/list :{error_message}, 500", "list_tags", index_container_to_use,user_email,get_adgroup_by_usecase(container_obj,index_container_to_use),start_datetime,"Failed","App Action","Fetching the list of tags")
        return jsonify({"error": str(e)}), 500  
    finally:  
        # Ensure that the SQL client is closed after the operation  
        sql_client.close() 
          
@bp.route('/get-use-cases', methods=['GET'])  
async def list_usecases():
    start_datetime = datetime.now()  
    current_user = get_authenticated_user_details(request_headers=request.headers) 
    container_obj = {} 
    container_to_use = None 
    try:
        # Safely get and parse USECASE_DESC
        try:
            if isinstance(USECASE_DESC, str):
                # If it's a string, try to parse it as JSON
                usecase_descriptions = json.loads(USECASE_DESC)
            elif isinstance(USECASE_DESC, dict):
                # If it's already a dict, use it directly
                usecase_descriptions = USECASE_DESC
            else:
                # Fallback to default descriptions
                usecase_descriptions = {
                    "rnd": "Chat with R&D SOPs", 
                    "cdc": "Chat with CDC Data",
                    "gsc": "Chat with GSC SOPs", 
                    "us-contracts": "US Contracting & Formulary Compliance", 
                    "dset": "Chat with DSET Demo Data", 
                    "im-portfolio": "IM Portfolio SWOT Analysis", 
                    "im-sales": "IM Sales Calls Analysis", 
                    "gia": "Search & Query GIA Audit Content"
                }
        except (json.JSONDecodeError, TypeError):
            # If parsing fails, use default descriptions
            usecase_descriptions = {
                "rnd": "Chat with R&D SOPs", 
                "cdc": "Chat with CDC Data",
                "gsc": "Chat with GSC SOPs", 
                "us-contracts": "US Contracting & Formulary Compliance", 
                "dset": "Chat with DSET Demo Data", 
                "im-portfolio": "IM Portfolio SWOT Analysis", 
                "im-sales": "IM Sales Calls Analysis", 
                "gia": "Search & Query GIA Audit Content"
            }
        
        # Get container obj i.e usecase, ad group mapping
        container_obj = get_container_obj(request, current_user)
        container_names = list(container_obj.values())

        logging.debug("Mapped Index: " + str(container_names))
        
        if len(container_names) > 0:
            usecase_dict = {}

            for container in container_names:
                usecase_dict[container] = usecase_descriptions.get(container, container)
            
            response = await make_response(jsonify(usecase_dict))
            
            if request.cookies.get('index_container_to_use', None) is not None:
                container_to_use = serializer.loads(request.cookies.get('index_container_to_use'))
                response = make_cookie_for_model(response, container_to_use)
            else:
                container_to_use = container_names[0]
                response = make_cookie_for_model(response, container_names[0])
            
            #log event to eventhub
            send_log_event_to_eventhub(credential, "INFO", f"Response :{usecase_dict}", "list_usecases", container_to_use, current_user['user_name'], get_adgroup_by_usecase(container_obj, container_to_use), start_datetime, "Success", "App Action", "Fetch the list of all the usecases for the user")
            return response, 200  
        else:
            return await make_response(jsonify({})), 200  
    except Exception as e:  
        logging.debug(e)
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /get-use-cases :{e}, 500", "list_usecases", container_to_use, current_user['user_name'], get_adgroup_by_usecase(container_obj, container_to_use), start_datetime, "Failed", "App Action", "Fetch the list of all the usecases for the user")
        return await make_response(jsonify({"error": str(e)})), 500

@bp.route('/use-case-desc', methods=['GET'])
async def get_usecase_desc():
    start_datetime = datetime.now()
    current_user = get_authenticated_user_details(request_headers=request.headers)
    container_obj = {}
    container_to_use = None 
    try:
        # Get containber obj i.e usecase, ad group mapping
        container_obj = get_container_obj(request,current_user)
        if request.cookies.get('index_container_to_use', None) is not None:
            container_to_use = serializer.loads(request.cookies.get('index_container_to_use'))
            
            title_to_use = USECASE_DESC.get(container_to_use, None)
            
            if title_to_use:
                return jsonify({"title": str(title_to_use)}), 200
        else:
            
            container_names = list(container_obj.values())
            container_to_use = container_names[0]

            title_to_use = USECASE_DESC.get(container_to_use, None)
            if title_to_use:
                return jsonify({"title": str(title_to_use)}), 200
            return jsonify({"error":"No AD Group Mapping Found"}), 200
    except Exception as e:
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /use-case-desc :{e}", "get_usecase_desc", container_to_use, current_user['user_name'],get_adgroup_by_usecase(container_obj,container_to_use),start_datetime,"Failed","App Action","Fetch the description for usecase")  
        return jsonify({"error": str(e)}), 500  

@bp.route('/delete-blob-file', methods=['DELETE']) 
async def delete_blob_file():
    start_datetime = datetime.now()
    data = await request.json 
    file_name = data.get('fileName')  
    current_user = get_authenticated_user_details(request_headers=request.headers) 
    container_obj = {}
    container_to_use = None 
    try:
        # Get containber obj i.e usecase, ad group mapping
        container_obj = get_container_obj(request,current_user)
        index_to_use = None
        
        if request.cookies.get('index_container_to_use', None) is not None:
            container_to_use = serializer.loads(request.cookies.get('index_container_to_use'))
            
        
        if container_to_use:
            blob_service_client = BlobServiceClient(account_url="https://stviovcaikeunopoc.blob.core.windows.net/", credential=credential)
            container_client = blob_service_client.get_container_client(container_to_use)  
            
            # Get the blob client for the specified blob  
            blob_client = container_client.get_blob_client(file_name)  
            
            try:
                blob_client.get_blob_properties()
                # Delete the blob file if exist
                blob_client.delete_blob() 
                await updateIndexer(container_to_use, container_to_use,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_to_use))
                if existIndexer(container_to_use+'-general'):
                    await updateIndexerGeneral(container_to_use, container_to_use,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_to_use))
                #log event to eventhub
                send_log_event_to_eventhub(credential, "INFO", f"File '{file_name}' deleted successfully.", "delete_blob_file", container_to_use,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_to_use),start_datetime,"Success","User Action","Deleting a file/folder")  
                return jsonify({"message": f"File {file_name} deleted successfully."}), 200

            except ResourceNotFoundError: 
                # Error occurs when the path is not a file but a folder
                # Fetch all the blobs within that folder
                blob_list = container_client.list_blobs(name_starts_with=file_name + '/')

                # Delete all the blobs
                for blob in blob_list:
                    blob_client = container_client.get_blob_client(blob.name)  
                    blob_client.delete_blob() 
                await updateIndexer(container_to_use, container_to_use,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_to_use))
                if existIndexer(container_to_use+'-general'):
                    await updateIndexerGeneral(container_to_use, container_to_use,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_to_use))
                #log event to eventhub
                send_log_event_to_eventhub(credential, "INFO", f"File '{file_name}' deleted successfully.", "delete_blob_file", container_to_use,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_to_use),start_datetime,"Success","User Action","Deleting a file/folder")  
                return jsonify({"message": f"File {file_name} deleted successfully."}), 200
    
        else:
            #log event to eventhub
            send_log_event_to_eventhub(credential, "ERROR", f"No container Mapped. Try to relogin.", "delete_blob_file", container_to_use,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_to_use),start_datetime,"Failed","User Action","Deleting a file/folder")  
            return jsonify({f"error": "No container Mapped. Try to relogin."}), 500
    except ResourceNotFoundError:  
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"File '{file_name}' not found", "delete_blob_file", container_to_use,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_to_use),start_datetime,"Failed","User Action","Deleting a file/folder")  
        return {"error": f"File {file_name} not found."}, 404  
    except Exception as e:  
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /delete-blob-file for '{file_name}' :{e}", "delete_blob_file", container_to_use,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_to_use),start_datetime,"Failed","User Action","Deleting a file/folder")  
        return jsonify({"error": str(e)}), 500  

def simple_hash(strings, length=10):
    # Define a set of characters to use in the hash
    characters = "abcdefghijklmnopqrstuvwxyz0123456789"
    char_len = len(characters)
    
    # Initialize the hash value
    hash_value = 0
    
    # Iterate over each string in the list
    for string in strings:
        # Iterate over each character in the string
        for char in string:
            # Update the hash value using a simple formula
            hash_value += ord(char)
            # Optionally, you can multiply by a prime number to spread out the values more
            hash_value *= 31
    
    # Convert the hash value to a human-readable string with a fixed length
    hash_string = ""
    for _ in range(length):
        hash_string += characters[hash_value % char_len]
        hash_value //= char_len
    
    return hash_string
# Using lru_cache to cache expensive operations like fetching user group names  
@lru_cache(maxsize=32)  
def cached_fetch_user_group_names(user_token):  
    return fetchUserGroupNames(user_token)  

@lru_cache(maxsize=128)  # This is a simple in-memory cache decorator  
def existIndex_cached(topic_name):  
    return existIndex(topic_name)

@bp.route('/set-topic-folder', methods=['POST'])
async def handle_set_topic_folder():
    start_datetime = datetime.now()
    current_user = get_authenticated_user_details(request_headers=request.headers)
    container_obj = {} 
    container_name = None
    try:
        container_all_flag = None
        # Get the binary data from the request body
        data = await request.body

        # Wrap the binary data in a file-like object
        # Extract the value of the 'folder' field
        folder = data.decode("utf-8")
        
        index_to_use = None
        # Get containber obj i.e usecase, ad group mapping
        container_obj = get_container_obj(request,current_user)
        if request.cookies.get('index_container_to_use', None) is not None:
            container_name = serializer.loads(request.cookies.get('index_container_to_use'))
            
        else:
            container_names = list(container_obj.values())
            container_name = container_names[0]

        if container_name:
            
            if folder is None:
                #log event to eventhub
                send_log_event_to_eventhub(credential, "ERROR", f"Folder not provided,400", "handle_set_topic_folder", container_name,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_name),start_datetime,"Failed","User Action","Setting a topic")
                return jsonify({'error': 'Folder not provided'}), 400

            topic = folder.lower().replace(" ", "-")
            topic_index = container_name + '-' + topic
            
            topic = ""
            container_all_flag = True
            topic_index = container_name
            
            if not existIndex(topic_index): 
                createNewAISearch(container_name, topic, container_all_flag,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_name))
                response = await make_response(jsonify({'success': 'Index Created', 'indexer_result': topic_index}))
                response.set_cookie('index_to_use',  serializer.dumps(f'{topic_index}'), secure=True, max_age=60*60*6)
                return response, 200
            else:
                #await updateIndexer(topic_index, container_name)
                response = await make_response(jsonify({'success': 'Topic folder updated', 'indexer_result': topic_index}))
                response.set_cookie('index_to_use',  serializer.dumps(f'{topic_index}'), secure=True, max_age=60*60*6)

                return response, 200
        else:    
            #log event to eventhub
            send_log_event_to_eventhub(credential, "ERROR", f"No container Mapped. Try to relogin.", "handle_set_topic_folder", container_name,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_name),start_datetime,"Failed","User Action","Setting a topic")
            return jsonify({f"error": "No container Mapped. Try to relogin."}), 400
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /set-topic-folder:{e},500", "set-topic-folder", container_name,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_name),start_datetime,"Failed","User Action","Setting a topic")  
        return jsonify({'error': str(e)}), 500

@bp.route('/api/topic-items', methods=['GET'])
def get_items_of_topic():
    start_datetime = datetime.now()
    try:
        # topic = str(get_encrypted_cookie(request.cookies, "index_container_to_use")).replace('-index', '')
        if request.cookies.get('index_to_use', None) is not None:
            
            index_to_use = serializer.loads(request.cookies.get('index_to_use'))
            container_name = serializer.loads(request.cookies.get('index_container_to_use'))
            topic = index_to_use.split("-")[-1:]

            if topic == "None":
                return jsonify([])

            # Creating BlobServiceClient using the DefaultAzureCredential
            #credential = DefaultAzureCredentialSync()
            #logging.debug(credential.get_token("https://storage.azure.com/.default"))
            blob_service_client = BlobServiceClient(account_url="https://stviovcaikeunopoc.blob.core.windows.net/", credential=credential)
            container_client = blob_service_client.get_container_client(container_name)
            blob_list = container_client.list_blobs()
            data = []
            for blob in blob_list:
                # print(blob.name)
                if f'{topic}' == blob.name.split('/')[0]:
                    blob_url = container_client.get_blob_client(blob).url
                    data.append({
                        "name": blob.name.split('/')[-1],
                        "full_name": blob.name,
                        "url": blob_url
                    })

            return jsonify(data)
        else:
            return jsonify({f"error": "No Topic is set. Try to select a topic or relogin."}), 500
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/list-topics', methods=['GET'])
async def get_items():
    start_datetime = datetime.now()
    current_user = get_authenticated_user_details(request_headers=request.headers)
    container_name = None
    container_obj = {} 
    try:
        index_container_to_use =request.cookies.get('index_container_to_use', None)
        # Get containber obj i.e usecase, ad group mapping
        container_obj = get_container_obj(request,current_user) 
        if index_container_to_use is not None:
            container_name = serializer.loads(request.cookies.get('index_container_to_use'))
        else:
            container_names = list(container_obj.values())
            container_name = container_names[0]

        if container_name:
            
            # Creating BlobServiceClient using the DefaultAzureCredential
            #credential = DefaultAzureCredentialSync()
            #logging.debug(credential.get_token("https://storage.azure.com/.default"))
            blob_service_client = BlobServiceClient(account_url="https://stviovcaikeunopoc.blob.core.windows.net/", credential=credential)
            container_client = blob_service_client.get_container_client(container_name)
            blob_list = container_client.list_blob_names()
            folder_list = list()
            
            for blob in blob_list:
                folder_name = blob  # Extract the first part as folder name
                if '/' in folder_name:
                    folder_list.append(folder_name.split('/')[0])
                else:
                    folder_list.append(folder_name)

            folder_list = list(set(folder_list))    
            # folder_list =['cement-index','anat']    
            # Create a response object
            response = await make_response(jsonify(folder_list))
            if index_container_to_use is None:
                response.set_cookie('index_container_to_use', serializer.dumps(str(container_name)), secure=True, max_age=60*60*6)
            
            # Set the cookie to None 
            # set_encrypted_cookie(response, 'index_container_to_use', 'None')
            #response.set_cookie('index_container_to_use', str(None), secure=True, max_age=60*60*6)
            
            # make_cookie_for_model(response)
            #log event to eventhub
            send_log_event_to_eventhub(credential, "INFO", f"Folder names fetched successfully", "get_items", container_name,current_user['user_name'], get_adgroup_by_usecase(container_obj,container_name),start_datetime,"Success","App Action","Fetching all the folder/topic names for the usecase")  
            
            return response
        else:
            #log event to eventhub
            send_log_event_to_eventhub(credential, "ERROR", f"No AD Group Mapping Found, 400", "get_items", container_name,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_name),start_datetime,"Failed","App Action","Fetching all the folder/topic names for the usecase")  
            return jsonify({"error":"No AD Group Mapping Found"}), 400
    except Exception as e:  
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /list-topics:{e}, 500", "get_items", container_name,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_name),start_datetime,"Failed","App Action","Fetching all the folder/topic names for the usecase")  
        return jsonify({"error": str(e)}), 500  

@bp.route('/set-use-case', methods=['POST'])  
async def set_use_case():
    start_datetime = datetime.now()
    data = await request.json
    container_name = data.get('containerName')
    container_obj = {}
    current_user = get_authenticated_user_details(request_headers=request.headers)
    try:
        # Get containber obj i.e usecase, ad group mapping
        container_obj = get_container_obj(request,current_user)  
        if container_name:  
            serialized_data = serializer.dumps(str(container_name))  
            response = await make_response(jsonify({'message': 'Cookie set'}))  
            response.set_cookie('index_to_use', '', secure=True, expires=0) # To clear the set index_to_use
            response.set_cookie('index_container_to_use', serialized_data, secure=True, max_age=60*60*6)
            make_cookie_for_model(response ,container_name) 
            #log event to eventhub
            send_log_event_to_eventhub(credential, "INFO", f"Container name Cookie set to {container_name}", "set_use_case", container_name , current_user['user_name'],get_adgroup_by_usecase(container_obj,container_name),start_datetime,"Success","User Action","Updating a usecase")  
             
            return response  
        else:  
            #log event to eventhub
            send_log_event_to_eventhub(credential, "ERROR", f"Container name not provided to set new usecase, 400", "set_use_case", container_name , current_user['user_name'],get_adgroup_by_usecase(container_obj,container_name),start_datetime,"Failed","User Action","User Action","Updating a usecase")  
            return jsonify({'message': 'Container name not provided'}), 400  
    except Exception as e:
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /set-use-case:{e}", "set_use_case", container_name, current_user['user_name'],get_adgroup_by_usecase(container_obj,container_name),start_datetime,"Failed","User Action","Updating a usecase")  
        return jsonify({"error":str(e)}),500             
  
@bp.route('/update-settings', methods=['POST'])  
async def update_settings():
    start_datetime = datetime.now()
    data = await request.json  
    container_name = None
    container_obj = {}
    current_user = get_authenticated_user_details(request_headers=request.headers)
    try:
        # Get containber obj i.e usecase, ad group mapping
        container_obj = get_container_obj(request,current_user) 
        
        temperature = data.get('temperature')  
        system_prompt = data.get('systemPrompt')
        index_container_to_use =request.cookies.get('index_container_to_use', None)
        if index_container_to_use is not None:
            container_name = serializer.loads(request.cookies.get('index_container_to_use'))

        if temperature is not None and temperature != '' and system_prompt is not None and system_prompt != '':  
            response = await make_response(jsonify({'message': 'Settings updated'})) 
            make_cookie_for_model(response, container_name)
            encoded_system_prompt = quote(system_prompt , safe='')
            # Set cookies with the provided settings  
            response.set_cookie('AZURE_OPENAI_TEMPERATURE', str(temperature), secure=True, max_age=60*60*6)  
            response.set_cookie('AZURE_OPENAI_SYSTEM_MESSAGE', encoded_system_prompt, secure=True, max_age=60*60*6)
            #log event to eventhub
            send_log_event_to_eventhub(credential, "INFO", f"Settings updated", "update_settings", container_name,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_name),start_datetime,"Success","User Action","Updating system prompt/temperature settings")  
            
            return response  
        else:  
            # Return an error message if either of the settings is not provided  
            #log event to eventhub
            send_log_event_to_eventhub(credential, "ERROR", f"Temperature or System Prompt not provided,400", "update_settings", container_name,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_name),start_datetime,"Failed","User Action","Updating system prompt/temperature settings")  
            return jsonify({'message': 'Temperature or System Prompt not provided'}), 400  
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Container name not provided,400", "update_settings", container_name,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_name),start_datetime,"Failed","User Action","Updating system prompt/temperature settings")  
        return jsonify({'message': 'Container name not provided'}), 400  
    except Exception as e:
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /update-settings:{e},500", "update_settings", container_name ,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_name),start_datetime,"Failed","User Action","Updating system prompt/temperature settings")  
        return jsonify({"error":str(e)}),500

@bp.route('/update-chStatus', methods=['POST'])  
async def update_chat_history():
    start_datetime = datetime.now()
    data = await request.json 
    container_name = None
    container_obj = {}
    current_user = get_authenticated_user_details(request_headers=request.headers)     
    try: 
        # Get containber obj i.e usecase, ad group mapping
        container_obj = get_container_obj(request,current_user)  
        chat_history_status = data.get('chatHistoryStatus')  # This should match your frontend's JSON key  
        index_container_to_use =request.cookies.get('index_container_to_use', None)
        if index_container_to_use is not None:
            container_name = serializer.loads(request.cookies.get('index_container_to_use'))
            if chat_history_status is not None:
                response = await make_response(jsonify({'message': 'Settings updated'})) 
                make_cookie_for_model(response, container_name)
                response.set_cookie('CHAT_HISTORY_ENABLED', str(chat_history_status), secure=True, max_age=60*60*24*30)  # Expires in 30 days
                #log event to eventhub
                send_log_event_to_eventhub(credential, "INFO", f"Chat History Status Updated to {chat_history_status}", "update_chat_history", container_name,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_name),start_datetime,"Success","User Action","Enabling/Disabling chat history")  
              
            return response  
        else:  
            # Return an error message if either of the settings is not provided  
            #log event to eventhub
            send_log_event_to_eventhub(credential, "INFO", f"Container Name not provided,400", "update_chat_history", container_name,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_name),start_datetime,"Success","User Action","Enabling/Disabling chat history")  
            return jsonify({'message': 'Error setting chat history status. Container Name not provided'}), 400  
    except Exception as e:
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /update-chStatus:{e},500", "update_chat_history", container_name ,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_name),start_datetime,"Failed","User Action","Enabling/Disabling chat history")  
        return jsonify({"error":str(e)}),500

def make_cookie_for_model(response, container_name):
    custom_config = None
    #ai search
    response.set_cookie('AZURE_SEARCH_TOP_K', str(AZURE_SEARCH_TOP_K), secure=True, max_age=60*60*6)
    response.set_cookie('AZURE_SEARCH_STRICTNESS', str(AZURE_SEARCH_STRICTNESS), secure=True, max_age=60*60*6)
    response.set_cookie('AZURE_SEARCH_QUERY_TYPE', str(AZURE_SEARCH_QUERY_TYPE), secure=True, max_age=60*60*6)
    response.set_cookie('AZURE_SEARCH_PAGE_CHUNK_SIZE', str(AZURE_SEARCH_PAGE_CHUNK_SIZE), secure=True, max_age=60*60*6)
    response.set_cookie('AZURE_SEARCH_PAGE_OVERLAP_SIZE', str(AZURE_SEARCH_PAGE_OVERLAP_SIZE), secure=True, max_age=60*60*6)
    #model
    response.set_cookie('AZURE_OPENAI_TOP_P', str(AZURE_OPENAI_TOP_P), secure=True, max_age=60*60*6)
    response.set_cookie('AZURE_OPENAI_MODEL', AZURE_OPENAI_MODEL, secure=True, max_age=60*60*6)
    response.set_cookie('AZURE_OPENAI_MAX_TOKENS', str(AZURE_OPENAI_MAX_TOKENS), secure=True, max_age=60*60*6)
    response.set_cookie('AZURE_OPENAI_TEMPERATURE', str(AZURE_OPENAI_TEMPERATURE), secure=True, max_age=60*60*6)
    response.set_cookie('custom_seed', str(123456789), secure=True, max_age=60*60*6)
    if container_name:
        encoded_system_prompt = quote(SYSTEM_PROMPT_CONFIG.get(container_name, AZURE_OPENAI_SYSTEM_MESSAGE), safe='')
        response.set_cookie('AZURE_OPENAI_SYSTEM_MESSAGE', encoded_system_prompt , secure=True, max_age=60*60*6)
    
        custom_config = CUSTOM_TUNING_CONFIG.get(container_name, None)
    if custom_config:
        #ai search
        
        if custom_config.get("top_k"):
            response.set_cookie('AZURE_SEARCH_TOP_K', str(custom_config.get("top_k")), secure=True, max_age=60*60*6)
        if custom_config.get("strictness"):
            response.set_cookie('AZURE_SEARCH_STRICTNESS', str(custom_config.get("strictness")), secure=True, max_age=60*60*6)
        if custom_config.get("query_type"):
            response.set_cookie('AZURE_SEARCH_QUERY_TYPE', str(custom_config.get("query_type")), secure=True, max_age=60*60*6)
        if custom_config.get("chunking"):
            response.set_cookie('CHUNKING', str(custom_config.get("chunking")), secure=True, max_age=60*60*6)
        
        #model
        if custom_config.get("top_p"):
            response.set_cookie('AZURE_OPENAI_TOP_P', str(custom_config.get("top_p")), secure=True, max_age=60*60*6)
        if custom_config.get("model"):
            response.set_cookie('AZURE_OPENAI_MODEL', custom_config.get("model"), secure=True, max_age=60*60*6)
        if custom_config.get("max_tokens"):
            response.set_cookie('AZURE_OPENAI_MAX_TOKENS', str(custom_config.get("max_tokens")), secure=True, max_age=60*60*6)
        if custom_config.get("temperature"):
            response.set_cookie('AZURE_OPENAI_TEMPERATURE', str(custom_config.get("temperature")), secure=True, max_age=60*60*6)
        if custom_config.get("seed"):
            response.set_cookie('custom_seed', str(custom_config.get("seed")), secure=True, max_age=60*60*6)
        
    #response.set_cookie('directPrompting', str(False), secure=True, max_age=60*60*6)
    #response.set_cookie('directPromptingVersion', str(1), secure=True, max_age=60*60*6)
    return response

def make_param_dict_from_cookie(cookies):
    param_dict = {}

    if cookies.get('AZURE_SEARCH_TOP_K'):
        param_dict['AZURE_SEARCH_TOP_K'] = cookies.get('AZURE_SEARCH_TOP_K')

    if cookies.get('AZURE_SEARCH_QUERY_TYPE'):
        param_dict['AZURE_SEARCH_QUERY_TYPE'] = cookies.get('AZURE_SEARCH_QUERY_TYPE')
    
    if cookies.get('AZURE_SEARCH_STRICTNESS'):
        param_dict['AZURE_SEARCH_STRICTNESS'] = cookies.get('AZURE_SEARCH_STRICTNESS')

    if cookies.get('AZURE_SEARCH_PAGE_CHUNK_SIZE'):
        param_dict['AZURE_SEARCH_PAGE_CHUNK_SIZE'] = cookies.get('AZURE_SEARCH_PAGE_CHUNK_SIZE')

    if cookies.get('AZURE_SEARCH_PAGE_OVERLAP_SIZE'):
        param_dict['AZURE_SEARCH_PAGE_OVERLAP_SIZE'] = cookies.get('AZURE_SEARCH_PAGE_OVERLAP_SIZE')
    
    #model
    if cookies.get('AZURE_OPENAI_TOP_P'):
        param_dict['AZURE_OPENAI_TOP_P'] = cookies.get('AZURE_OPENAI_TOP_P')

    if cookies.get('AZURE_OPENAI_MAX_TOKENS'):
        param_dict['AZURE_OPENAI_MAX_TOKENS'] = cookies.get('AZURE_OPENAI_MAX_TOKENS')

    if cookies.get('AZURE_OPENAI_TEMPERATURE'):
        param_dict['AZURE_OPENAI_TEMPERATURE'] = cookies.get('AZURE_OPENAI_TEMPERATURE')

    if cookies.get('custom_seed'):
        param_dict['custom_seed'] = cookies.get('custom_seed')

    if cookies.get('AZURE_OPENAI_SYSTEM_MESSAGE'):
        param_dict['AZURE_OPENAI_SYSTEM_MESSAGE'] = cookies.get('AZURE_OPENAI_SYSTEM_MESSAGE')
    
    if cookies.get('AZURE_OPENAI_MODEL'):
        param_dict['AZURE_OPENAI_MODEL'] = cookies.get('AZURE_OPENAI_MODEL')

    #response.set_cookie('directPrompting', str(False), secure=True, max_age=60*60*6)
    #response.set_cookie('directPromptingVersion', str(1), secure=True, max_age=60*60*6)
    return param_dict

@bp.route('/list-files', methods=['GET'])  
async def list_files():  
    start_datetime = datetime.now()
    current_user = get_authenticated_user_details(request_headers=request.headers)
    container_name = None
    container_obj = {} 
    try:
        # Check if the 'index_container_to_use' cookie is present  
        
        index_container_to_use = request.cookies.get('index_container_to_use')
        # Get containber obj i.e usecase, ad group mapping
        container_obj = get_container_obj(request,current_user) 
        if index_container_to_use:  
            # Deserialize the cookie value  
            container_name = serializer.loads(index_container_to_use)  
        else:  
            
            container_names = list(container_obj.values())
            container_name = container_names[0]
            logging.debug("Mapped Index: "+str(container_name))
            
        #request_container_name = request.args.get('container_name', "rnd") 
        
        if container_name is not None:
            
        #if request_container_name is not None:
            # Creating BlobServiceClient using the DefaultAzureCredential
            #credential = DefaultAzureCredentialSync()
            #logging.debug(credential.get_token("https://storage.azure.com/.default"))
            blob_service_client = BlobServiceClient(account_url="https://stviovcaikeunopoc.blob.core.windows.net/", credential=credential)
            container_client = blob_service_client.get_container_client(container_name)  
            blob_list = container_client.list_blobs()  
            # blobs = [{"name": blob.name} for blob in blob_list if DUMMY_FILE_NAME not in blob.name]
            # Initialize a set for folders  
            folders = set()  
            # Initialize an empty list for filtered blobs  
            filtered_blobs = []  
            
            for blob in blob_list:  
                # Split the blob name by '/'  
                parts = blob.name.split('/')  
                # If the blob has a '/', it's inside a folder  
                if len(parts) > 1:  
                    folder_name = '/'.join(parts[:-1])  
                    # Add the folder name to the set if it's not already present  
                    if folder_name not in folders:  
                        folders.add(folder_name)  
                        # Add the folder itself to the filtered blobs list  
                        filtered_blobs.append({"name": folder_name + '/'})  
                # Check if it's not the dummy file  
                if not blob.name.endswith(f"/{DUMMY_FILE_NAME}"):  
                    # Add the blob to the list, it's a regular file or a non-empty folder  
                    filtered_blobs.append({"name": blob.name})  
            
            # Return the list of blobs and folders, excluding the dummy file  
            response = await make_response(jsonify(filtered_blobs))  

        
            if index_container_to_use is None:
                response.set_cookie('index_container_to_use', serializer.dumps(str(container_name)), secure=True, max_age=60*60*6)
            #log event to eventhub
            send_log_event_to_eventhub(credential, "INFO", f"All the files fetched successfully", "list_files", container_name ,current_user['user_name'], get_adgroup_by_usecase(container_obj,container_name),start_datetime,"Success","App Action","Fetching all the files for the selected usecase")  
            return response, 200  
        else:
            #log event to eventhub
            send_log_event_to_eventhub(credential, "ERROR", f"No container name is provided, 400", "list_files", container_name,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_name),start_datetime,"Failed","App Action","Fetching all the files for the selected usecase")  
            return jsonify("No container name is provided"), 400  
    except Exception as e:  
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /list-files:{e}, 500", "list_files", index_container_to_use, current_user['user_name'],get_adgroup_by_usecase(container_obj,container_name),start_datetime,"Failed","App Action","Fetching all the files for the selected usecase")  
        return jsonify({"error": str(e)}), 500  

def dictFromJSON(file_path):
# Open the JSON file
    with open(file_path, 'r') as file:
        # Load JSON data from file
        data_dict = json.load(file)

    # Now data_dict is a Python dictionary that contains the data from the JSON file
    return data_dict


def changeValueOfKeyDict(d, k, v):
    d[k] = v
    return d

def deleteAISearchResourcesForTopic(container_name, topic_name,user,ad_group):
    start_datetime = datetime.now()
    try:
        topic_prefix = f"{container_name}-{topic_name}"

        data_sources_list = getDataSources()
        indexes_list = getIndexes()
        indexers_list = getIndexers()
        skillset_list = getSkillsets()

        indexer_client = SearchIndexerClient(f"https://{AZURE_SEARCH_SERVICE}.search.windows.net", credential)
        index_client = SearchIndexClient(f"https://{AZURE_SEARCH_SERVICE}.search.windows.net", credential)
        
        for indexer_name in indexers_list:
            if topic_prefix in indexer_name:
                try:
                    indexer_client.delete_indexer(indexer_name)
                except Exception as e:
                    logging.error(e)
                    #log event to eventhub
                    send_log_event_to_eventhub(credential, "ERROR", f"Exception in deleteAISearchResourcesForTopic :{e}", "deleteAISearchResourcesForTopic", container_name,user,ad_group,start_datetime,"Failed")  

        for skillset_name in skillset_list:
            if topic_prefix in skillset_name:
                try:
                    indexer_client.delete_skillset(skillset_name)
                except Exception as e:
                    logging.error(e)
                    #log event to eventhub
                    send_log_event_to_eventhub(credential, "ERROR", f"Exception in deleteAISearchResourcesForTopic :{e}", "deleteAISearchResourcesForTopic", container_name,user,ad_group,start_datetime,"Failed")  

        for data_source in data_sources_list:
            if topic_prefix in data_source:
                try:
                    indexer_client.delete_skillset(data_source)
                except Exception as e:
                    logging.error(e)
                    #log event to eventhub
                    send_log_event_to_eventhub(credential, "ERROR", f"Exception in deleteAISearchResourcesForTopic :{e}", "deleteAISearchResourcesForTopic", container_name,user,ad_group,start_datetime,"Failed")  

        for index_name in indexes_list:
            if topic_prefix in index_name:
                try:
                    index_client.delete_index(index_name)
                
                except Exception as e:
                    logging.error(e)
                    #log event to eventhub
                    send_log_event_to_eventhub(credential, "ERROR", f"Exception in deleteAISearchResourcesForTopic :{e}", "deleteAISearchResourcesForTopic", container_name,user,ad_group,start_datetime,"Failed","App Action","Deleting the AI search resources")  

        #log event to eventhub
        send_log_event_to_eventhub(credential, "INFO", f"Deleted AISearchResourcesForTopic :[{indexer_name} indexer,{index_name} index,{data_source} datasource,{skillset_name} skillset]", "deleteAISearchResourcesForTopic", container_name,user,ad_group,start_datetime,"Success","App Action","Deleting the AI search resources")  

    except Exception as e:
        logging.error(e)
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in deleteAISearchResourcesForTopic :{e}", "deleteAISearchResourcesForTopic", container_name,start_datetime,"Failed","App Action","Deleting the AI search resources")  

def createNewAISearch(container_name, topic_name, container_all_flag,user,ad_group):
    start_datetime = datetime.now()
    # Create INDEX
    try:
        custom_config = CUSTOM_TUNING_CONFIG.get(container_name, {})
        chunking = "true"
        
        if custom_config.get("chunking"):
            
            if custom_config.get("chunking") == "false":
                logging.debug("Setting chunking to false")
                chunking = "false"
        newIndexDict = dictFromJSON(r'./AISearch/index_temp.json')
        skillset_name = None
        index_name = None
        if container_all_flag:
            newIndexDict = changeValueOfKeyDict(newIndexDict, "name", f'{container_name}-index')
            index_name = f'{container_name}-index'
            skillset_name = f'{container_name}-general-skillset'
            pagenumber_skillset_name = f'{container_name}-pagenumber-skillset'
        
        print(newIndexDict['name'])
        newIndex = SearchIndex.from_dict(newIndexDict)
        # print(newIndexDict)
        try:
            SearchIndexClient(f"https://{AZURE_SEARCH_SERVICE}.search.windows.net", credential).create_index(newIndex)
        except Exception as e:
            #log event to eventhub
            send_log_event_to_eventhub(credential, "ERROR", f"Exception in createNewAISearch :{e}", "createNewAISearch", container_name,user,ad_group,start_datetime,"Failed","App Action","Creating the new AI search resources")  
            logging.error(e)

        # Create a General Skillset with Split and Embed Skill
        newSkillsetDict = dictFromJSON(r'./AISearch/skillset_temp.json')
        newProjection = newSkillsetDict["indexProjections"]["selectors"][0]
        newProjection = changeValueOfKeyDict(newProjection, "targetIndexName", index_name)
        newSkillsetDict["indexProjections"]["selectors"] = [newProjection]
        newSkillsetDict = changeValueOfKeyDict(newSkillsetDict, "name", skillset_name)

        newPagenumberSkillsetDict = dictFromJSON(r'./AISearch/pagenumber_skillset_temp.json')
        newPageNumberProjection = newPagenumberSkillsetDict["indexProjections"]["selectors"][0]
        newPageNumberProjection = changeValueOfKeyDict(newPageNumberProjection, "targetIndexName", index_name)
        newPagenumberSkillsetDict["indexProjections"]["selectors"] = [newPageNumberProjection]
        newPagenumberSkillsetDict = changeValueOfKeyDict(newPagenumberSkillsetDict, "name", pagenumber_skillset_name)

        if chunking == "true":
            newSkillset = SearchIndexerSkillset.from_dict(newSkillsetDict)
            newPagenumberSkillset = SearchIndexerSkillset.from_dict(newPagenumberSkillsetDict)
            try:
                skillset = SearchIndexerClient(f"https://{AZURE_SEARCH_SERVICE}.search.windows.net", credential).create_or_update_skillset(newSkillset) 
                pagenunmber_skillset = SearchIndexerClient(f"https://{AZURE_SEARCH_SERVICE}.search.windows.net", credential).create_or_update_skillset(newPagenumberSkillset)
            except Exception as e:
                #log event to eventhub
                send_log_event_to_eventhub(credential, "ERROR", f"Exception in createNewAISearch :{e}", "createNewAISearch", container_name,user,ad_group,start_datetime,"Failed","App Action","Creating the new AI search resources")  
                logging.error(e)
        #Create data source
        
        newDataSourceDict = dictFromJSON(r'./AISearch/data_source_temp.json')
        
        if container_all_flag:
            newDataSourceDict = changeValueOfKeyDict(newDataSourceDict, "name", f'{container_name}-data-source')
            newDataSourceDict["container"]  = changeValueOfKeyDict(newDataSourceDict["container"], "name", f'{container_name}')
            newDataSourceDict["container"] = changeValueOfKeyDict(newDataSourceDict["container"], "query", None)
        else:
            newDataSourceDict = changeValueOfKeyDict(newDataSourceDict, "name", f'{container_name}-{topic_name}-data-source')
            newDataSourceDict["container"]  = changeValueOfKeyDict(newDataSourceDict["container"], "name", f'{container_name}')
            newDataSourceDict["container"] = changeValueOfKeyDict(newDataSourceDict["container"], "query", f'{topic_name}')
        try:
            newDataSource = SearchIndexerDataSourceConnection.from_dict(newDataSourceDict)
        except Exception as e:
            #log event to eventhub
            send_log_event_to_eventhub(credential, "ERROR", f"Exception in createNewAISearch :{e}", "createNewAISearch", container_name,user,ad_group,start_datetime,"Failed","App Action","Creating the new AI search resources")  
            logging.error(e)

        try:
            SearchIndexerClient(endpoint=f"https://{AZURE_SEARCH_SERVICE}.search.windows.net", credential=credential, api_version="2024-05-01-preview").create_data_source_connection(newDataSource)
        except Exception as e:
            #log event to eventhub
            send_log_event_to_eventhub(credential, "ERROR", f"Exception in createNewAISearch :{e}", "createNewAISearch", container_name,user,ad_group,start_datetime,"Failed","App Action","Creating the new AI search resources")  
            logging.error(e)

        # #Create Indexer

        newIndexerDict = dictFromJSON(r'./AISearch/indexer_temp.json')
        newIndexerDict = changeValueOfKeyDict(newIndexerDict, "skillsetName", None)

        newPagenumberIndexerDict = dictFromJSON(r'./AISearch/pagenumber_indexer_temp.json')
        newPagenumberIndexerDict = changeValueOfKeyDict(newPagenumberIndexerDict, "skillsetName", None)

        if container_all_flag:
            newIndexerDict = changeValueOfKeyDict(newIndexerDict, "name", f'{container_name}-general-indexer')
            newIndexerDict = changeValueOfKeyDict(newIndexerDict, "targetIndexName", f'{container_name}-index')
            newIndexerDict = changeValueOfKeyDict(newIndexerDict, "dataSourceName", f'{container_name}-data-source')
            
            newPagenumberIndexerDict = changeValueOfKeyDict(newPagenumberIndexerDict, "name", f'{container_name}-indexer')
            newPagenumberIndexerDict = changeValueOfKeyDict(newPagenumberIndexerDict, "targetIndexName", f'{container_name}-index')
            newPagenumberIndexerDict = changeValueOfKeyDict(newPagenumberIndexerDict, "dataSourceName", f'{container_name}-data-source')
            
            if chunking == "true":
                newIndexerDict = changeValueOfKeyDict(newIndexerDict, "skillsetName", f'{container_name}-general-skillset')
                newPagenumberIndexerDict = changeValueOfKeyDict(newPagenumberIndexerDict, "skillsetName", f'{container_name}-pagenumber-skillset')

        newIndexer = SearchIndexer.from_dict(newIndexerDict)
        newPagenumberIndexer = SearchIndexer.from_dict(newPagenumberIndexerDict)
        try:
            SearchIndexerClient(f"https://{AZURE_SEARCH_SERVICE}.search.windows.net", credential).create_indexer(newIndexer)
        except Exception as e:
            #log event to eventhub
            send_log_event_to_eventhub(credential, "ERROR", f"Exception in createNewAISearch :{e}", "createNewAISearch", container_name,user,ad_group,start_datetime,"Failed","App Action","Creating the new AI search resources")  
            logging.error(e)

        try:
            SearchIndexerClient(f"https://{AZURE_SEARCH_SERVICE}.search.windows.net", credential).create_indexer(newPagenumberIndexer)
        except Exception as e:
            #log event to eventhub
            send_log_event_to_eventhub(credential, "ERROR", f"Exception in createNewAISearch :{e}", "createNewAISearch", container_name,user,ad_group,start_datetime,"Failed","App Action","Creating the new AI search resources")  
            logging.error(e)

        if container_all_flag:
            #log event to eventhub
            send_log_event_to_eventhub(credential, "INFO", f"NewAISearch created: '{container_name}-index', f'{container_name}-indexer', f'{container_name}-data-source'", "createNewAISearch", container_name,user,ad_group,start_datetime,"Success","App Action","Creating the new AI search resources")  
            return True, [f'{container_name}-index', f'{container_name}-indexer', f'{container_name}-data-source']
        
    except Exception as e:
        logging.exception(e)
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in createNewAISearch for {container_name}-indexer :{e}", "createNewAISearch", container_name,user,ad_group,start_datetime,"Failed","App Action","Creating the new AI search resources")  
        return False, ['Error', e]

def getSkillsets():
    skillset_list = SearchIndexerClient(f"https://{AZURE_SEARCH_SERVICE}.search.windows.net", credential).get_skillset_names()
    return skillset_list

def getIndexes():
    indexes_list = SearchIndexClient(f"https://{AZURE_SEARCH_SERVICE}.search.windows.net", credential).list_index_names()
    return indexes_list

def getDataSources():
    data_sources = SearchIndexerClient(f"https://{AZURE_SEARCH_SERVICE}.search.windows.net", credential).get_data_source_connection_names()
    return data_sources

def getIndexers():
    return SearchIndexerClient(f"https://{AZURE_SEARCH_SERVICE}.search.windows.net", credential).get_indexer_names()

def existIndexer(topic_name):
    return f'{topic_name}-indexer' in SearchIndexerClient(f"https://{AZURE_SEARCH_SERVICE}.search.windows.net", credential).get_indexer_names()

def existIndex(topic_name):
    return f'{topic_name}-index' in SearchIndexClient(f"https://{AZURE_SEARCH_SERVICE}.search.windows.net", credential).list_index_names()



def createOnlyIndexerForSingle(indexer_name, container_name,user,ad_group): 
    start_datetime = datetime.now()   
    custom_config = CUSTOM_TUNING_CONFIG.get(container_name, {})
    chunking = "true"
    
    if custom_config.get("chunking"):
        if custom_config.get("chunking") == "false":
            logging.debug("Setting chunking to false")
            chunking = "false"
    
    
    # Create a Skillset with Split and Embed Skill
    newSkillsetDict = dictFromJSON(r'./AISearch/skillset_temp.json')
    newProjection = newSkillsetDict["indexProjections"]["selectors"][0]
    newProjection = changeValueOfKeyDict(newProjection, "targetIndexName", f'{indexer_name}-index')
    newSkillsetDict["indexProjections"]["selectors"] = [newProjection]
    newSkillsetDict = changeValueOfKeyDict(newSkillsetDict, "name", f'{indexer_name}-skillset')
    
    if chunking == "true":
        newSkillset = SearchIndexerSkillset.from_dict(newSkillsetDict)
        try:
            skillset = SearchIndexerClient(f"https://{AZURE_SEARCH_SERVICE}.search.windows.net", credential).create_or_update_skillset(newSkillset) 
            #log event to eventhub
            send_log_event_to_eventhub(credential, "INFO", f"Skillset was created or updated:{newSkillset}", "createOnlyIndexerForSingle", container_name,user,ad_group,start_datetime,"Success","App Action","Creating Indexer")  

        except Exception as e:
            logging.error(e)
            #log event to eventhub
            send_log_event_to_eventhub(credential, "ERROR", f"Exception in createOnlyIndexerForSingle for {newSkillset}:{e}", "createOnlyIndexerForSingle", container_name,user,ad_group,start_datetime,"Failed","App Action","Creating Indexer")  

    newIndexerDict = dictFromJSON(r'./AISearch/indexer_temp.json')
    newIndexerDict = changeValueOfKeyDict(newIndexerDict, "skillsetName", None)
    newIndexerDict = changeValueOfKeyDict(newIndexerDict, "name", f'{indexer_name}-indexer')
    newIndexerDict = changeValueOfKeyDict(newIndexerDict, "targetIndexName", f'{indexer_name}-index')
    newIndexerDict = changeValueOfKeyDict(newIndexerDict, "dataSourceName", f'{indexer_name}-data-source')
    if chunking == "true":
        newIndexerDict = changeValueOfKeyDict(newIndexerDict, "skillsetName", f'{indexer_name}-skillset')
    newIndexer = SearchIndexer.from_dict(newIndexerDict)
    try:
        SearchIndexerClient(f"https://{AZURE_SEARCH_SERVICE}.search.windows.net", credential).create_indexer(newIndexer)
        #log event to eventhub
        send_log_event_to_eventhub(credential, "INFO", f"New Indexer was created:{newIndexer}", "createOnlyIndexerForSingle", container_name,user,ad_group,start_datetime,"Success","App Action","Creating Indexer")  
  
    except Exception as e:
        logging.exception(e)
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in createOnlyIndexerForSingle {newIndexer} :{e}", "createOnlyIndexerForSingle", container_name,user,ad_group,start_datetime,"Failed","App Action","Creating Indexer")  

            
async def updateIndexer(topic_name, container_name,user,ad_group):
    start_datetime = datetime.now()
    try:        
        SearchIndexerClient(f"https://{AZURE_SEARCH_SERVICE}.search.windows.net", credential).run_indexer(f'{topic_name}-indexer')

        if existIndex(topic_name):
            response = await make_response(jsonify(topic_name))
            # set_encrypted_cookie(response, 'index_container_to_use', f'{topic_name}-index')
            response.set_cookie('index_container_to_use', str(f'{topic_name}-index'), secure=True, max_age=60*60*6)
            #log event to eventhub
            send_log_event_to_eventhub(credential, "INFO", f"{topic_name}-index Indexer was updated", "updateIndexer", container_name,user,ad_group,start_datetime,"Success","App Action","Updating Indexer")  
        
            return response
    except Exception as e:
        createOnlyIndexerForSingle(topic_name, container_name,user,ad_group)
        logging.exception(e)
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in updateIndexer for {topic_name}-index:{e}", "updateIndexer", container_name,user,ad_group,start_datetime,"Failed","App Action","Updating Indexer")  
        return False, {'Error': e}

async def updateIndexerGeneral(topic_name, container_name,user,ad_group):
    start_datetime = datetime.now()
    try:        
        SearchIndexerClient(f"https://{AZURE_SEARCH_SERVICE}.search.windows.net", credential).run_indexer(f'{topic_name}-general-indexer')

        if existIndex(topic_name):
            response = await make_response(jsonify(topic_name))
            # set_encrypted_cookie(response, 'index_container_to_use', f'{topic_name}-index')
            response.set_cookie('index_container_to_use', str(f'{topic_name}-index'), secure=True, max_age=60*60*6)
            #log event to eventhub
            send_log_event_to_eventhub(credential, "INFO", f"{topic_name}-index Indexer was updated", "updateIndexerGeneral", container_name,user,ad_group,start_datetime,"Success","App Action","Updating General Indexer")  
        
            return response
    except Exception as e:
        logging.exception(e)
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in updateIndexerGeneral for {topic_name}-index:{e}", "updateIndexerGeneral", container_name,user,ad_group,start_datetime,"Failed","App Action","Updating General Indexer")  
        return False, {'Error': e}

async def process_files(uploaded_files, request, folderName, isOverwrite, start_datetime):   
    # Change after testing
    container_all_flag = None
    index_to_use = None
    current_user = get_authenticated_user_details(request_headers=request.headers)
    container_to_use = None
    container_obj ={}
    if request.cookies.get('index_container_to_use', None) is not None:
        # Get containber obj i.e usecase, ad group mapping
        container_obj = get_container_obj(request,current_user) 
        # index_to_use = serializer.loads(request.cookies.get('index_container_to_use'))
        #Modified the cookie to be used to get the container
        container_to_use = serializer.loads(request.cookies.get('index_container_to_use'))
        topic = folderName.lower().replace(" ", "-")   #If foldername is present then use it
        if request.cookies.get('index_to_use'):
            index_to_use = serializer.loads(request.cookies.get('index_to_use'))
        try:
            
            tasks = {}
            for files_new in uploaded_files: 
                
                tasks[files_new.filename] = upload(files_new, container_to_use, folderName, isOverwrite, current_user, container_obj,start_datetime)
            # Logic to update particular indexer if multiple topics are selected while uploading
            
            if container_to_use:
                if existIndexer(container_to_use):
                    await updateIndexer(container_to_use, container_to_use, current_user['user_name'],get_adgroup_by_usecase(container_obj,container_to_use))
                    await updateIndexerGeneral(container_to_use, container_to_use, current_user['user_name'],get_adgroup_by_usecase(container_obj,container_to_use))
                else:
                    createNewAISearch(container_to_use, "", True, current_user['user_name'],get_adgroup_by_usecase(container_obj,container_to_use))                

            
            if all(value == True for value in tasks.values()):
                #log event to eventhub
                send_log_event_to_eventhub(credential, "INFO", f"All the files uploaded successfully in the folder {folderName}: {[f.filename for f in uploaded_files]}", "process_files", container_to_use,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_to_use),start_datetime,"Success","User Action", "Uploading a file")             
                return jsonify({"info": f"All files uploaded successfully {[f.filename for f in uploaded_files]}"}), 201
            elif all(value == False for value in tasks.values()):
                #log event to eventhub
                send_log_event_to_eventhub(credential, "INFO", f"File Already exists in the folder {folderName}: {[f.filename for f in uploaded_files]}", "process_files", container_to_use, current_user['user_name'],get_adgroup_by_usecase(container_obj,container_to_use),start_datetime,"Success","User Action", "Uploading a file")             
                return jsonify({"info": f"File Already exists {[f.filename for f in uploaded_files]}"}), 409
            else:
                #log event to eventhub
                send_log_event_to_eventhub(credential, "ERROR", f"Some files failed to upload in the folder {folderName}: {[f.filename for f in uploaded_files]}, 400", "process_files", container_to_use,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_to_use),start_datetime,"Failed","User Action", "Uploading a file")             
                return "Some files failed to upload. Check logs for details."
        except Exception as e:
            logging.error(e)
            #log event to eventhub
            send_log_event_to_eventhub(credential, "ERROR", f"Exception in process_files in the folder {folderName}: {[f.filename for f in uploaded_files]} :{e},500", "process_files", container_to_use,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_to_use),start_datetime,"Failed","User Action", "Uploading a file")  
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "No mapped container found, please hard refresh or start again in an incognito window"}), 400

def upload(file, container_to_use, topic, isOverwrite, current_user, container_obj, start_datetime):
    try:
        
        blob_service_client = BlobServiceClient(account_url="https://stviovcaikeunopoc.blob.core.windows.net/", credential=DefaultAzureCredentialSync())
        container_client = blob_service_client.get_container_client(container_to_use)  
        blob_client = container_client.get_blob_client(topic +'\\'+ file.filename)

        # check if blob exists
        blob_exists = blob_client.exists()

        if blob_exists and isOverwrite != 'true':
            return False

        blob_client.upload_blob(file, overwrite=True)
        
        return True
    except Exception as e:
        logging.error(e)
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in upload in the folder {topic}: {file.filename}:{e}", "upload", container_to_use,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_to_use),start_datetime,"Failed","User Action", "Uploading a file")  
        return False

### ---------------- UPLOAD ------------------###
@bp.route('/upload', methods=['POST'])
async def upload_files():
    start_datetime = datetime.now()
    folderName = request.args.get('folderName') 
    isOverwrite = request.args.get('isOverwrite') 
    files = await request.files  
    uploaded_files = files.getlist('files') 
    
    return await process_files(uploaded_files, request, folderName, isOverwrite,start_datetime)

### -------------Download file -------------------###
@bp.route('/api/topic-items/<path:blob_name>', methods=['GET'])
async def get_blob_content(blob_name):
    start_datetime = datetime.now()
    container_to_use= None
    current_user = get_authenticated_user_details(request_headers=request.headers)
    container_obj = {}
    try:
        if request.cookies.get('index_container_to_use', None) is not None:
            container_to_use = serializer.loads(request.cookies.get('index_container_to_use'))
        blob_service_client = BlobServiceClient(account_url="https://stviovcaikeunopoc.blob.core.windows.net/", credential=DefaultAzureCredentialSync())
        container_client = blob_service_client.get_container_client(container_to_use) 
        blob_client = container_client.get_blob_client(blob_name)
        # Get containber obj i.e usecase, ad group mapping
        container_obj = get_container_obj(request,current_user) 
        # Download blob data into memory
        blob_data =  blob_client.download_blob()
        blob_stream =  blob_data.readall()
        mime_type, _ = mimetypes.guess_type(blob_name)
        # Create a file-like object from the blob data
        blob_bytes = io.BytesIO(blob_stream)
        
        #log event to eventhub
        send_log_event_to_eventhub(credential, "INFO", f"File Downloaded successfully:{blob_name}", "get_blob_content", container_to_use, current_user['user_name'],get_adgroup_by_usecase(container_obj,container_to_use),start_datetime,"Success","User Action","Downloading a file")  
        
        # Encode the filename to handle special characters  
        safe_filename = quote(blob_name.split('/')[-1])  
    
        # Send the file-like object as a response  
        return await send_file(  
            blob_bytes,  
            mimetype=mime_type or 'application/octet-stream',  
            as_attachment=True,  
            attachment_filename=safe_filename  
        ) 
    except ResourceNotFoundError:  
        error_message = f"Blob '{blob_name}' not found in container '{container_to_use}'."  
        #log event to eventhub  
        send_log_event_to_eventhub(credential, "ERROR", error_message, "get_blob_content", container_to_use, current_user['user_name'], get_adgroup_by_usecase(container_obj,container_to_use), start_datetime, "Failed", "User Action", "Downloading a file")  
        return jsonify({"error": error_message}), 404      
    except Exception as e:
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in get_blob_content for '{blob_name}' :{e},500", "get_blob_content", container_to_use, current_user['user_name'],get_adgroup_by_usecase(container_obj,container_to_use),start_datetime,"Failed","User Action","Downloading a file")  
        return jsonify({"error":str(e)}), 500

async def generate_title(conversation_messages, container_name,current_user,container_obj):
    start_datetime = datetime.now()
    ## make sure the messages are sorted by _ts descending
    title_prompt = 'Summarize the conversation so far into a 4-word or less title. Do not use any quotation marks or punctuation. Respond with a json object in the format {"title": string}. Do not include any other commentary or description.'

    messages = [{'role': msg['role'], 'content': msg['content']} for msg in conversation_messages]
    messages.append({'role': 'user', 'content': title_prompt})

    try:
        azure_openai_client = init_openai_client(None, None, use_data=False)
        response = await azure_openai_client.chat.completions.create(
            model=AZURE_OPENAI_MODEL,
            messages=messages,
            temperature=1,
            max_tokens=64
        )        
        content_str = response.choices[0].message.content.replace('{{', '{').replace('}}', '}').strip('`').replace('json\n', '', 1)
        logging.debug(content_str) ##for testing in prod
        title = json.loads(content_str).get('title')
        
        return title
    except Exception as e:

        logging.error("Exception in generate_title", e)  
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in generate_title: {e}", "generate_title", container_name,current_user,get_adgroup_by_usecase(container_obj,container_name),start_datetime,"Failed","App Action","Generating a title for conversation")  
        return messages[-2]['content']

### -------------Rename the topic ----------------###
@bp.route('/rename-folder', methods=['POST'])
async def handle_rename_folder():
    start_datetime = datetime.now()
    current_user = get_authenticated_user_details(request_headers=request.headers)
    container_name= None
    container_obj = {}
    try:
        # Get containber obj i.e usecase, ad group mapping
        container_obj = get_container_obj(request,current_user) 
        data = await request.body
        payload = json.loads(data.decode('utf-8'))

        # Extract fields from the payload
        old_folder_name = payload.get('old_topic_name')
        new_folder_name = payload.get('new_topic_name')
       
        # Validate if both old and new folder names are provided
        if not old_folder_name or not new_folder_name:
            return jsonify({'error': 'Old and new folder names must be provided'}), 400
        
        # Construct the container name (your Azure Blob Storage container name)
        if request.cookies.get('index_container_to_use', None) is not None:
        
        # index_to_use = serializer.loads(request.cookies.get('index_container_to_use'))
        #Modified the cookie to be used to get the container
            container_name = serializer.loads(request.cookies.get('index_container_to_use'))
       
        else:
            #log event to eventhub
            send_log_event_to_eventhub(credential, "ERROR", f"No mapped container found", "handle_rename_folder", container_name,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_name),start_datetime,"Failed","User Action","Renaming a folder")  
            return jsonify({"error": "No mapped container found, please hard refresh or start again in an incognito window"}), 400
        
        # Create BlobClient to interact with the container
        blob_service_client = BlobServiceClient(account_url="https://stviovcaikeunopoc.blob.core.windows.net/", credential=DefaultAzureCredentialSync())
        container_client = blob_service_client.get_container_client(container_name)
        
        # Extract the path without the old folder name  
        base_path = os.path.dirname(old_folder_name)  
        
        # Create the new folder path by combining the base path with the new folder name  
        new_folder_path = os.path.join(base_path, new_folder_name)  
        blob_list = container_client.list_blobs(name_starts_with=old_folder_name + '/')
        
        
        # Rename each blob to the new folder
        for blob in blob_list:

            old_blob_name = blob.name
  
            # Replace the old folder path with the new folder path  
            new_blob_name = old_blob_name.replace(old_folder_name, new_folder_path, 1)
            
            # Get the blob client for the source blob
            blob_client = container_client.get_blob_client(old_blob_name)
            
            # Get the URL of the source blob
            source_blob_url = blob_client.url
            
            # Create the blob client for the destination blob
            destination_blob_client = container_client.get_blob_client(new_blob_name)
            
            # Start the copy operation
            destination_blob_client.start_copy_from_url(source_blob_url)
        
        # Optional: If needed, delete the old folder after renaming all blobs
            container_client.delete_blob(old_blob_name)  # Uncomment to delete the old folder
        await updateIndexer(container_name, container_name,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_name))
        if existIndexer(container_name+'general'):
            await updateIndexerGeneral(container_name, container_name,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_name))
        #log event to eventhub
        send_log_event_to_eventhub(credential, "INFO", f"{old_folder_name} Folder renamed successfully to:{new_folder_name}", "handle_rename_folder", container_name, current_user['user_name'],get_adgroup_by_usecase(container_obj,container_name),start_datetime,"Success","User Action","Renaming a folder")  
        
        return jsonify({'success': 'Folder renamed successfully'}), 200
    
    except Exception as e:
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception in /rename-folder while renaming a folder '{old_folder_name}':{e},500", "handle_rename_folder", container_name, current_user['user_name'],get_adgroup_by_usecase(container_obj,container_name),start_datetime,"Failed","User Action","Renaming a folder")  
        return jsonify({'error': str(e)}), 500

### -------------Create new folder with dummy file -----#
@bp.route('/create-new-folder', methods=['POST'])  
async def create_new_folder(): 
    start_datetime = datetime.now()
    current_user = get_authenticated_user_details(request_headers=request.headers)
    container_name= None 
    container_obj = {}
    try:  
        # Parse the folder name from the request body  
        folder_name = await request.get_data(as_text=True)  
        # Get containber obj i.e usecase, ad group mapping
        container_obj = get_container_obj(request,current_user) 
        if request.cookies.get('index_container_to_use', None) is not None:
        
        # index_to_use = serializer.loads(request.cookies.get('index_container_to_use'))
        #Modified the cookie to be used to get the container
            container_name = serializer.loads(request.cookies.get('index_container_to_use'))
       
        else:
            return jsonify({"error": "No mapped container found, please hard refresh or start again in an incognito window"}), 400
        
        # Create BlobClient to interact with the container
        blob_service_client = BlobServiceClient(account_url="https://stviovcaikeunopoc.blob.core.windows.net/", credential=DefaultAzureCredentialSync())
        container_client = blob_service_client.get_container_client(container_name) 
  
        # Construct the full blob name for the dummy file within the new folder  
        dummy_blob_name = f"{folder_name}/{DUMMY_FILE_NAME}"  
  
        # Get the blob client for the dummy file  
        blob_client = container_client.get_blob_client(dummy_blob_name)  
  
        # Upload the dummy file to create the folder  
        blob_client.upload_blob(DUMMY_FILE_CONTENT, overwrite=True)  
        
        #log event to eventhub
        send_log_event_to_eventhub(credential, "INFO", f"New folder '{folder_name}' was created successfully", "create_new_folder", container_name, current_user['user_name'],get_adgroup_by_usecase(container_obj,container_name),start_datetime,"Success","User Action","Creating a new Folder")  
        
        # Return a success response  
        return jsonify({"message": f"Folder '{folder_name}' created successfully with dummy file."}), 201  
  
    except Exception as e:  
        #log event to eventhub
        send_log_event_to_eventhub(credential, "ERROR", f"Exception /create-new-folder while creating a new folder '{folder_name}':{e},500", "create_new_folder", container_name,current_user['user_name'],get_adgroup_by_usecase(container_obj,container_name),start_datetime,"Failed","User Action","Creating a new Folder")  
        return jsonify({"error": str(e)}), 500

app = create_app()
if __name__ == "__main__": 
    asyncio.run(app.run_task(debug=True))
