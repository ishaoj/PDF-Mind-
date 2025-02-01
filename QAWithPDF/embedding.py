from llama_index.core import VectorStoreIndex
from llama_index.core import ServiceContext
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.gemini import GeminiEmbedding

from QAWithPDF.data_ingestion import load_data
from QAWithPDF.model_api import load_model

import sys
from exception import CustomException
from logger import logging

def download_gemini_embedding(model,document):
    """
    Downloads and initializes a Gemini Embedding model for vector embeddings.

    Returns:
    - VectorStoreIndex: An index of vector embeddings for efficient similarity queries.
    """
    try:
        logging.info("")
        gemini_embed_model = GeminiEmbedding(model_name="models/embedding-001")
        from llama_index.core.settings import Settings

# Store settings values in variables, but work with Settings directly
        Settings.llm = model
        Settings.embed_model = gemini_embed_model
        Settings.chunk_size = 800
        Settings.chunk_overlap = 20

        # If you need to reference these settings later, you can access them directly
        current_llm = Settings.llm
        current_embed_model = Settings.embed_model
        current_chunk_size = Settings.chunk_size
        current_chunk_overlap = Settings.chunk_overlap
        logging.info("")
        index = VectorStoreIndex.from_documents(
            document,
            llm=Settings.llm,
            embed_model=Settings.embed_model
        )
        index.storage_context.persist()

        logging.info("")
        query_engine = index.as_query_engine()
        return query_engine
    except Exception as e:
        raise CustomException(e,sys)