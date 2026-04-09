import re

from config.settings import settings


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}


def get_vector_store_embedding_model_name() -> str:
    model_name = (settings.EMBEDDING_MODEL or DEFAULT_EMBEDDING_MODEL).strip()
    return model_name or DEFAULT_EMBEDDING_MODEL


def get_vector_store_collection_name() -> str:
    model_name = get_vector_store_embedding_model_name()
    if model_name == DEFAULT_EMBEDDING_MODEL:
        return settings.VECTOR_STORE_COLLECTION_NAME

    suffix = re.sub(r"[^a-zA-Z0-9]+", "-", model_name).strip("-").lower()
    if not suffix:
        return settings.VECTOR_STORE_COLLECTION_NAME

    return f"{settings.VECTOR_STORE_COLLECTION_NAME}-{suffix}"


def get_vector_store_connection_args() -> dict[str, str]:
    connection_args = {"uri": settings.VECTOR_STORE_URI}
    if settings.VECTOR_STORE_TOKEN:
        connection_args["token"] = settings.VECTOR_STORE_TOKEN
    return connection_args


def get_vector_store_dimension() -> int:
    if settings.VECTOR_STORE_DIM > 0:
        return settings.VECTOR_STORE_DIM

    model_name = get_vector_store_embedding_model_name().lower()
    return EMBEDDING_DIMENSIONS.get(model_name, 1536)
