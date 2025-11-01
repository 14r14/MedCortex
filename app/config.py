import os
from dataclasses import dataclass


@dataclass
class Settings:
    ibm_cloud_api_key: str
    watsonx_region: str
    watsonx_project_id: str
    watsonx_embed_model: str
    watsonx_gen_model: str

    cos_endpoint: str
    cos_bucket: str
    cos_instance_crn: str
    cos_api_key: str | None
    cos_auth_endpoint: str
    cos_hmac_access_key_id: str
    cos_hmac_secret_access_key: str

    milvus_host: str
    milvus_port: int
    milvus_db: str | None
    milvus_tls: bool

    faiss_index_path: str
    faiss_meta_path: str

    chunk_size: int
    chunk_overlap: int
    top_k: int
    temperature: float
    embedding_dim: int

    @staticmethod
    def _get_bool(value: str | None, default: bool = False) -> bool:
        if value is None:
            return default
        return value.lower() in {"1", "true", "t", "yes", "y"}

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            ibm_cloud_api_key=os.getenv("IBM_CLOUD_API_KEY", ""),
            watsonx_region=os.getenv("WATSONX_REGION", "us-south"),
            watsonx_project_id=os.getenv("WATSONX_PROJECT_ID", ""),
            watsonx_embed_model=os.getenv("WATSONX_EMBED_MODEL", "ibm/granite-embedding-30m-english"),
            watsonx_gen_model=os.getenv("WATSONX_GEN_MODEL", "ibm/granite-13b-instruct-v2"),
            cos_endpoint=os.getenv("COS_ENDPOINT", ""),
            cos_bucket=os.getenv("COS_BUCKET", ""),
            cos_instance_crn=os.getenv("COS_INSTANCE_CRN", ""),
            cos_api_key=os.getenv("COS_API_KEY") or os.getenv("IBM_CLOUD_API_KEY"),
            cos_auth_endpoint=os.getenv("COS_AUTH_ENDPOINT", "https://iam.cloud.ibm.com/identity/token"),
            cos_hmac_access_key_id=os.getenv("COS_HMAC_ACCESS_KEY_ID", ""),
            cos_hmac_secret_access_key=os.getenv("COS_HMAC_SECRET_ACCESS_KEY", ""),
            milvus_host=os.getenv("MILVUS_HOST", "localhost"),
            milvus_port=int(os.getenv("MILVUS_PORT", "19530")),
            milvus_db=os.getenv("MILVUS_DB"),
            milvus_tls=cls._get_bool(os.getenv("MILVUS_TLS"), False),
            faiss_index_path=os.getenv("FAISS_INDEX_PATH", "data/index.faiss"),
            faiss_meta_path=os.getenv("FAISS_META_PATH", "data/meta.json"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
            top_k=int(os.getenv("TOP_K", "6")),
            temperature=float(os.getenv("TEMPERATURE", "0.2")),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
        )


