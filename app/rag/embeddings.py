from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings as WXEmbeddings

from app.config import Settings


class EmbeddingClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        credentials = Credentials(
            api_key=settings.ibm_cloud_api_key,
            url=f"https://{settings.watsonx_region}.ml.cloud.ibm.com",
        )
        self.client = WXEmbeddings(
            model_id=settings.watsonx_embed_model,
            project_id=settings.watsonx_project_id,
            credentials=credentials,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        result = self.client.embed_documents(texts)
        data = result.get_result() if hasattr(result, "get_result") else result
        # Supported shapes
        # 1) {"results": [{"embedding"|"vector"|"values": [...]}, ...]}
        if (
            isinstance(data, dict)
            and "results" in data
            and isinstance(data["results"], list)
        ):
            out = []
            for item in data["results"]:
                if isinstance(item, dict):
                    if "embedding" in item:
                        out.append(item["embedding"])  # type: ignore[assignment]
                    elif "vector" in item:
                        out.append(item["vector"])  # type: ignore[assignment]
                    elif "values" in item:
                        out.append(item["values"])  # type: ignore[assignment]
            if out:
                return out
        # 2) {"embeddings": [[...], ...]}
        if isinstance(data, dict) and "embeddings" in data:
            return data["embeddings"]  # type: ignore[return-value]
        # 3) direct list of vectors
        if isinstance(data, list) and data and isinstance(data[0], list):
            return data  # type: ignore[return-value]
        # 4) attribute style
        if hasattr(result, "embeddings"):
            return result.embeddings  # type: ignore[attr-defined]
        raise RuntimeError(
            f"Unexpected embeddings response format from watsonx.ai: {type(data)} keys={list(data.keys()) if isinstance(data, dict) else 'n/a'}"
        )

    def embed_query(self, text: str) -> list[float]:
        result = self.client.embed_query(text)
        data = result.get_result() if hasattr(result, "get_result") else result
        if isinstance(data, dict):
            # {"results": [{"embedding"|"vector"|"values": [...]}, ...]}
            if (
                "results" in data
                and isinstance(data["results"], list)
                and data["results"]
            ):
                first = data["results"][0]
                if isinstance(first, dict):
                    if "embedding" in first:
                        return first["embedding"]  # type: ignore[return-value]
                    if "vector" in first:
                        return first["vector"]  # type: ignore[return-value]
                    if "values" in first:
                        return first["values"]  # type: ignore[return-value]
            if "embedding" in data:
                return data["embedding"]  # type: ignore[return-value]
            if data.get("embeddings"):
                return data["embeddings"][0]  # type: ignore[return-value]
        # list-shaped: either a single vector or list of vectors
        if isinstance(data, list):
            if data and isinstance(data[0], list):
                return data[0]  # type: ignore[return-value]
            # single vector as list of floats
            if data and isinstance(data[0], (int, float)):
                return data  # type: ignore[return-value]
        if hasattr(result, "embedding"):
            return result.embedding  # type: ignore[attr-defined]
        raise RuntimeError(
            f"Unexpected query embedding response format from watsonx.ai: {type(data)} keys={list(data.keys()) if isinstance(data, dict) else 'n/a'}"
        )
