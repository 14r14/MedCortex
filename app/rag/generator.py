import re
from typing import List

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

from app.config import Settings


SYSTEM_PROMPT = (
    "You are a research assistant providing detailed, comprehensive answers to researchers. "
    "Answer only using the provided context. If the answer is not in the context, say you don't know. "
    "Provide thorough, detailed information including specific details, numbers, names, dates, methodologies, "
    "and nuanced explanations. Do not oversimplify - researchers need complete and accurate information."
)


class GeneratorClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        credentials = Credentials(
            api_key=settings.ibm_cloud_api_key,
            url=f"https://{settings.watsonx_region}.ml.cloud.ibm.com",
        )
        self.client = ModelInference(
            model_id=settings.watsonx_gen_model,
            project_id=settings.watsonx_project_id,
            credentials=credentials,
        )

    def build_prompt(self, question: str, contexts: List[str]) -> str:
        # Use full context as provided
        joined = "\n\n".join(contexts)
        return (
            f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n\nContext:\n{joined}\n\n"
            "Provide a comprehensive, detailed answer suitable for researchers. Include specific details, "
            "exact figures, methodologies, findings, limitations, and relevant nuances from the context. "
            "Do not oversimplify - aim for thoroughness and precision. "
            "Provide only your answer directly without repeating the question, context, or any labels like 'Answer:' or 'Source:'. "
            "Just provide the answer text itself."
        )

    def build_compression_prompt(self, question: str, contexts: List[str]) -> str:
        joined = "\n\n".join(contexts)
        return (
            "Compress the following context into a comprehensive summary relevant to the question. "
            "Retain all important details: names, figures, dates, methodologies, findings, statistics, "
            "limitations, and nuanced claims. Preserve specificity - do not oversimplify. "
            "Do not speculate.\n\n"
            f"Question: {question}\n\nContext:\n{joined}\n\n"
            "Return only the compressed summary with all relevant details preserved."
        )

    def clean_output(self, text: str) -> str:
        """Remove prompt artifacts and structure labels from model output."""
        cleaned = text
        
        # Split by common prompt patterns to extract just the actual answers
        # Pattern: "Answer: ..." followed by other prompt elements
        parts = re.split(r"Answer:\s*", cleaned, flags=re.IGNORECASE)
        if len(parts) > 1:
            # Take the last "Answer:" section as it's likely the final answer
            cleaned = parts[-1]
            # Remove everything after the next "Source:" or "Question:" or "Context:" label
            cleaned = re.split(r"Source:\s*|Question:\s*|Context:\s*", cleaned, flags=re.IGNORECASE)[0]
        
        # Remove "Source: Context" lines (standalone or inline)
        cleaned = re.sub(r"^Source:\s*Context\s*$", "", cleaned, flags=re.MULTILINE | re.IGNORECASE)
        cleaned = re.sub(r"Source:\s*Context\s*", "", cleaned, flags=re.IGNORECASE)
        
        # Remove "Question:" sections and everything after them
        cleaned = re.sub(r"Question:\s*.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove "Context:" sections (standalone labels, not content)
        cleaned = re.sub(r"^Context:\s*$", "", cleaned, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove instruction text
        cleaned = re.sub(r"Provide a concise, complete answer and cite sources succinctly\.\s*", "", cleaned, flags=re.IGNORECASE)
        
        # Remove system prompt if present
        cleaned = re.sub(r"You are a helpful assistant.*?say you don't know\.\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove any repeated prompt structures (catch-all for repeated patterns)
        # Remove if we see "Answer: ... Source: Context" pattern repeated
        cleaned = re.sub(r"(Answer:.*?Source:\s*Context\s*){2,}", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove multiple consecutive newlines
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        
        # Final cleanup: remove leading/trailing whitespace and normalize
        cleaned = cleaned.strip()
        
        return cleaned

    def generate(self, question: str, contexts: List[str], temperature: float = 0.2) -> str:
        prompt = self.build_prompt(question, contexts)
        params = {
            GenParams.TEMPERATURE: float(temperature),
            GenParams.MAX_NEW_TOKENS: 4096,  # Increased for detailed research answers
            GenParams.TRUNCATE_INPUT_TOKENS: 0,
            GenParams.RETURN_OPTIONS: {"input_tokens": True, "generated_tokens": True},
        }

        # Prefer official streaming API if available
        raw_answer = ""
        try:
            stream_resp = self.client.generate_text_stream(prompt=prompt, params=params)
            text_parts: List[str] = []
            for chunk in stream_resp:  # type: ignore[assignment]
                text_parts.append(str(chunk))
            raw_answer = "".join(text_parts).strip()
        except Exception:
            # streaming not supported or failed; fall back below
            pass

        if not raw_answer:
            # Fallback: non-stream call
            response = self.client.generate(prompt=prompt, params=params)
            data = response.get_result() if hasattr(response, "get_result") else response
            if isinstance(data, str):
                raw_answer = data
            elif isinstance(data, dict):
                if "results" in data and data["results"]:
                    raw_answer = data["results"][0].get("generated_text", "")
                elif "generated_text" in data:
                    raw_answer = data["generated_text"]
                else:
                    raw_answer = str(data)
            elif hasattr(response, "generated_text"):
                raw_answer = response.generated_text  # type: ignore[attr-defined]
            else:
                raw_answer = str(data)
        
        # Clean the output to remove prompt artifacts
        return self.clean_output(raw_answer)

    def compress_context(self, question: str, contexts: List[str], temperature: float = 0.0) -> str:
        prompt = self.build_compression_prompt(question, contexts)
        params = {
            GenParams.TEMPERATURE: float(temperature),
            GenParams.MAX_NEW_TOKENS: 2048,
            GenParams.TRUNCATE_INPUT_TOKENS: 0,
            GenParams.RETURN_OPTIONS: {"input_tokens": True},
        }
        # Prefer streaming compression (accumulate chunks)
        try:
            chunks = self.client.generate_text_stream(prompt=prompt, params=params)
            parts: List[str] = []
            for ch in chunks:
                parts.append(str(ch))
            text = "".join(parts).strip()
            if text:
                return text
        except Exception:
            pass

        resp = self.client.generate(prompt=prompt, params=params)
        data = resp.get_result() if hasattr(resp, "get_result") else resp
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            if "results" in data and data["results"]:
                return data["results"][0].get("generated_text", "")
            if "generated_text" in data:
                return data["generated_text"]
            return str(data)
        if hasattr(resp, "generated_text"):
            return resp.generated_text  # type: ignore[attr-defined]
        return str(data)


