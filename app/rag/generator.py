import re

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

from app.config import Settings

SYSTEM_PROMPT = (
    "You are an expert medical research assistant providing highly detailed, comprehensive answers to medical and clinical researchers. "
    "Your audience consists of medical professionals, researchers, and scientists who need precise, thorough information. "
    "Answer only using the provided context. If the answer is not in the context, explicitly state that you don't know. "
    "For medical research queries, provide: "
    "- Specific quantitative data: exact figures, percentages, p-values, confidence intervals, sample sizes, statistical significance levels "
    "- Detailed methodologies: study designs, protocols, inclusion/exclusion criteria, data collection methods "
    "- Clinical findings: patient outcomes, efficacy rates, side effects, safety profiles, dosing information "
    "- Comparative analyses: treatment comparisons, protocol differences, efficacy across populations "
    "- Contextual nuances: limitations, confounding factors, study constraints, generalizability concerns "
    "- Technical terminology: Use precise medical and scientific terminology appropriate for researchers "
    "- Comprehensive scope: Address all aspects of the question, including related considerations, implications, and applications "
    "Do not oversimplify or summarize. Medical researchers need complete, accurate, and detailed information with all relevant specifics. "
    "Be thorough and precise. Include all relevant details, figures, methodologies, findings, and nuances from the source material. "
    "Do NOT include placeholder citations like [Source 1], [Source 2], [Table Data], etc. in your answer. "
    "Sources will be listed separately - just provide the answer text itself."
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

    def build_prompt(self, question: str, contexts: list[str]) -> str:
        # Use full context as provided
        joined = "\n\n".join(contexts)
        return (
            f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n\nContext:\n{joined}\n\n"
            "Provide a comprehensive, highly detailed answer tailored for medical researchers. "
            "Include ALL relevant specifics: exact quantitative data (percentages, p-values, confidence intervals, sample sizes), "
            "detailed methodologies (study designs, protocols, procedures), clinical findings (outcomes, efficacy, safety profiles), "
            "comparative analyses, limitations, contextual nuances, and technical medical terminology. "
            "Do not oversimplify or summarize - medical researchers need complete information with all details and subtleties. "
            "Be thorough, precise, and comprehensive. Address all aspects of the question and related considerations. "
            "Use precise medical and scientific terminology appropriate for the research community. "
            "Provide only your answer directly without repeating the question, context, or any labels like 'Answer:' or 'Source:'. "
            "Do NOT include placeholder citations like [Source 1], [Source 2], [Table Data], (Source 1, Source 2), etc. "
            "Just provide the detailed answer text itself - sources will be listed separately."
        )

    def build_compression_prompt(self, question: str, contexts: list[str]) -> str:
        joined = "\n\n".join(contexts)
        return (
            "Compress the following context into a comprehensive summary relevant to this medical research question. "
            "Retain ALL critical details essential for medical researchers: "
            "exact quantitative data (percentages, p-values, confidence intervals, sample sizes, statistical significance), "
            "detailed methodologies (study designs, protocols, procedures, inclusion/exclusion criteria), "
            "clinical findings (patient outcomes, efficacy rates, side effects, safety profiles, dosing), "
            "comparative analyses, limitations, contextual nuances, and technical medical terminology. "
            "Preserve all specificity - do not oversimplify or summarize. Medical researchers need complete information. "
            "Do not speculate.\n\n"
            f"Question: {question}\n\nContext:\n{joined}\n\n"
            "Return only the compressed summary with all relevant details preserved for medical research analysis."
        )

    def clean_output(self, text: str) -> str:
        """Remove prompt artifacts and structure labels from model output."""
        cleaned = text

        # Remove placeholder source citations like [Source 1], [Source 2], etc.
        cleaned = re.sub(r"\[Source\s+\d+\]", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\[Table\s+Data\]", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"\(Source\s+\d+(?:,\s*Source\s+\d+)*\)", "", cleaned, flags=re.IGNORECASE
        )

        # Remove evidence citations like (Evidence 1), (Evidence 2), etc.
        cleaned = re.sub(
            r"\(Evidence\s+\d+(?:,\s*Evidence\s+\d+)*\)",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\[Evidence\s+\d+\]", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bEvidence\s+\d+\b", "", cleaned, flags=re.IGNORECASE)

        # Remove part references like (Part 1), (Part 2), etc.
        cleaned = re.sub(
            r"\(Part\s+\d+(?:,\s*Part\s+\d+)*\)", "", cleaned, flags=re.IGNORECASE
        )
        cleaned = re.sub(r"\[Part\s+\d+\]", "", cleaned, flags=re.IGNORECASE)

        # Split by common prompt patterns to extract just the actual answers
        # Pattern: "Answer: ..." followed by other prompt elements
        parts = re.split(r"Answer:\s*", cleaned, flags=re.IGNORECASE)
        if len(parts) > 1:
            # Take the last "Answer:" section as it's likely the final answer
            cleaned = parts[-1]
            # Remove everything after the next "Source:" or "Question:" or "Context:" label
            cleaned = re.split(
                r"Source:\s*|Question:\s*|Context:\s*", cleaned, flags=re.IGNORECASE
            )[0]

        # Remove "Source: Context" lines (standalone or inline)
        cleaned = re.sub(
            r"^Source:\s*Context\s*$", "", cleaned, flags=re.MULTILINE | re.IGNORECASE
        )
        cleaned = re.sub(r"Source:\s*Context\s*", "", cleaned, flags=re.IGNORECASE)

        # Remove "Sources:" section at the end if present
        cleaned = re.sub(
            r"\n\s*Sources?\s*:.*$",
            "",
            cleaned,
            flags=re.MULTILINE | re.IGNORECASE | re.DOTALL,
        )

        # Remove meta-commentary paragraphs that describe what the answer includes
        meta_patterns = [
            r"This\s+(?:synthesized\s+)?answer\s+(?:integrates|includes|provides|addresses).*?(?:\n\n|\Z)",
            r"This\s+(?:response|answer)\s+(?:integrates|includes|provides|addresses).*?(?:\n\n|\Z)",
            r"The\s+(?:above\s+)?answer\s+(?:integrates|includes|provides|addresses).*?(?:\n\n|\Z)",
            r"Note:\s*This\s+answer.*?(?:\n\n|\Z)",
            r"In\s+summary,\s*this\s+answer.*?(?:\n\n|\Z)",
        ]
        for pattern in meta_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)

        # Remove paragraphs that start with meta-descriptions
        cleaned = re.sub(
            r"\n\n(?:This|The|Note:|In summary,).*?(?:integrates|includes|provides|addresses|combines|synthesizes).*?(?:information|evidence|sources|data|findings).*?(?:all aspects|comprehensive|detailed|thorough|complete).*?\n\n",
            "\n\n",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Remove "Question:" sections and everything after them
        cleaned = re.sub(
            r"Question:\s*.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE
        )

        # Remove "Context:" sections (standalone labels, not content)
        cleaned = re.sub(
            r"^Context:\s*$", "", cleaned, flags=re.MULTILINE | re.IGNORECASE
        )

        # Remove instruction text
        cleaned = re.sub(
            r"Provide a concise, complete answer and cite sources succinctly\.\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        # Remove system prompt if present
        cleaned = re.sub(
            r"You are a helpful assistant.*?say you don't know\.\s*",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Remove any repeated prompt structures (catch-all for repeated patterns)
        # Remove if we see "Answer: ... Source: Context" pattern repeated
        cleaned = re.sub(
            r"(Answer:.*?Source:\s*Context\s*){2,}",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Remove multiple consecutive newlines
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        # Final cleanup: remove leading/trailing whitespace and normalize
        cleaned = cleaned.strip()

        return cleaned

    def generate(
        self, question: str, contexts: list[str], temperature: float = 0.2
    ) -> str:
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
            text_parts: list[str] = []
            for chunk in stream_resp:  # type: ignore[assignment]
                text_parts.append(str(chunk))
            raw_answer = "".join(text_parts).strip()
        except Exception:
            # streaming not supported or failed; fall back below
            pass

        if not raw_answer:
            # Fallback: non-stream call
            response = self.client.generate(prompt=prompt, params=params)
            data = (
                response.get_result() if hasattr(response, "get_result") else response
            )
            if isinstance(data, str):
                raw_answer = data
            elif isinstance(data, dict):
                if data.get("results"):
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

    def generate_from_prompt(self, prompt: str, temperature: float = 0.2) -> str:
        """
        Generate text directly from a raw prompt string.

        Useful for special cases like query decomposition, synthesis, etc.
        that need custom prompt formatting.
        """
        params = {
            GenParams.TEMPERATURE: float(temperature),
            GenParams.MAX_NEW_TOKENS: 4096,
            GenParams.TRUNCATE_INPUT_TOKENS: 0,
            GenParams.RETURN_OPTIONS: {"input_tokens": True, "generated_tokens": True},
        }

        raw_answer = ""
        try:
            stream_resp = self.client.generate_text_stream(prompt=prompt, params=params)
            text_parts: list[str] = []
            for chunk in stream_resp:
                text_parts.append(str(chunk))
            raw_answer = "".join(text_parts).strip()
        except Exception:
            pass

        if not raw_answer:
            response = self.client.generate(prompt=prompt, params=params)
            data = (
                response.get_result() if hasattr(response, "get_result") else response
            )
            if isinstance(data, str):
                raw_answer = data
            elif isinstance(data, dict):
                if data.get("results"):
                    raw_answer = data["results"][0].get("generated_text", "")
                elif "generated_text" in data:
                    raw_answer = data["generated_text"]
                else:
                    raw_answer = str(data)
            elif hasattr(response, "generated_text"):
                raw_answer = response.generated_text
            else:
                raw_answer = str(data)

        return self.clean_output(raw_answer)

    def compress_context(
        self, question: str, contexts: list[str], temperature: float = 0.0
    ) -> str:
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
            parts: list[str] = []
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
            if data.get("results"):
                return data["results"][0].get("generated_text", "")
            if "generated_text" in data:
                return data["generated_text"]
            return str(data)
        if hasattr(resp, "generated_text"):
            return resp.generated_text  # type: ignore[attr-defined]
        return str(data)
