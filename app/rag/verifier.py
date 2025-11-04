"""Post-generation answer verification using Natural Language Inference (NLI)."""

import logging
import re

from app.config import Settings
from app.rag.generator import GeneratorClient

logger = logging.getLogger(__name__)


class AnswerVerifier:
    """
    Verifies factual claims in generated answers against cited source chunks.

    Uses Natural Language Inference (NLI) to check if each claim is supported,
    refuted, or not mentioned in the source material.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.gen = GeneratorClient(settings)

    def deconstruct_claims(self, answer: str) -> list[str]:
        """
        Deconstruct answer into individual factual claims.

        Splits answer into sentences and filters for factual statements
        (containing numbers, statistics, specific findings, etc.).

        Args:
            answer: Generated answer text

        Returns:
            List of individual claims (sentences/phrases)
        """
        # Split by sentence boundaries (period, semicolon, newline)
        sentences = re.split(r"[.;]\s+|\n+", answer)

        # Filter and clean claims
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()

            # Skip very short sentences
            if len(sentence) < 20:
                continue

            # Skip questions, meta-commentary, and instructions
            if any(
                skip in sentence.lower()
                for skip in [
                    "this answer",
                    "this response",
                    "the above",
                    "note:",
                    "in summary",
                    "question:",
                    "answer:",
                    "source:",
                    "evidence",
                    "part ",
                ]
            ):
                continue

            # Prioritize claims with quantitative data (numbers, percentages, p-values, etc.)
            has_quantitative = bool(
                re.search(
                    r"\d+[.%]|\bp\s*[<>=]\s*\d|confidence|interval|sample\s*size",
                    sentence,
                    re.IGNORECASE,
                )
            )

            # Include claims with specific findings, outcomes, or results
            has_finding = any(
                term in sentence.lower()
                for term in [
                    "showed",
                    "found",
                    "demonstrated",
                    "indicated",
                    "revealed",
                    "result",
                    "outcome",
                    "efficacy",
                    "safety",
                    "response",
                    "rate",
                    "percentage",
                    "improvement",
                    "reduction",
                    "increase",
                ]
            )

            # Include if it's a factual statement (not just descriptive)
            if has_quantitative or has_finding or len(sentence) > 50:
                claims.append(sentence)

        # Return all claims for verification (no limit)
        return claims

    def verify_claim(self, claim: str, source_chunks: list[str]) -> dict[str, str]:
        """
        Verify a single claim against source chunks using NLI.

        Args:
            claim: Factual claim to verify
            source_chunks: List of source chunk texts

        Returns:
            Dict with 'status' ('Supports', 'Refutes', 'Not Mentioned') and 'chunk_index'
        """
        if not source_chunks:
            return {"status": "Not Mentioned", "chunk_index": -1, "confidence": 0.0}

        # Check against each source chunk
        best_status = "Not Mentioned"
        best_chunk_idx = -1

        for idx, chunk in enumerate(source_chunks):
            # Limit chunk length for NLI (use first 500 chars)
            chunk_preview = chunk[:500] if len(chunk) > 500 else chunk

            prompt = f"""Given the source text, does it support the following claim? Answer only 'Supports', 'Refutes', or 'Not Mentioned'.

Source: {chunk_preview}

Claim: {claim}

Answer (only 'Supports', 'Refutes', or 'Not Mentioned'):"""

            try:
                response = self.gen.generate_from_prompt(prompt=prompt, temperature=0.0)
                response = response.strip().upper()

                # Parse response
                if "SUPPORTS" in response or response.startswith("SUPPORT"):
                    if best_status != "Supports":
                        best_status = "Supports"
                        best_chunk_idx = idx
                    # Supports is the best status, keep checking for better match
                    break  # Found support, can stop
                if "REFUTES" in response or response.startswith("REFUTE"):
                    if best_status == "Not Mentioned":
                        best_status = "Refutes"
                        best_chunk_idx = idx
                    # Continue checking - might find Supports later
                # else: Not Mentioned, continue

            except Exception as e:
                logger.warning(
                    f"Verification check failed for claim '{claim[:50]}...': {e}"
                )
                continue

        return {
            "status": best_status,
            "chunk_index": best_chunk_idx,
            "confidence": 1.0 if best_status == "Supports" else 0.5,
        }

    def verify_answer(
        self, answer: str, source_chunks: list[str]
    ) -> list[dict[str, any]]:
        """
        Verify all claims in an answer against source chunks.

        Args:
            answer: Generated answer text
            source_chunks: List of source chunk texts from retrieval

        Returns:
            List of verification results, each with:
            - claim: The factual claim
            - status: 'Supports', 'Refutes', or 'Not Mentioned'
            - chunk_index: Index of supporting/refuting chunk (-1 if not found)
            - answer_position: Position in original answer (for annotation)
        """
        if not source_chunks:
            logger.warning("No source chunks provided for verification")
            return []

        claims = self.deconstruct_claims(answer)

        if not claims:
            logger.info("No claims extracted from answer for verification")
            return []

        logger.info(
            f"Verifying {len(claims)} claims against {len(source_chunks)} source chunks"
        )

        verification_results = []

        for claim in claims:
            result = self.verify_claim(claim, source_chunks)
            verification_results.append(
                {
                    "claim": claim,
                    "status": result["status"],
                    "chunk_index": result["chunk_index"],
                    "confidence": result.get("confidence", 0.5),
                }
            )

        return verification_results

    def annotate_answer(
        self, answer: str, verification_results: list[dict[str, any]]
    ) -> str:
        """
        Annotate answer with verification status markers.

        Args:
            answer: Original answer text
            verification_results: List of verification results

        Returns:
            Annotated answer with verification markers (for display purposes)
        """
        if not verification_results:
            return answer

        annotated = answer

        # Create a mapping of claims to status
        claim_to_status = {r["claim"]: r["status"] for r in verification_results}

        # For each claim in results, add marker if found in answer
        for result in verification_results:
            claim = result["claim"]
            status = result["status"]

            # Find claim in answer (approximate match)
            claim_clean = re.escape(claim[:50])  # Use first 50 chars for matching
            pattern = f"({claim_clean}.*?)"

            if re.search(pattern, annotated, re.IGNORECASE | re.DOTALL):
                # Add verification marker based on status
                if status == "Supports":
                    marker = " ✓"
                elif status == "Refutes" or status == "Not Mentioned":
                    marker = " ⚠️"
                else:
                    marker = ""

                # Only add marker once per unique claim
                if marker and marker not in annotated:
                    annotated = re.sub(
                        pattern,
                        f"\\1{marker}",
                        annotated,
                        count=1,
                        flags=re.IGNORECASE | re.DOTALL,
                    )

        return annotated
