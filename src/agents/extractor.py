"""Extraction router with escalation guard logic.

This module implements the ExtractionRouter that orchestrates document extraction
using multiple strategies with automatic escalation based on confidence scores.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.models.document_profile import DocumentProfile
from src.models.extracted_document import ExtractedDocument
from src.strategies.base import ExtractionStrategy
from src.strategies.fast_text import FastTextExtractor
from src.strategies.layout_aware import LayoutExtractor
from src.strategies.layout_mineru import MinerUExtractor
from src.strategies.router import select_layout_strategy
from src.strategies.vision_augmented import VisionExtractor

logger = logging.getLogger(__name__)


class ExtractionRouter:
    """Router for document extraction with escalation guard.

    The router selects an initial extraction strategy based on the DocumentProfile's
    estimated_cost, then implements an escalation guard that automatically upgrades
    to more sophisticated (and expensive) strategies if confidence scores fall
    below a threshold.

    Escalation path:
        Strategy A (FastText) → Strategy B (Layout-aware) → Strategy C (Vision)

    All extraction attempts are logged to extraction_ledger.jsonl for auditing
    and analysis.

    Attributes:
        confidence_threshold: Minimum confidence score to accept a strategy (default: 0.74).
        ledger_path: Path to extraction_ledger.jsonl file.
        enable_parallel: Whether to enable parallel processing for multi-page documents.
        max_workers: Maximum number of parallel workers (if parallel enabled).
    """

    def __init__(
        self,
        confidence_threshold: float = 0.74,
        ledger_path: Optional[Path] = None,
        enable_parallel: bool = False,
        max_workers: int = 4,
    ) -> None:
        """Initialize the extraction router.

        Args:
            confidence_threshold: Minimum confidence score to accept a strategy.
            ledger_path: Path to extraction_ledger.jsonl. Defaults to .refinery/extraction_ledger.jsonl.
            enable_parallel: Enable parallel processing for speed.
            max_workers: Maximum number of parallel workers.
        """
        self.confidence_threshold = confidence_threshold
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers

        # Set up ledger path
        if ledger_path is None:
            ledger_path = Path(".refinery") / "extraction_ledger.jsonl"
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize strategies
        self.strategy_a = FastTextExtractor()
        self.strategy_b_layout = LayoutExtractor()
        self.strategy_b_mineru = MinerUExtractor()
        self.strategy_c = VisionExtractor()

        logger.info(
            "ExtractionRouter initialized",
            extra={
                "confidence_threshold": confidence_threshold,
                "ledger_path": str(self.ledger_path),
                "parallel_enabled": enable_parallel,
            },
        )

    def extract(
        self, profile: DocumentProfile, document_path: str
    ) -> ExtractedDocument:
        """Extract content from a document using escalation guard logic.

        The router:
        1. Selects an initial strategy based on profile.estimated_cost
        2. Attempts extraction with that strategy
        3. Checks confidence score
        4. Escalates to next strategy if confidence < threshold
        5. Logs all attempts to extraction_ledger.jsonl

        Args:
            profile: DocumentProfile from the triage agent.
            document_path: Path to the document file.

        Returns:
            ExtractedDocument with extracted content.

        Raises:
            FileNotFoundError: If the document file does not exist.
            RuntimeError: If all strategies fail or confidence remains below threshold.
        """
        doc_path = Path(document_path).resolve()
        if not doc_path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")

        # Determine initial strategy based on estimated_cost
        initial_strategy = self._select_initial_strategy(profile)
        escalation_path: List[str] = [initial_strategy.name]

        logger.info(
            "Starting extraction",
            extra={
                "document_id": profile.doc_id,
                "document_path": str(doc_path),
                "initial_strategy": initial_strategy.name,
                "estimated_cost": profile.estimated_cost,
            },
        )

        # Try strategies in escalation order
        strategies_to_try = self._get_escalation_sequence(initial_strategy)

        for strategy in strategies_to_try:
            start_time = time.time()
            try:
                # Perform extraction
                if self.enable_parallel:
                    extracted_doc = self._extract_parallel(strategy, str(doc_path))
                else:
                    extracted_doc = strategy.extract(str(doc_path))

                processing_time = time.time() - start_time

                # Calculate confidence score
                confidence = strategy.confidence_score(str(doc_path))
                cost_estimate = strategy.cost_estimate(str(doc_path))

                # Log extraction attempt for each page
                self._log_extraction(
                    profile=profile,
                    document_path=str(doc_path),
                    strategy_used=strategy.name,
                    confidence_score=confidence,
                    cost_estimate=cost_estimate,
                    processing_time=processing_time,
                    escalation_path=escalation_path.copy(),
                    success=True,
                )

                # Check if confidence meets threshold
                if confidence >= self.confidence_threshold:
                    logger.info(
                        "Extraction successful with acceptable confidence",
                        extra={
                            "document_id": profile.doc_id,
                            "strategy": strategy.name,
                            "confidence": confidence,
                            "threshold": self.confidence_threshold,
                        },
                    )
                    return extracted_doc
                else:
                    logger.warning(
                        "Confidence below threshold, escalating",
                        extra={
                            "document_id": profile.doc_id,
                            "strategy": strategy.name,
                            "confidence": confidence,
                            "threshold": self.confidence_threshold,
                        },
                    )

            except Exception as exc:
                processing_time = time.time() - start_time
                try:
                    cost_estimate = strategy.cost_estimate(str(doc_path))
                except Exception:
                    cost_estimate = {"total_cost_usd": 0.0, "cost_per_page": 0.0}

                logger.exception(
                    "Extraction failed",
                    extra={
                        "document_id": profile.doc_id,
                        "strategy": strategy.name,
                        "error": str(exc),
                    },
                )

                # Log failed attempt
                self._log_extraction(
                    profile=profile,
                    document_path=str(doc_path),
                    strategy_used=strategy.name,
                    confidence_score=0.0,
                    cost_estimate=cost_estimate,
                    processing_time=processing_time,
                    escalation_path=escalation_path.copy(),
                    success=False,
                    error=str(exc),
                )

                # Continue to next strategy
                continue

            # Add next strategy to escalation path
            next_strategy = self._get_next_strategy(strategy)
            if next_strategy:
                escalation_path.append(next_strategy.name)

        # If we reach here, all strategies failed or had low confidence
        raise RuntimeError(
            f"All extraction strategies exhausted for {profile.doc_id}. "
            f"Escalation path: {' -> '.join(escalation_path)}"
        )

    def _select_initial_strategy(
        self, profile: DocumentProfile
    ) -> ExtractionStrategy:
        """Select initial strategy based on profile.estimated_cost.

        Args:
            profile: DocumentProfile from triage agent.

        Returns:
            Initial ExtractionStrategy to try.
        """
        if profile.estimated_cost == "fast_text_sufficient":
            return self.strategy_a
        elif profile.estimated_cost == "needs_layout_model":
            # Use router to select between LayoutExtractor and MinerUExtractor
            layout_strategy = select_layout_strategy(profile)
            if layout_strategy:
                return layout_strategy
            # Fallback to LayoutExtractor if router returns None
            return self.strategy_b_layout
        elif profile.estimated_cost == "needs_vision_model":
            return self.strategy_c
        else:
            # Default to fast text for unknown cost estimates
            logger.warning(
                "Unknown estimated_cost, defaulting to fast text",
                extra={"estimated_cost": profile.estimated_cost},
            )
            return self.strategy_a

    def _get_escalation_sequence(
        self, initial_strategy: ExtractionStrategy
    ) -> List[ExtractionStrategy]:
        """Get the escalation sequence starting from initial strategy.

        Args:
            initial_strategy: The strategy to start with.

        Returns:
            List of strategies to try in order.
        """
        sequence: List[ExtractionStrategy] = []

        # Build sequence based on initial strategy
        if initial_strategy.name == "fast_text":
            sequence = [
                self.strategy_a,
                self.strategy_b_layout,  # Try layout-aware next
                self.strategy_c,  # Finally vision
            ]
        elif initial_strategy.name in ("layout_aware", "layout_mineru"):
            sequence = [
                initial_strategy,  # Try the selected layout strategy first
                self.strategy_c,  # Then vision
            ]
        elif initial_strategy.name == "vision_augmented":
            sequence = [self.strategy_c]  # Vision is the final strategy
        else:
            # Fallback: try all strategies in order
            sequence = [
                self.strategy_a,
                self.strategy_b_layout,
                self.strategy_c,
            ]

        return sequence

    def _get_next_strategy(
        self, current_strategy: ExtractionStrategy
    ) -> Optional[ExtractionStrategy]:
        """Get the next strategy in escalation sequence.

        Args:
            current_strategy: Current strategy that was tried.

        Returns:
            Next strategy to try, or None if no more strategies.
        """
        if current_strategy.name == "fast_text":
            return self.strategy_b_layout
        elif current_strategy.name in ("layout_aware", "layout_mineru"):
            return self.strategy_c
        elif current_strategy.name == "vision_augmented":
            return None  # Vision is the final strategy
        else:
            return None

    def _extract_parallel(
        self, strategy: ExtractionStrategy, document_path: str
    ) -> ExtractedDocument:
        """Extract document using parallel processing for speed.

        This method processes pages in parallel when possible. Note that not all
        strategies support true parallel processing, so this is a best-effort
        optimization.

        Args:
            strategy: Extraction strategy to use.
            document_path: Path to the document.

        Returns:
            ExtractedDocument with extracted content.
        """
        # For now, most strategies don't support true parallel page processing,
        # so we fall back to sequential extraction. This is a placeholder for
        # future optimization where strategies could process pages in parallel.
        logger.debug(
            "Parallel extraction requested, falling back to sequential",
            extra={"strategy": strategy.name},
        )
        return strategy.extract(document_path)

    def _log_extraction(
        self,
        profile: DocumentProfile,
        document_path: str,
        strategy_used: str,
        confidence_score: float,
        cost_estimate: Dict[str, float],
        processing_time: float,
        escalation_path: List[str],
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Log extraction attempt to extraction_ledger.jsonl.

        Logs one entry per page to provide granular tracking. Each page gets
        the same document-level metrics (confidence, cost, processing time).

        Args:
            profile: DocumentProfile from triage agent.
            document_path: Path to the document.
            strategy_used: Name of the strategy used.
            confidence_score: Confidence score from the strategy.
            cost_estimate: Cost estimate dictionary.
            processing_time: Time taken for extraction in seconds.
            escalation_path: List of strategies tried in order.
            success: Whether extraction was successful.
            error: Error message if extraction failed.
        """
        # Get page count from profile metadata
        page_count = profile.metadata.page_count

        # Log one entry per page as requested
        for page_num in range(1, page_count + 1):
            log_entry = {
                "document_id": profile.doc_id,
                "page_num": page_num,
                "document_path": document_path,
                "strategy_used": strategy_used,
                "confidence_score": round(confidence_score, 4),
                "cost_estimate": {
                    "total_cost_usd": round(cost_estimate.get("total_cost_usd", 0.0), 6),
                    "cost_per_page": round(
                        cost_estimate.get("cost_per_page", 0.0), 6
                    ),
                },
                "processing_time_seconds": round(processing_time, 3),
                "escalation_path": escalation_path,
                "success": success,
                "timestamp": time.time(),
            }

            if error:
                log_entry["error"] = error

            # Append to ledger file (JSONL format)
            try:
                with self.ledger_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception as exc:
                logger.error(
                    "Failed to write to extraction ledger",
                    extra={"ledger_path": str(self.ledger_path), "error": str(exc)},
                )

    def extract_batch(
        self,
        profiles_and_paths: List[Tuple[DocumentProfile, str]],
    ) -> List[Tuple[DocumentProfile, ExtractedDocument]]:
        """Extract multiple documents in parallel.

        Args:
            profiles_and_paths: List of (DocumentProfile, document_path) tuples.

        Returns:
            List of (DocumentProfile, ExtractedDocument) tuples.
        """
        results: List[Tuple[DocumentProfile, ExtractedDocument]] = []

        if self.enable_parallel and len(profiles_and_paths) > 1:
            # Process documents in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_profile = {
                    executor.submit(self.extract, profile, path): profile
                    for profile, path in profiles_and_paths
                }

                for future in as_completed(future_to_profile):
                    profile = future_to_profile[future]
                    try:
                        extracted_doc = future.result()
                        results.append((profile, extracted_doc))
                    except Exception as exc:
                        logger.exception(
                            "Batch extraction failed for document",
                            extra={"document_id": profile.doc_id, "error": str(exc)},
                        )
        else:
            # Process sequentially
            for profile, path in profiles_and_paths:
                try:
                    extracted_doc = self.extract(profile, path)
                    results.append((profile, extracted_doc))
                except Exception as exc:
                    logger.exception(
                        "Batch extraction failed for document",
                        extra={"document_id": profile.doc_id, "error": str(exc)},
                    )

        return results


__all__ = ["ExtractionRouter"]
