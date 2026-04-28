"""
CAAAPT Human Review Module
Manual review delegation for low-confidence samples

As described in Section 3.2 and Equation (4):
If C_final < tau_back, the sample triggers manual review.
This combines LLM automation with human expertise to control
false positives and reduce labor costs.

Two-tier quality control mechanism:
- tau_front: Prioritizes efficiency (frontend screening)
- tau_back: Ensures trustworthiness (backend attribution)

Sanitized version for submission - No sensitive paths, API keys, or internal IPs
"""

import os
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ReviewStatus(Enum):
    """Status of a review request"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    EXPIRED = "expired"


class ReviewDecision(Enum):
    """Possible decisions from human review"""
    ACCEPT = "accept"  # Accept the attribution
    REJECT = "reject"  # Reject as false positive
    NEEDS_MORE_INFO = "needs_more_info"  # Insufficient evidence
    ESCALATE = "escalate"  # Escalate to senior analyst
    DEFER = "defer"  # Defer to later analysis


@dataclass
class ReviewRequest:
    """
    A sample that requires human review

    Triggered when C_final < tau_back (Equation 4)
    """
    request_id: str
    sample_id: str
    timestamp: str
    final_confidence: float
    tau_back_used: float
    confidence_gap: float  # tau_back - final_confidence

    # Original analysis results
    frontend_score: float
    backend_report: Dict[str, Any]
    stage_outputs: List[Dict[str, Any]]
    retrieval_context: Dict[str, Any]

    # Review metadata
    status: ReviewStatus = ReviewStatus.PENDING
    assigned_to: Optional[str] = None
    priority: int = 0  # 0=low, 1=medium, 2=high, 3=critical
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Review results (populated when completed)
    review_decision: Optional[ReviewDecision] = None
    review_notes: Optional[str] = None
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    override_confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "sample_id": self.sample_id,
            "timestamp": self.timestamp,
            "final_confidence": self.final_confidence,
            "tau_back_used": self.tau_back_used,
            "confidence_gap": self.confidence_gap,
            "frontend_score": self.frontend_score,
            "backend_report": self.backend_report,
            "stage_outputs": self.stage_outputs,
            "status": self.status.value,
            "assigned_to": self.assigned_to,
            "priority": self.priority,
            "created_at": self.created_at,
            "review_decision": self.review_decision.value if self.review_decision else None,
            "review_notes": self.review_notes,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at
        }

    def to_review_interface(self) -> str:
        """Generate human-readable review interface text"""
        lines = [
            "=" * 80,
            f"REVIEW REQUEST: {self.request_id}",
            "=" * 80,
            f"Sample ID: {self.sample_id}",
            f"Created: {self.created_at}",
            f"Confidence: {self.final_confidence:.4f} (Threshold: {self.tau_back_used})",
            f"Confidence Gap: {self.confidence_gap:.4f}",
            f"Priority: {['Low', 'Medium', 'High', 'Critical'][self.priority]}",
            "-" * 40,
            "FRONTEND SCORE:",
            f"  XGBoost Anomaly Score: {self.frontend_score:.4f}",
            "-" * 40,
            "BACKEND ATTRIBUTION REPORT:",
        ]

        # Add backend report summary
        if self.backend_report:
            for key, value in self.backend_report.items():
                if isinstance(value, str) and len(value) > 200:
                    value = value[:200] + "..."
                lines.append(f"  {key}: {value}")

        lines.extend([
            "-" * 40,
            "REVIEW DECISION OPTIONS:",
            "  1. ACCEPT - Confirm attribution is correct",
            "  2. REJECT - Label as false positive",
            "  3. NEEDS_MORE_INFO - Insufficient evidence for decision",
            "  4. ESCALATE - Escalate to senior analyst",
            "  5. DEFER - Defer to later analysis",
            "-" * 40,
            "Enter decision and notes:",
            "=" * 80
        ])

        return "\n".join(lines)


@dataclass
class ReviewStatistics:
    """Statistics for manual review operations"""
    total_requests: int = 0
    pending: int = 0
    in_progress: int = 0
    completed: int = 0
    escalated: int = 0
    expired: int = 0

    # Decision breakdown
    accepted: int = 0
    rejected: int = 0
    needs_more_info: int = 0
    escalated_count: int = 0
    deferred: int = 0

    # Performance metrics
    average_review_time_seconds: float = 0.0
    average_confidence_gap: float = 0.0

    # Feedback loop metrics
    false_positive_corrections: int = 0  # Rejected that were marked as anomalies
    false_negative_corrections: int = 0  # Accepted that were missed by frontend

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "pending": self.pending,
            "in_progress": self.in_progress,
            "completed": self.completed,
            "escalated": self.escalated,
            "expired": self.expired,
            "decisions": {
                "accepted": self.accepted,
                "rejected": self.rejected,
                "needs_more_info": self.needs_more_info,
                "escalated": self.escalated_count,
                "deferred": self.deferred
            },
            "performance": {
                "average_review_time_seconds": self.average_review_time_seconds,
                "average_confidence_gap": self.average_confidence_gap
            },
            "feedback_loop": {
                "false_positive_corrections": self.false_positive_corrections,
                "false_negative_corrections": self.false_negative_corrections
            }
        }


class ReviewQueue:
    """
    Queue manager for manual review requests

    Implements priority-based queuing and assignment tracking
    """

    def __init__(self, max_size: int = 10000, expiration_hours: int = 168):
        """
        Initialize review queue

        Args:
            max_size: Maximum number of pending requests
            expiration_hours: Hours until a request expires
        """
        self.max_size = max_size
        self.expiration_hours = expiration_hours
        self._requests: Dict[str, ReviewRequest] = {}
        self._queue: List[str] = []  # Ordered list of request IDs
        self._statistics = ReviewStatistics()

    def add_request(self, request: ReviewRequest) -> bool:
        """
        Add a review request to the queue

        Args:
            request: ReviewRequest to add

        Returns:
            True if added successfully
        """
        if len(self._requests) >= self.max_size:
            # Remove oldest expired requests
            self._clean_expired()

            if len(self._requests) >= self.max_size:
                print(f"[WARN] Review queue full, request {request.request_id} rejected")
                return False

        self._requests[request.request_id] = request
        self._queue.append(request.request_id)
        self._statistics.total_requests += 1
        self._statistics.pending += 1

        # Update average confidence gap
        total_gap = self._statistics.average_confidence_gap * (self._statistics.total_requests - 1)
        self._statistics.average_confidence_gap = (total_gap + request.confidence_gap) / self._statistics.total_requests

        return True

    def get_next_request(self, analyst_id: Optional[str] = None) -> Optional[ReviewRequest]:
        """
        Get the next pending request (highest priority, FIFO within priority)

        Args:
            analyst_id: Analyst to assign the request to

        Returns:
            ReviewRequest or None if queue empty
        """
        # Sort by priority (higher priority first), then by creation time
        pending_ids = [
            rid for rid, req in self._requests.items()
            if req.status == ReviewStatus.PENDING
        ]

        if not pending_ids:
            return None

        # Sort by priority (descending) and created_at (ascending)
        pending_ids.sort(key=lambda rid: (
            -self._requests[rid].priority,
            self._requests[rid].created_at
        ))

        request_id = pending_ids[0]
        request = self._requests[request_id]
        request.status = ReviewStatus.IN_PROGRESS
        request.assigned_to = analyst_id

        self._statistics.pending -= 1
        self._statistics.in_progress += 1

        return request

    def get_request(self, request_id: str) -> Optional[ReviewRequest]:
        """Get request by ID"""
        return self._requests.get(request_id)

    def submit_review(self,
                      request_id: str,
                      decision: ReviewDecision,
                      notes: str,
                      reviewed_by: str,
                      override_confidence: Optional[float] = None) -> bool:
        """
        Submit review decision for a request

        Args:
            request_id: ID of the review request
            decision: Review decision
            notes: Reviewer notes
            reviewed_by: Identifier of the reviewer
            override_confidence: Optional confidence override

        Returns:
            True if successful
        """
        if request_id not in self._requests:
            print(f"[ERROR] Request {request_id} not found")
            return False

        request = self._requests[request_id]

        if request.status not in [ReviewStatus.PENDING, ReviewStatus.IN_PROGRESS]:
            print(f"[ERROR] Request {request_id} already {request.status.value}")
            return False

        request.status = ReviewStatus.COMPLETED
        request.review_decision = decision
        request.review_notes = notes
        request.reviewed_by = reviewed_by
        request.reviewed_at = datetime.now().isoformat()
        request.override_confidence = override_confidence

        # Calculate review time
        if request.created_at:
            created = datetime.fromisoformat(request.created_at)
            reviewed = datetime.fromisoformat(request.reviewed_at)
            review_time = (reviewed - created).total_seconds()

            # Update average review time
            total_time = self._statistics.average_review_time_seconds * (self._statistics.completed)
            self._statistics.average_review_time_seconds = (total_time + review_time) / (self._statistics.completed + 1)

        # Update statistics
        self._statistics.completed += 1
        self._statistics.in_progress = max(0, self._statistics.in_progress - 1)

        if decision == ReviewDecision.ACCEPT:
            self._statistics.accepted += 1
        elif decision == ReviewDecision.REJECT:
            self._statistics.rejected += 1
        elif decision == ReviewDecision.NEEDS_MORE_INFO:
            self._statistics.needs_more_info += 1
        elif decision == ReviewDecision.ESCALATE:
            self._statistics.escalated_count += 1
        elif decision == ReviewDecision.DEFER:
            self._statistics.deferred += 1

        return True

    def escalate_request(self, request_id: str, reason: str) -> bool:
        """
        Escalate a request to senior analyst

        Args:
            request_id: ID of the review request
            reason: Reason for escalation

        Returns:
            True if successful
        """
        if request_id not in self._requests:
            return False

        request = self._requests[request_id]
        request.status = ReviewStatus.ESCALATED
        request.review_notes = f"[ESCALATED] {reason}\n{request.review_notes or ''}"
        request.priority = min(3, request.priority + 1)  # Increase priority

        self._statistics.pending = max(0, self._statistics.pending - 1)
        self._statistics.escalated += 1

        return True

    def _clean_expired(self):
        """Remove expired requests"""
        now = datetime.now()
        expired_ids = []

        for rid, req in self._requests.items():
            if req.status == ReviewStatus.PENDING:
                created = datetime.fromisoformat(req.created_at)
                age_hours = (now - created).total_seconds() / 3600
                if age_hours > self.expiration_hours:
                    expired_ids.append(rid)

        for rid in expired_ids:
            self._requests[rid].status = ReviewStatus.EXPIRED
            self._statistics.pending -= 1
            self._statistics.expired += 1

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        return {
            "total": len(self._requests),
            "pending": self._statistics.pending,
            "in_progress": self._statistics.in_progress,
            "completed": self._statistics.completed,
            "escalated": self._statistics.escalated,
            "expired": self._statistics.expired,
            "avg_wait_time_seconds": self._estimate_wait_time()
        }

    def _estimate_wait_time(self) -> float:
        """Estimate average wait time based on queue depth"""
        if self._statistics.average_review_time_seconds == 0:
            return 0

        pending_count = self._statistics.pending
        # Assume 1 analyst processing in parallel
        estimated_wait = pending_count * self._statistics.average_review_time_seconds
        return estimated_wait

    def get_statistics(self) -> ReviewStatistics:
        """Get review statistics"""
        return self._statistics


class HumanReviewHandler:
    """
    Handler for human-in-the-loop review process

    Implements the manual review delegation described in Section 3.2:
    If C_final < tau_back, sample triggers manual review.
    """

    def __init__(self,
                 tau_back: float = 0.75,
                 auto_escalation_threshold: float = 0.5,
                 max_queue_size: int = 10000):
        """
        Initialize human review handler

        Args:
            tau_back: Confidence threshold for triggering review
            auto_escalation_threshold: Confidence below which auto-escalation occurs
            max_queue_size: Maximum size of review queue
        """
        self.tau_back = tau_back
        self.auto_escalation_threshold = auto_escalation_threshold
        self.review_queue = ReviewQueue(max_size=max_queue_size)

        # Feedback loop for model improvement
        self.feedback_samples: List[Dict[str, Any]] = []

        print(f"[INFO] HumanReviewHandler initialized with tau_back={tau_back}")

    def should_trigger_review(self, final_confidence: float) -> bool:
        """
        Determine if a sample should trigger manual review

        As per Equation (4): If C_final < tau_back, trigger manual review
        """
        return final_confidence < self.tau_back

    def create_review_request(self,
                              sample_id: str,
                              final_confidence: float,
                              frontend_score: float,
                              backend_report: Dict[str, Any],
                              stage_outputs: List[Dict[str, Any]],
                              retrieval_context: Dict[str, Any]) -> ReviewRequest:
        """
        Create a review request for a low-confidence sample

        Args:
            sample_id: Identifier of the sample
            final_confidence: C_final value
            frontend_score: XGBoost anomaly score (C_d)
            backend_report: Attribution report from backend
            stage_outputs: Outputs from CoT stages
            retrieval_context: Retrieved knowledge context

        Returns:
            ReviewRequest object
        """
        request_id = hashlib.md5(f"{sample_id}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        confidence_gap = self.tau_back - final_confidence

        # Determine priority based on confidence gap
        if confidence_gap > 0.3:
            priority = 3  # Critical
        elif confidence_gap > 0.2:
            priority = 2  # High
        elif confidence_gap > 0.1:
            priority = 1  # Medium
        else:
            priority = 0  # Low

        # Auto-escalate if confidence is extremely low
        if final_confidence < self.auto_escalation_threshold:
            priority = min(3, priority + 1)

        return ReviewRequest(
            request_id=request_id,
            sample_id=sample_id,
            timestamp=datetime.now().isoformat(),
            final_confidence=final_confidence,
            tau_back_used=self.tau_back,
            confidence_gap=confidence_gap,
            frontend_score=frontend_score,
            backend_report=backend_report,
            stage_outputs=stage_outputs,
            retrieval_context=retrieval_context,
            priority=priority
        )

    def submit_for_review(self, review_request: ReviewRequest) -> bool:
        """
        Submit a sample for manual review

        Args:
            review_request: ReviewRequest to submit

        Returns:
            True if successfully added to queue
        """
        return self.review_queue.add_request(review_request)

    def process_auto_review(self,
                            final_confidence: float,
                            sample_id: str,
                            frontend_score: float,
                            backend_report: Dict,
                            stage_outputs: List[Dict],
                            retrieval_context: Dict) -> Tuple[bool, Optional[ReviewRequest]]:
        """
        Complete workflow: check, create, and submit for review if needed

        Args:
            final_confidence: C_final from backend
            sample_id: Sample identifier
            frontend_score: XGBoost score
            backend_report: Attribution report
            stage_outputs: CoT stage outputs
            retrieval_context: Retrieved knowledge

        Returns:
            Tuple of (needs_review, review_request)
        """
        if self.should_trigger_review(final_confidence):
            review_request = self.create_review_request(
                sample_id=sample_id,
                final_confidence=final_confidence,
                frontend_score=frontend_score,
                backend_report=backend_report,
                stage_outputs=stage_outputs,
                retrieval_context=retrieval_context
            )
            self.submit_for_review(review_request)
            return True, review_request

        return False, None

    def get_pending_count(self) -> int:
        """Get number of pending reviews"""
        return self.review_queue.get_statistics().pending

    def get_next_review(self, analyst_id: Optional[str] = None) -> Optional[ReviewRequest]:
        """Get next pending review for an analyst"""
        return self.review_queue.get_next_request(analyst_id)

    def submit_decision(self,
                        request_id: str,
                        decision: ReviewDecision,
                        notes: str,
                        reviewer_id: str,
                        override_confidence: Optional[float] = None) -> bool:
        """
        Submit a review decision

        Args:
            request_id: ID of the review request
            decision: Review decision
            notes: Reviewer notes
            reviewer_id: Identifier of the reviewer
            override_confidence: Optional confidence override

        Returns:
            True if successful
        """
        success = self.review_queue.submit_review(
            request_id=request_id,
            decision=decision,
            notes=notes,
            reviewed_by=reviewer_id,
            override_confidence=override_confidence
        )

        if success:
            # Record for feedback loop
            request = self.review_queue.get_request(request_id)
            if request:
                self._record_feedback(request, decision)

        return success

    def _record_feedback(self, request: ReviewRequest, decision: ReviewDecision):
        """Record feedback for model improvement"""
        feedback = {
            "sample_id": request.sample_id,
            "request_id": request.request_id,
            "final_confidence": request.final_confidence,
            "decision": decision.value,
            "review_notes": request.review_notes,
            "reviewed_at": request.reviewed_at,
            "reviewed_by": request.reviewed_by
        }

        # Track false positive/negative corrections
        if decision == ReviewDecision.REJECT:
            # Sample was flagged as anomaly but human says benign
            self.review_queue._statistics.false_positive_corrections += 1
        elif decision == ReviewDecision.ACCEPT:
            # Sample was correctly identified
            pass

        self.feedback_samples.append(feedback)

    def get_feedback_for_retraining(self) -> List[Dict[str, Any]]:
        """
        Get feedback samples for model retraining

        Returns:
            List of feedback samples with correct labels
        """
        return [
            {
                "sample_id": f["sample_id"],
                "original_confidence": f["final_confidence"],
                "corrected_label": 1 if f["decision"] == "accept" else 0,
                "review_notes": f["review_notes"]
            }
            for f in self.feedback_samples
            if f["decision"] in ["accept", "reject"]
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive review statistics"""
        stats = self.review_queue.get_statistics()

        return {
            "queue": self.review_queue.get_queue_status(),
            "decisions": {
                "accepted": stats.accepted,
                "rejected": stats.rejected,
                "needs_more_info": stats.needs_more_info,
                "escalated": stats.escalated_count,
                "deferred": stats.deferred
            },
            "performance": {
                "average_review_time_seconds": stats.average_review_time_seconds,
                "average_confidence_gap": stats.average_confidence_gap
            },
            "feedback_loop": {
                "total_feedback_samples": len(self.feedback_samples),
                "false_positive_corrections": stats.false_positive_corrections,
                "false_negative_corrections": stats.false_negative_corrections
            },
            "config": {
                "tau_back": self.tau_back,
                "auto_escalation_threshold": self.auto_escalation_threshold,
                "queue_max_size": self.review_queue.max_size
            }
        }

    def generate_review_summary(self) -> str:
        """Generate a human-readable review summary"""
        stats = self.get_statistics()

        lines = [
            "=" * 60,
            "HUMAN REVIEW SUMMARY",
            "=" * 60,
            f"Queue Status: {stats['queue']['pending']} pending, {stats['queue']['in_progress']} in progress",
            f"Completed Reviews: {stats['queue']['completed']}",
            f"Average Review Time: {stats['performance']['average_review_time_seconds']:.1f} seconds",
            f"Average Confidence Gap: {stats['performance']['average_confidence_gap']:.3f}",
            "-" * 40,
            "DECISION BREAKDOWN:",
            f"  Accepted: {stats['decisions']['accepted']}",
            f"  Rejected: {stats['decisions']['rejected']}",
            f"  Needs More Info: {stats['decisions']['needs_more_info']}",
            f"  Escalated: {stats['decisions']['escalated']}",
            f"  Deferred: {stats['decisions']['deferred']}",
            "-" * 40,
            "FEEDBACK LOOP:",
            f"  False Positive Corrections: {stats['feedback_loop']['false_positive_corrections']}",
            f"  Retraining Samples Available: {len(self.get_feedback_for_retraining())}",
            "=" * 60
        ]

        return "\n".join(lines)

    def export_review_log(self, output_path: str):
        """Export all review requests to a JSON file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        export_data = {
            "timestamp": datetime.now().isoformat(),
            "statistics": self.get_statistics(),
            "reviews": [
                req.to_dict() for req in self.review_queue._requests.values()
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"[INFO] Review log exported to {output_path}")

    def clear_completed(self, older_than_days: int = 30):
        """Clear completed review requests older than specified days"""
        now = datetime.now()
        to_remove = []

        for rid, req in self.review_queue._requests.items():
            if req.status in [ReviewStatus.COMPLETED, ReviewStatus.EXPIRED]:
                if req.reviewed_at:
                    reviewed = datetime.fromisoformat(req.reviewed_at)
                    age_days = (now - reviewed).total_seconds() / 86400
                    if age_days > older_than_days:
                        to_remove.append(rid)

        for rid in to_remove:
            del self.review_queue._requests[rid]

        print(f"[INFO] Cleared {len(to_remove)} completed reviews older than {older_than_days} days")


# ================================================================
# Review Interface CLI (for human analysts)
# ================================================================

class ReviewCLI:
    """
    Command-line interface for human reviewers
    """

    def __init__(self, handler: HumanReviewHandler):
        self.handler = handler
        self.current_analyst_id = "analyst_001"

    def run_interactive(self):
        """Run interactive review session"""
        print("=" * 60)
        print("CAAAPT Human Review Interface")
        print("=" * 60)
        print(f"Analyst: {self.current_analyst_id}")
        print(f"Pending reviews: {self.handler.get_pending_count()}")
        print("-" * 40)

        while True:
            # Get next review
            review = self.handler.get_next_review(self.current_analyst_id)

            if review is None:
                print("\nNo pending reviews. Exiting.")
                break

            # Display review interface
            print("\n" + review.to_review_interface())

            # Get user input
            print("\nDecision (1-5): ")
            print("  1. ACCEPT")
            print("  2. REJECT")
            print("  3. NEEDS_MORE_INFO")
            print("  4. ESCALATE")
            print("  5. DEFER")
            print("  q. Quit")

            choice = input("\nEnter choice: ").strip().lower()

            if choice == 'q':
                break

            decision_map = {
                '1': ReviewDecision.ACCEPT,
                '2': ReviewDecision.REJECT,
                '3': ReviewDecision.NEEDS_MORE_INFO,
                '4': ReviewDecision.ESCALATE,
                '5': ReviewDecision.DEFER
            }

            if choice not in decision_map:
                print("Invalid choice. Please try again.")
                continue

            decision = decision_map[choice]
            notes = input("Review notes: ").strip()

            success = self.handler.submit_decision(
                request_id=review.request_id,
                decision=decision,
                notes=notes,
                reviewer_id=self.current_analyst_id
            )

            if success:
                print(f"\n[SUCCESS] Review submitted for {review.request_id}")
            else:
                print(f"\n[ERROR] Failed to submit review")

            print("-" * 40)


# ================================================================
# Testing and Demonstration
# ================================================================

def test_human_review():
    """Test human review module"""
    print("=" * 60)
    print("Testing Human Review Module")
    print("=" * 60)

    # Create handler
    handler = HumanReviewHandler(tau_back=0.75)

    # Create sample review request
    sample_backend_report = {
        "summary": "Suspicious PowerShell execution detected",
        "tactic_sequence": [{"tactic": "Execution", "confidence": 0.65}],
        "attack_narrative": "Attacker attempting to download and execute malware",
        "final_confidence": 0.62
    }

    stage_outputs = [
        {"stage_name": "reconstruction", "confidence": 0.70},
        {"stage_name": "alignment", "confidence": 0.60},
        {"stage_name": "intention", "confidence": 0.58},
        {"stage_name": "confidence", "confidence": 0.62}
    ]

    # Test should_trigger_review
    print("\n[TEST 1] Trigger review condition")
    confidences = [0.85, 0.70, 0.60, 0.45]
    for conf in confidences:
        should_review = handler.should_trigger_review(conf)
        print(f"  Confidence {conf:.2f}: should_review={should_review}")

    # Test create review request
    print("\n[TEST 2] Create review request")
    request = handler.create_review_request(
        sample_id="sample_001",
        final_confidence=0.62,
        frontend_score=0.78,
        backend_report=sample_backend_report,
        stage_outputs=stage_outputs,
        retrieval_context={"knowledge_entries": ["PowerShell T1059"]}
    )
    print(f"  Request ID: {request.request_id}")
    print(f"  Priority: {request.priority}")
    print(f"  Confidence gap: {request.confidence_gap:.3f}")

    # Test submit for review
    print("\n[TEST 3] Submit for review")
    success = handler.submit_for_review(request)
    print(f"  Submitted: {success}")
    print(f"  Pending count: {handler.get_pending_count()}")

    # Test get next review
    print("\n[TEST 4] Get next review")
    next_review = handler.get_next_review("test_analyst")
    if next_review:
        print(f"  Retrieved: {next_review.request_id}")
        print(f"  Assigned to: {next_review.assigned_to}")

    # Test submit decision
    print("\n[TEST 5] Submit decision")
    success = handler.submit_decision(
        request_id=request.request_id,
        decision=ReviewDecision.ACCEPT,
        notes="Confirmed: PowerShell execution with encoded command",
        reviewer_id="test_analyst"
    )
    print(f"  Decision submitted: {success}")

    # Test statistics
    print("\n[TEST 6] Statistics")
    stats = handler.get_statistics()
    print(json.dumps(stats, indent=2))

    # Test feedback for retraining
    print("\n[TEST 7] Feedback for retraining")
    feedback = handler.get_feedback_for_retraining()
    print(f"  Samples available: {len(feedback)}")

    # Test summary
    print("\n[TEST 8] Review summary")
    print(handler.generate_review_summary())

    print("\n[INFO] Human review module test completed")


def demo_workflow():
    """Demonstrate complete review workflow"""
    print("=" * 60)
    print("Human Review Workflow Demo")
    print("=" * 60)

    handler = HumanReviewHandler(tau_back=0.75)

    # Simulate multiple samples requiring review
    samples = [
        {"id": "sample_001", "confidence": 0.62, "frontend": 0.78, "desc": "Suspicious PowerShell"},
        {"id": "sample_002", "confidence": 0.48, "frontend": 0.55, "desc": "Unusual process chain"},
        {"id": "sample_003", "confidence": 0.71, "frontend": 0.82, "desc": "Registry modification"},
        {"id": "sample_004", "confidence": 0.35, "frontend": 0.40, "desc": "Multiple failed logins"},
        {"id": "sample_005", "confidence": 0.68, "frontend": 0.75, "desc": "Fileless malware pattern"}
    ]

    print("\n[SIMULATION] Creating review requests...")
    for sample in samples:
        request = handler.create_review_request(
            sample_id=sample["id"],
            final_confidence=sample["confidence"],
            frontend_score=sample["frontend"],
            backend_report={"description": sample["desc"]},
            stage_outputs=[],
            retrieval_context={}
        )
        handler.submit_for_review(request)
        print(f"  {sample['id']}: conf={sample['confidence']:.2f}, priority={request.priority}")

    print(f"\n[QUEUE] Total pending: {handler.get_pending_count()}")

    # Simulate review processing
    print("\n[SIMULATION] Processing reviews...")
    decisions = [ReviewDecision.ACCEPT, ReviewDecision.REJECT, ReviewDecision.ACCEPT,
                 ReviewDecision.ESCALATE, ReviewDecision.NEEDS_MORE_INFO]

    for i in range(5):
        review = handler.get_next_review("analyst_001")
        if review:
            decision = decisions[i % len(decisions)]
            handler.submit_decision(
                request_id=review.request_id,
                decision=decision,
                notes=f"Automated review decision: {decision.value}",
                reviewer_id="analyst_001"
            )
            print(f"  {review.sample_id}: {decision.value}")

    # Display final statistics
    print("\n[FINAL STATISTICS]")
    print(handler.generate_review_summary())

    print("\n[INFO] Demo completed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CAAAPT Human Review Module")
    parser.add_argument("--test", action="store_true", help="Run human review tests")
    parser.add_argument("--demo", action="store_true", help="Run workflow demo")
    parser.add_argument("--cli", action="store_true", help="Run interactive CLI (requires pending reviews)")

    args = parser.parse_args()

    if args.cli:
        handler = HumanReviewHandler()
        cli = ReviewCLI(handler)
        cli.run_interactive()
    elif args.demo:
        demo_workflow()
    elif args.test:
        test_human_review()
    else:
        print("CAAAPT Human Review Module")
        print("Run with --test to test, --demo for workflow demo, --cli for interactive review")