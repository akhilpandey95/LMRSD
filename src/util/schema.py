# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, you can obtain one at the repository root in LICENSE.

from pydantic import BaseModel

# structured output schema for peer review scoring
class PeerReviewScoringTuple(BaseModel):
    description: str
    score: int

# structured output schema for paper idea
class PeerReviewIdea(BaseModel):
    idea_only_review_confidence: PeerReviewScoringTuple
    idea_only_review_content: str
    idea_only_review_rating: PeerReviewScoringTuple

# structured output schema for paper content
class PeerReviewPaperContent(BaseModel):
    review_confidence: PeerReviewScoringTuple
    review_content: str
    review_rating: PeerReviewScoringTuple

# structured output schema for peer review scoring
class PeerReview(BaseModel):
    peer_review_title: str
    peer_review_summary: str
    peer_review_paper_idea: PeerReviewIdea
    peer_review_paper_content: PeerReviewPaperContent
