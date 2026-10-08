"""
/conversations endpoints (Day 13): create a conversation, then pass its id
as `conversation_id` on /ask or /ask/stream to get contextualized, remembered
multi-turn Q&A. See services/contextualizer.py and
repositories/conversation_repository.py for the mechanism.

Same tenant-isolation pattern as /documents: `get_owner()` returning None OR
a different tenant_id both produce an identical 404 -- never confirm that a
conversation_id exists to a caller who doesn't own it.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.auth import TenantContext, get_current_tenant
from app.dependencies import get_conversation_repository
from app.models.conversation import ConversationHistoryResponse, CreateConversationResponse
from app.repositories.conversation_repository import ConversationRepository

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=CreateConversationResponse, status_code=201)
async def create_conversation(
    tenant: TenantContext = Depends(get_current_tenant),
    repo: ConversationRepository = Depends(get_conversation_repository),
) -> CreateConversationResponse:
    conversation_id = repo.create_conversation(tenant.tenant_id)
    return CreateConversationResponse(conversation_id=conversation_id)


@router.get("/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    conversation_id: str,
    tenant: TenantContext = Depends(get_current_tenant),
    repo: ConversationRepository = Depends(get_conversation_repository),
) -> ConversationHistoryResponse:
    owner = repo.get_owner(conversation_id)
    if owner is None or owner != tenant.tenant_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    turns = repo.get_turns(conversation_id)
    return ConversationHistoryResponse(conversation_id=conversation_id, turns=turns)
