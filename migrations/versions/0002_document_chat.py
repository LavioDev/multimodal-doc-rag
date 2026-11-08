"""Persist per-document chat turns and pinned answers."""

from alembic import op

from multimodal_doc_rag.models import ChatTurnRecord, PinnedAnswerRecord

revision = "0002_document_chat"
down_revision = "0001_full_flow"
branch_labels = None
depends_on = None


def upgrade():
    ChatTurnRecord.__table__.create(bind=op.get_bind(), checkfirst=True)
    PinnedAnswerRecord.__table__.create(bind=op.get_bind(), checkfirst=True)


def downgrade():
    PinnedAnswerRecord.__table__.drop(bind=op.get_bind(), checkfirst=True)
    ChatTurnRecord.__table__.drop(bind=op.get_bind(), checkfirst=True)
