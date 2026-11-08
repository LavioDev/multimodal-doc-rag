"""Store checked citations alongside each chat answer."""

from alembic import op
from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB

revision = "0003_inline_citations"
down_revision = "0002_document_chat"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("chat_turns", Column("answer_parts", JSONB, nullable=True))
    op.execute("UPDATE chat_turns SET answer_parts = '[]'::jsonb WHERE answer_parts IS NULL")
    op.alter_column("chat_turns", "answer_parts", nullable=False)


def downgrade():
    op.drop_column("chat_turns", "answer_parts")
