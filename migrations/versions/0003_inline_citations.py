"""Store checked citations alongside each chat answer."""

from alembic import op

revision = "0003_inline_citations"
down_revision = "0002_document_chat"
branch_labels = None
depends_on = None


def upgrade():
    # 0002 creates chat_turns from the current model, which already has this column on a
    # fresh database; only databases created before 0003 need it added.
    op.execute("ALTER TABLE chat_turns ADD COLUMN IF NOT EXISTS answer_parts JSONB")
    op.execute("UPDATE chat_turns SET answer_parts = '[]'::jsonb WHERE answer_parts IS NULL")
    op.alter_column("chat_turns", "answer_parts", nullable=False)


def downgrade():
    op.drop_column("chat_turns", "answer_parts")
