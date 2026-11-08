"""Create document, page, element, vector and durable job tables."""

from alembic import op

from multimodal_doc_rag.models import (
    Base,
    DocumentRecord,
    ElementRecord,
    EmbeddingRecord,
    JobRecord,
    PageRecord,
)

LEGACY_TABLES = [
    DocumentRecord.__table__,
    PageRecord.__table__,
    ElementRecord.__table__,
    EmbeddingRecord.__table__,
    JobRecord.__table__,
]

revision = "0001_full_flow"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    Base.metadata.create_all(bind=op.get_bind(), tables=LEGACY_TABLES)


def downgrade():
    Base.metadata.drop_all(bind=op.get_bind(), tables=LEGACY_TABLES)
