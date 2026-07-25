"""Database-level tests for the repository layer, against a real (in-memory) engine."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from ai_research_assistant.db.models.chat import ChatMessage, ChatSession, MessageRole
from ai_research_assistant.db.models.document import (
    Document,
    DocumentChunk,
    DocumentStatus,
    DocumentType,
)
from ai_research_assistant.db.models.user import User
from ai_research_assistant.repositories.chat_repository import ChatRepository
from ai_research_assistant.repositories.document_repository import DocumentRepository
from ai_research_assistant.repositories.user_repository import UserRepository


async def _create_user(session: AsyncSession) -> User:
    repo = UserRepository(session)
    return await repo.create(User(email=f"{uuid4()}@example.com", hashed_password="x"))


async def test_user_repository_create_and_get_by_id(db_session: AsyncSession) -> None:
    repo = UserRepository(db_session)
    user = await repo.create(User(email="alice@example.com", hashed_password="hashed"))

    fetched = await repo.get_by_id(user.id)

    assert fetched is not None
    assert fetched.email == "alice@example.com"


async def test_user_repository_get_by_id_returns_none_when_missing(
    db_session: AsyncSession,
) -> None:
    repo = UserRepository(db_session)
    assert await repo.get_by_id(uuid4()) is None


async def test_user_repository_get_by_email(db_session: AsyncSession) -> None:
    repo = UserRepository(db_session)
    await repo.create(User(email="bob@example.com", hashed_password="hashed"))

    fetched = await repo.get_by_email("bob@example.com")
    missing = await repo.get_by_email("nobody@example.com")

    assert fetched is not None
    assert fetched.email == "bob@example.com"
    assert missing is None


async def test_document_repository_create_and_get_by_id(db_session: AsyncSession) -> None:
    user = await _create_user(db_session)
    repo = DocumentRepository(db_session)
    document = await repo.create(
        Document(
            owner_id=user.id,
            filename="a.txt",
            document_type=DocumentType.TXT,
            content_type="text/plain",
            file_size_bytes=10,
            status=DocumentStatus.READY,
        )
    )

    fetched = await repo.get_by_id(document.id, user.id)
    wrong_owner = await repo.get_by_id(document.id, uuid4())

    assert fetched is not None
    assert fetched.filename == "a.txt"
    assert wrong_owner is None


async def test_document_repository_list_by_owner_orders_newest_first(
    db_session: AsyncSession,
) -> None:
    user = await _create_user(db_session)
    repo = DocumentRepository(db_session)
    now = datetime.now(UTC)
    first = await repo.create(
        Document(
            owner_id=user.id,
            filename="first.txt",
            document_type=DocumentType.TXT,
            content_type="text/plain",
            file_size_bytes=1,
            status=DocumentStatus.READY,
            created_at=now,
        )
    )
    second = await repo.create(
        Document(
            owner_id=user.id,
            filename="second.txt",
            document_type=DocumentType.TXT,
            content_type="text/plain",
            file_size_bytes=1,
            status=DocumentStatus.READY,
            created_at=now + timedelta(seconds=1),
        )
    )

    documents = await repo.list_by_owner(user.id)

    assert [document.id for document in documents] == [second.id, first.id]


async def test_document_repository_add_chunks_and_get_with_chunks(
    db_session: AsyncSession,
) -> None:
    user = await _create_user(db_session)
    repo = DocumentRepository(db_session)
    document = await repo.create(
        Document(
            owner_id=user.id,
            filename="a.txt",
            document_type=DocumentType.TXT,
            content_type="text/plain",
            file_size_bytes=10,
            status=DocumentStatus.READY,
        )
    )

    await repo.add_chunks(
        [
            DocumentChunk(
                document_id=document.id,
                chunk_index=0,
                content="chunk",
                token_count=1,
                vector_id="v1",
            )
        ]
    )

    fetched = await repo.get_with_chunks(document.id, user.id)

    assert fetched is not None
    assert len(fetched.chunks) == 1
    assert fetched.chunks[0].vector_id == "v1"


async def test_document_repository_save_flushes_in_place_changes(
    db_session: AsyncSession,
) -> None:
    user = await _create_user(db_session)
    repo = DocumentRepository(db_session)
    document = await repo.create(
        Document(
            owner_id=user.id,
            filename="a.txt",
            document_type=DocumentType.TXT,
            content_type="text/plain",
            file_size_bytes=10,
            status=DocumentStatus.PROCESSING,
        )
    )

    document.status = DocumentStatus.READY
    saved = await repo.save(document)

    assert saved.status == DocumentStatus.READY


async def test_document_repository_commit_persists_pending_changes(
    db_session: AsyncSession,
) -> None:
    user = await _create_user(db_session)
    repo = DocumentRepository(db_session)
    document = await repo.create(
        Document(
            owner_id=user.id,
            filename="a.txt",
            document_type=DocumentType.TXT,
            content_type="text/plain",
            file_size_bytes=10,
            status=DocumentStatus.PROCESSING,
        )
    )

    document.status = DocumentStatus.FAILED
    document.error_message = "boom"
    await repo.save(document)
    await repo.commit()

    fetched = await repo.get_by_id(document.id, user.id)
    assert fetched is not None
    assert fetched.status == DocumentStatus.FAILED
    assert fetched.error_message == "boom"


async def test_document_repository_delete(db_session: AsyncSession) -> None:
    user = await _create_user(db_session)
    repo = DocumentRepository(db_session)
    document = await repo.create(
        Document(
            owner_id=user.id,
            filename="a.txt",
            document_type=DocumentType.TXT,
            content_type="text/plain",
            file_size_bytes=10,
            status=DocumentStatus.READY,
        )
    )

    await repo.delete(document)

    assert await repo.get_by_id(document.id, user.id) is None


async def test_chat_repository_create_session_and_list(db_session: AsyncSession) -> None:
    user = await _create_user(db_session)
    repo = ChatRepository(db_session)
    session = await repo.create_session(ChatSession(owner_id=user.id, title="First"))

    sessions = await repo.list_sessions(user.id)

    assert [s.id for s in sessions] == [session.id]


async def test_chat_repository_get_session_scopes_by_owner(db_session: AsyncSession) -> None:
    user = await _create_user(db_session)
    other_user = await _create_user(db_session)
    repo = ChatRepository(db_session)
    session = await repo.create_session(ChatSession(owner_id=user.id, title="Mine"))

    mine = await repo.get_session(session.id, user.id)
    not_mine = await repo.get_session(session.id, other_user.id)

    assert mine is not None
    assert not_mine is None


async def test_chat_repository_messages_ordering_and_recent_limit(
    db_session: AsyncSession,
) -> None:
    user = await _create_user(db_session)
    repo = ChatRepository(db_session)
    session = await repo.create_session(ChatSession(owner_id=user.id, title="t"))

    now = datetime.now(UTC)
    for i in range(3):
        await repo.add_message(
            ChatMessage(
                session_id=session.id,
                role=MessageRole.USER,
                content=f"msg-{i}",
                created_at=now + timedelta(seconds=i),
            )
        )

    all_messages = await repo.get_all_messages(session.id)
    recent = await repo.get_recent_messages(session.id, limit=2)

    assert [m.content for m in all_messages] == ["msg-0", "msg-1", "msg-2"]
    assert [m.content for m in recent] == ["msg-1", "msg-2"]


async def test_chat_repository_delete_session_cascades_messages(
    db_session: AsyncSession,
) -> None:
    user = await _create_user(db_session)
    repo = ChatRepository(db_session)
    session = await repo.create_session(ChatSession(owner_id=user.id, title="t"))
    await repo.add_message(
        ChatMessage(session_id=session.id, role=MessageRole.USER, content="hello")
    )

    await repo.delete_session(session)

    assert await repo.get_session(session.id, user.id) is None
