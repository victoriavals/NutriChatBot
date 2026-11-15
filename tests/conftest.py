"""Shared pytest fixtures and lightweight fakes for offline deterministic testing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


@dataclass
class FakeGeminiResponse:
    """Minimal mock of Gemini generate_content response."""

    text: str


@dataclass
class FakeOpenAIMessage:
    """Minimal mock of OpenAI chat completion message."""

    content: str | None
    role: str = "assistant"
    reasoning_content: str | None = None


@dataclass
class FakeOpenAIChoice:
    """Minimal mock of OpenAI chat completion choice."""

    message: FakeOpenAIMessage
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class FakeOpenAIChatResponse:
    """Minimal mock of OpenAI chat completion response."""

    choices: list[FakeOpenAIChoice]


@dataclass
class FakeStreamDelta:
    content: str | None


@dataclass
class FakeStreamChoice:
    delta: FakeStreamDelta


@dataclass
class FakeStreamChunk:
    choices: list[FakeStreamChoice]


class FakeChatCompletions:
    """Fake completions endpoint capturing calls."""

    def __init__(self, reply: str = "Jawaban contoh.", should_fail: bool = False) -> None:
        self.reply = reply
        self.should_fail = should_fail
        self.calls: list[dict[str, Any]] = []

    def create(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
                **kwargs,
            }
        )
        if self.should_fail:
            raise RuntimeError("Simulated upstream failure with sk-FakeSecretKey1234567890")

        if stream:
            # Yield in word chunks to simulate true token streaming
            words = self.reply.split(" ")
            chunks = []
            for idx, word in enumerate(words):
                space = " " if idx < len(words) - 1 else ""
                chunks.append(FakeStreamChunk(choices=[FakeStreamChoice(delta=FakeStreamDelta(content=word + space))]))
            return iter(chunks)

        return FakeOpenAIChatResponse(
            choices=[FakeOpenAIChoice(message=FakeOpenAIMessage(content=self.reply))]
        )



class FakeChat:
    """Fake chat namespace."""

    def __init__(self, completions: FakeChatCompletions) -> None:
        self.completions = completions


class FakeOpenAIClient:
    """Fake client matching openai.OpenAI interface."""

    def __init__(self, reply: str = "Jawaban contoh.", should_fail: bool = False) -> None:
        self._completions = FakeChatCompletions(reply=reply, should_fail=should_fail)
        self.chat = FakeChat(self._completions)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self._completions.calls


class FakeGeminiModel:
    """Dual-mode fake supporting both generate_content and chat.completions.create."""

    def __init__(self, reply: str = "Jawaban contoh.", should_fail: bool = False) -> None:
        self.reply = reply
        self.should_fail = should_fail
        self._completions = FakeChatCompletions(reply=reply, should_fail=should_fail)
        self.chat = FakeChat(self._completions)
        self.calls: list[dict[str, Any]] = []

    def generate_content(
        self,
        prompt: str,
        generation_config: dict[str, Any] | None = None,
    ) -> FakeGeminiResponse:
        self.calls.append({"prompt": prompt, "generation_config": generation_config})
        if self.should_fail:
            raise RuntimeError("Simulated upstream failure with AIzaSyFakeSecretKey1234567890")
        return FakeGeminiResponse(text=self.reply)


@dataclass
class FakeChromaCollection:
    """In-memory stand-in for a ChromaDB collection."""

    name: str
    embedding_function: Any = None
    batches: list[dict[str, Any]] = field(default_factory=list)
    documents: list[str] = field(default_factory=list)

    def add(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        self.batches.append({"documents": documents, "metadatas": metadatas, "ids": ids})
        self.documents.extend(documents)

    def count(self) -> int:
        return len(self.documents)

    def query(self, query_texts: list[str], n_results: int = 3) -> dict[str, Any]:
        return {"documents": [self.documents[:n_results]]}


class FakeChromaClient:
    """In-memory stand-in for chromadb.PersistentClient."""

    def __init__(self) -> None:
        self.collections: dict[str, FakeChromaCollection] = {}
        self.deleted_names: list[str] = []

    def list_collections(self) -> list[FakeChromaCollection]:
        return list(self.collections.values())

    def delete_collection(self, name: str) -> None:
        if name not in self.collections:
            raise ValueError(f"Collection {name} does not exist")
        self.deleted_names.append(name)
        del self.collections[name]

    def get_or_create_collection(
        self,
        name: str,
        embedding_function: Any = None,
    ) -> FakeChromaCollection:
        if name not in self.collections:
            self.collections[name] = FakeChromaCollection(
                name=name,
                embedding_function=embedding_function,
            )
        return self.collections[name]

    def get_collection(
        self,
        name: str,
        embedding_function: Any = None,
    ) -> FakeChromaCollection:
        if name not in self.collections:
            raise ValueError(f"Collection {name} not found")
        return self.collections[name]


@pytest.fixture
def sample_csv_file(tmp_path: Path) -> Path:
    """Create a temporary valid nutrition CSV file including edge-case rows."""
    csv_path = tmp_path / "nutrition.csv"
    df = pd.DataFrame(
        [
            {
                "id": 1,
                "calories": 250.0,
                "proteins": 8.0,
                "fat": 9.0,
                "carbohydrate": 34.0,
                "name": "Nasi Goreng",
                "image": "https://example.com/nasgor.jpg",
            },
            {
                "id": 2,
                "calories": 165.0,
                "proteins": 31.0,
                "fat": 3.6,
                "carbohydrate": 0.0,
                "name": "Dada Ayam",
                "image": None,
            },
            {
                "id": 3,
                "calories": 120.0,
                "proteins": 4.0,
                "fat": 2.0,
                "carbohydrate": 20.0,
                "name": "100% Tempe_Goreng",
                "image": "https://example.com/tempe.jpg",
            },
        ]
    )
    df.to_csv(csv_path, index=False)
    return csv_path
