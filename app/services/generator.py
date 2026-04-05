import time
from typing import List
from openai import OpenAI, APIConnectionError, AuthenticationError, RateLimitError
from app.models.domain import RetrievedContext
from app.services.prompt_builder import PromptBuilder
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class GeneratorService:

    def __init__(self, prompt_builder: PromptBuilder = None):
        self._client: OpenAI = None
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY missing. .env mein add karo.")

        self._client = OpenAI(api_key=settings.openai_api_key)
        logger.info(f"OpenAI client initialized. Model: {settings.openai_model}")
        self._initialized = True

    def generate(self, query: str, contexts: List[RetrievedContext]) -> str:
        if not self._initialized:
            raise RuntimeError("GeneratorService not initialized.")

        messages = self._prompt_builder.build_messages(
            query=query,
            contexts=contexts,
        )

        logger.info(
            f"Sending to OpenAI | Model: {settings.openai_model} | "
            f"Contexts: {len(contexts)} | Query: {query[:60]}..."
        )

        # Retry logic: try up to 3 times
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=settings.openai_model,
                    messages=messages,
                    max_tokens=settings.openai_max_tokens,
                    temperature=settings.openai_temperature,
                )
                answer = response.choices[0].message.content.strip()
                tokens_used = response.usage.total_tokens
                logger.info(f"Answer generated. Tokens used: {tokens_used}")
                return answer

            except RateLimitError:
                # wait = 2 ** attempt  # 2s, 4s, 8s
                wait = 10 * attempt
                logger.warning(
                    f"Rate limit hit. Attempt {attempt}/{max_retries}. "
                    f"Waiting {wait}s..."
                )
                if attempt == max_retries:
                    return (
                        "OpenAI rate limit exceed ho gaya. "
                        "Thoda wait karo aur dobara try karo. "
                        f"Retrieved context: {contexts[0].chunk.content[:300] if contexts else 'N/A'}"
                    )
                time.sleep(wait)

            except AuthenticationError:
                logger.error("Invalid OpenAI API key!")
                raise ValueError("OpenAI API key invalid. .env check karo.")

            except APIConnectionError:
                logger.error("OpenAI connection failed!")
                return "OpenAI se connect nahi ho pa raha. Internet check karo."

            except Exception as e:
                logger.error(f"OpenAI error: {e}")
                raise
