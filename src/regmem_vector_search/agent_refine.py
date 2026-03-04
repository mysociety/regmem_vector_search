import asyncio
import hashlib
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_store import PydanticDBM

from regmem_vector_search.config import settings

SYSTEM_PROMPT = """
You are an expert at analyzing parliamentary speeches for declarations of interest.
Parliamentary rules require that when MPs declare an interest,
they must clearly state the nature of that interest.
Generic statements like 'I refer to the register' or
 'As declared in the register' without specifying the actual interest are NOT considered clear. 
A clear declaration would specify the nature of the interest,
 for example: 'I declare my interest as a landlord' or
'I have shares in company X as noted in the register'. 

Full version of the rule: Declarations must be informative but succinct. A Member who has already registered an interest may refer to his or her Register entry. But such a reference will not suffice on its own, as the declaration must provide sufficient information to convey the nature of the interest without the listener or the reader having to have recourse to the Register or other publication.

In some cases declarations and interests are used rhetorically and it's not a strict declaration of a perceived conflict. For instance "I have Bushmills in my constituency, which provokes my interest in this issue".

Analyze the speech and determine if there is a declaration and if so,  whether it meets the clarity requirement. 

"""


@lru_cache
def _hash_string(text: str) -> str:
    """Generate a short hash for a string to use as a database key."""
    text = SYSTEM_PROMPT + text
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class InterestDeclaration(BaseModel):
    """Analysis of whether a speech contains an interest declaration."""

    contains_declaration: bool = Field(
        description="Whether the speech contains a declaration of interest"
    )
    declaration_is_clear: bool | None = Field(
        default=None,
        description="Whether the nature of the interest is clearly stated. "
        "Should be False if the declaration only refers to the register without specifics. "
        "None if no declaration was found.",
    )
    explanation: str = Field(description="Brief explanation of the decision")


model = OpenAIResponsesModel(
    "gpt-5.2", provider=OpenAIProvider(api_key=settings.OPENAI_API_KEY)
)
agent = Agent(
    model=model,
    output_type=InterestDeclaration,
    system_prompt=SYSTEM_PROMPT,
)


def analyze_interest_declarations(
    speeches: list[str],
) -> list[InterestDeclaration]:
    """
    Analyze multiple speeches for interest declarations, using cache when available.
    """

    cache_dir = Path("data", "regmem_vector_search")
    cache_dir.mkdir(parents=True, exist_ok=True)
    store = PydanticDBM[InterestDeclaration](cache_dir / "interest_declarations.sqlite")

    # Find speeches that need processing
    speeches_to_process = []
    for speech in speeches:
        key = _hash_string(speech)
        if key not in store:
            speeches_to_process.append(speech)

    # Process uncached speeches and update cache
    if speeches_to_process:
        processed = asyncio.run(process_batch(speeches_to_process))
        for speech, declaration in zip(speeches_to_process, processed):
            key = _hash_string(speech)
            store[key] = declaration

    # Extract all results from cache using original speech keys
    results = []
    for speech in speeches:
        declaration = store[_hash_string(speech)]
        results.append(declaration)

    store.close()
    return results


async def process_batch(speeches_to_process: list[str]) -> list[InterestDeclaration]:
    tasks = [
        analyze_interest_declaration_async(speech) for speech in speeches_to_process
    ]
    return await asyncio.gather(*tasks)


async def analyze_interest_declaration_async(speech_text: str) -> InterestDeclaration:
    result = await agent.run("Speech: " + speech_text)
    return result.output
