from collections.abc import AsyncGenerator, Iterable

from schemas.response import ContentKind
from utils.response_util import ResponseFactory


async def stream_text_packets(
    chunks: Iterable[str],
    kind: ContentKind = ContentKind.ANSWER,
) -> AsyncGenerator[str, None]:
    for chunk in chunks:
        yield "data: " + ResponseFactory.build_text(
            chunk,
            kind,
        ).model_dump_json() + "\n\n"

    yield "data: " + ResponseFactory.build_finish().model_dump_json() + "\n\n"
