"""基于 crawl4ai 实现 web fetch 工具

playwright install --with-deps chromium
"""

import asyncio
from typing import Any

from langchain_core.tools import tool
from loguru import logger

_CRAWL_SEMAPHORE = asyncio.Semaphore(8)


async def _crawl_one(url: str) -> dict[str, Any]:
    """Crawl a single URL and return structured result dict."""
    try:
        from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
        from crawl4ai.content_filter_strategy import PruningContentFilter
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    except ImportError as exc:
        raise ImportError(
            "crawl4ai is required for web fetching. Install with: uv add crawl4ai"
        ) from exc

    prune_filter = PruningContentFilter(
        threshold=1.0,
        threshold_type="fixed",
        min_word_threshold=10,
    )
    md_generator = DefaultMarkdownGenerator(
        options={
            "ignore_links": True,
            "ignore_images": False,
            "escape_html": False,
        },
        content_filter=prune_filter,
    )
    config = CrawlerRunConfig(
        markdown_generator=md_generator,
        exclude_external_links=True,
        exclude_internal_links=True,
        exclude_social_media_links=True,
    )

    empty: dict[str, Any] = {
        "url": url,
        "title": "",
        "description": "",
        "content": "",
        "image_url": "",
    }

    async with _CRAWL_SEMAPHORE:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(
                url=url,
                config=config,
            )

    if not getattr(result, "success", False):
        logger.warning("Crawl failed for {}", url)
        return empty
    metadata = getattr(result, "metadata", {}) or {}
    content = result.markdown.replace("\n\n", "\n").strip()
    image_url = (
        metadata.get(
            "og:image",
            metadata.get("twitter:image", ""),
        )
        or ""
    )
    title = (
        metadata.get(
            "title",
            metadata.get(
                "og:title",
                metadata.get("twitter:title", ""),
            ),
        )
        or ""
    )
    description = (
        metadata.get(
            "description",
            metadata.get(
                "og:description",
                metadata.get("twitter:description", ""),
            ),
        )
        or ""
    )

    return {
        "url": getattr(result, "url", url) or url,
        "title": title,
        "description": description,
        "content": content,
        "image_url": image_url,
    }


@tool
async def web_fetch(urls: list[str]) -> list[dict]:
    """Fetch content directly from specific URLs as markdown.

    Args:
        urls: List of complete URLs to fetch content from.
            Use when the user provides specific URLs they
            want to read or analyse.
    """
    logger.debug(
        "Fetching content from {} URLs: {}",
        len(urls),
        urls,
    )
    blocked: list[dict] = []

    tasks = [_crawl_one(u) for u in urls]
    results = await asyncio.gather(
        *tasks,
        return_exceptions=True,
    )
    output: list[dict] = list(blocked)
    for url, res in zip(urls, results):
        if isinstance(res, Exception):
            logger.error("Error fetching {}: {}", url, res)
            output.append(
                {
                    "url": url,
                    "title": "",
                    "description": "",
                    "content": "",
                    "image_url": "",
                }
            )
        else:
            output.append(res)

    logger.debug("Fetch returned {} results", len(output))
    return output


if __name__ == "__main__":
    value = asyncio.run(
        web_fetch.ainvoke({"urls": ["https://www.runoob.com/rust/rust-setup.html"]})
    )
    print(value)
