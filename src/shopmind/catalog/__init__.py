"""catalog 包：数据源 adapter、mock 生成器、版本化快照存储。"""

from shopmind.catalog.mock_source import generate_catalog
from shopmind.catalog.source import CatalogSource, doc_text, query_text_from_requirements
from shopmind.catalog.store import CatalogStore

__all__ = [
    "CatalogSource",
    "CatalogStore",
    "doc_text",
    "generate_catalog",
    "query_text_from_requirements",
]
