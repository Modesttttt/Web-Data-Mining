import re

import ftfy
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem


class CleaningPipeline:
    """Исправляет кодировку и нормализует текстовые поля."""

    TEXT_FIELDS = ("title", "summary", "description", "source")

    def process_item(self, item):
        adapter = ItemAdapter(item)

        for field in self.TEXT_FIELDS:
            value = adapter.get(field)

            if isinstance(value, str):
                value = ftfy.fix_text(value)
                value = re.sub(r"\s+", " ", value).strip()
                adapter[field] = value

        # Нормализованное название удобно использовать для сравнения товаров.
        title = adapter.get("title")
        if isinstance(title, str):
            adapter["normalized_title"] = title.lower()

        return item


class DuplicationPipeline:
    """Удаляет дубликаты товаров по URL."""

    @classmethod
    def from_crawler(cls, crawler):
        pipeline = cls()
        pipeline.seen_urls = set()
        return pipeline

    def process_item(self, item):
        adapter = ItemAdapter(item)
        url = adapter.get("url")

        if not url:
            return item

        # Один и тот же товар может встретиться несколько раз при пагинации.
        if url in self.seen_urls:
            raise DropItem(f"Дубликат URL: {url}")

        self.seen_urls.add(url)
        return item


class ValidationPipeline:
    """Проверяет наличие обязательных полей."""

    def process_item(self, item):
        adapter = ItemAdapter(item)
        required = ["title", "url"]

        missing = [
            field
            for field in required
            if not adapter.get(field)
        ]

        if missing:
            raise DropItem(
                f"Отсутствуют обязательные поля {missing}: {dict(adapter)}"
            )

        return item