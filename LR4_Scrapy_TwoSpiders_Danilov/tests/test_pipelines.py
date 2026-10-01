from lab4_scraper.pipelines import CleaningPipeline, DuplicationPipeline, ValidationPipeline
from scrapy.exceptions import DropItem


def test_cleaning_pipeline():
    pipeline = CleaningPipeline()
    item = {"title": "  CafÃ©   Test\n", "summary": "  text\ttext "}
    result = pipeline.process_item(item, None)
    assert result["title"] == "Café Test"
    assert result["normalized_title"] == "café test"
    assert result["summary"] == "text text"


def test_duplication_pipeline():
    pipeline = DuplicationPipeline()
    pipeline.open_spider(None)
    pipeline.process_item({"url": "https://example.test/1"}, None)
    try:
        pipeline.process_item({"url": "https://example.test/1"}, None)
    except DropItem:
        return
    raise AssertionError("Duplicate URL was not dropped")


def test_validation_pipeline():
    pipeline = ValidationPipeline()
    try:
        pipeline.process_item({"title": "Only title"}, None)
    except DropItem:
        return
    raise AssertionError("Invalid item was not dropped")
