BOT_NAME = "metals_scraper"

SPIDER_MODULES = ["metals_scraper.spiders"]
NEWSPIDER_MODULE = "metals_scraper.spiders"

ROBOTSTXT_OBEY = True

USER_AGENT = "student-scrapy-practice/1.0"

DOWNLOAD_DELAY = 0.25
CONCURRENT_REQUESTS = 8

FEED_EXPORT_ENCODING = "utf-8"

ITEM_PIPELINES = {
    "metals_scraper.pipelines.CleanDataPipeline": 300,
}

LOG_LEVEL = "INFO"

FEEDS = {
    "output.csv": {
        "format": "csv",
        "encoding": "utf-8-sig",
        "item_export_kwargs": {
            "delimiter": ";"
        },
    },
}