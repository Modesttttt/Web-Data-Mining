BOT_NAME = "lab4_scraper"

SPIDER_MODULES = ["lab4_scraper.spiders"]
NEWSPIDER_MODULE = "lab4_scraper.spiders"

ROBOTSTXT_OBEY = True
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/150.0.0.0 Safari/537.36"
)

# Ограничения на частоту и количество запросов.
DOWNLOAD_DELAY = 1.0
DOWNLOAD_DELAY_JITTER = 0.5
CONCURRENT_REQUESTS = 8
CONCURRENT_REQUESTS_PER_DOMAIN = 2

RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [429, 500, 502, 503, 504]

DOWNLOAD_TIMEOUT = 30

ITEM_PIPELINES = {
    "lab4_scraper.pipelines.CleaningPipeline": 100,
    "lab4_scraper.pipelines.DuplicationPipeline": 200,
    "lab4_scraper.pipelines.ValidationPipeline": 300,
}

# Подключение Playwright для динамических страниц.
DOWNLOAD_HANDLERS = {
    "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
    "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
}

TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
PLAYWRIGHT_BROWSER_TYPE = "chromium"
PLAYWRIGHT_LAUNCH_OPTIONS = {"headless": True}
PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT = 30_000

FEEDS = {
    "output/default.json": {
        "format": "json",
        "encoding": "utf8",
        "indent": 2,
        "overwrite": True,
    },
}

LOG_LEVEL = "INFO"