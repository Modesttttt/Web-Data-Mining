import scrapy


class NewsItem(scrapy.Item):
    title = scrapy.Field()
    normalized_title = scrapy.Field()
    summary = scrapy.Field()
    date = scrapy.Field()
    url = scrapy.Field()
    source = scrapy.Field()


class ProductItem(scrapy.Item):
    title = scrapy.Field()
    normalized_title = scrapy.Field()
    price = scrapy.Field()
    rating = scrapy.Field()
    description = scrapy.Field()
    url = scrapy.Field()
    source = scrapy.Field()
