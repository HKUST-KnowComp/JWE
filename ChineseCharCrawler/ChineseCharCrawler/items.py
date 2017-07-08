# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class ChineseCharItem(scrapy.Item):
    url = scrapy.Field()
    char = scrapy.Field()
    pinyin = scrapy.Field()
    phonetic_notation = scrapy.Field()
    radical = scrapy.Field()
    stroke_count = scrapy.Field()
    strokes = scrapy.Field()
    stroke_numbers = scrapy.Field()
    wubi86 = scrapy.Field()
    wubi98 = scrapy.Field()
    four_corner = scrapy.Field()
    cangjie = scrapy.Field()
    UniCode = scrapy.Field()
    is_normal = scrapy.Field()
    is_family_name = scrapy.Field()
    components = scrapy.Field()
