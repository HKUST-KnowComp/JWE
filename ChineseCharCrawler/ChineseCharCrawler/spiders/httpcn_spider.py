# -*- coding: utf-8 -*-
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from ChineseCharCrawler.items import *

__author__ = 'Jian Xun'


class HttpcnSpider(CrawlSpider):
    name = 'httpcn'
    allowed_domains = ['tool.httpcn.com']
    start_urls = [
        'http://tool.httpcn.com/Zi/BuShou.html'
    ]

    rules = (
        Rule(LinkExtractor(allow=(r'tool\.httpcn\.com/Html/zi/BuShou',))),
        Rule(LinkExtractor(allow=(r'/Html/Zi/[0-9]+/[a-zA-Z]+\.shtml',)),
             callback='parse_char'),
    )

    def __init__(self):
        CrawlSpider.__init__(self)

    def parse_char(self, response):
        self.logger.info('Character page: %s', response.url)
        item = ChineseCharItem()
        item['url'] = response.url
        item['char'] = response.xpath('//div[@class="content_nav"]/text()[2]').extract()[0].split('>')[1].strip()

        item['pinyin'] = response.xpath('//div[@id="div_a1"]/table/tr/td[2]/p/span[@class="pinyin"][1]/text()').extract()[0].strip()
        ph_not = response.xpath('//div[@id="div_a1"]/table/tr/td[2]/p/span[@class="pinyin"][2]/text()').extract()
        if len(ph_not) > 0:
            item['phonetic_notation'] = ph_not[0].strip()
        structures = response.xpath('//div[@id="div_a1"]/table/tr/td[2]/p//text()').extract()
        # self.logger.info(str(structures))
        for i in xrange(len(structures)):
            mark = structures[i]
            if u'部首：' in mark:
                item['radical'] = structures[i + 1].strip()
            if u'总笔画：' in mark:
                item['stroke_count'] = structures[i + 1].strip()
        structures = response.xpath('//div[@id="div_a1"]/p[1]//text()').extract()
        for i in xrange(len(structures)):
            mark = structures[i]
            if u'五笔86：' in mark:
                item['wubi86'] = structures[i + 1].strip()
            if u'五笔98：' in mark:
                item['wubi98'] = structures[i + 1].strip()
            if u'仓颉：' in mark:
                item['cangjie'] = structures[i + 1].strip()
            if u'四角号码：' in mark:
                item['four_corner'] = structures[i + 1].strip()
            if u'UniCode：' in mark:
                item['UniCode'] = structures[i + 1].strip()
        item['is_normal'] = u'是否为常用字：是' in response.xpath('//div[@id="div_a1"]/div[1]/text()').extract()[0]
        item['is_family_name'] = u'姓名学：姓' in response.xpath('//div[@id="div_a1"]/div[1]/text()').extract()[1]
        item['components'] = item['char']
        structures = response.xpath('//div[@id="div_a1"]/div[2]//text()').extract()
        # self.logger.info(str(structures))
        for i in xrange(len(structures)):
            mark = structures[i]
            if u'笔顺编号' in mark:
                item['stroke_numbers'] = structures[i + 1].split(u'：')[1].strip()
            if u'汉字部件构造' in mark:
                item['components'] = structures[i + 1].split(u'：')[1].strip()
            if u'笔顺读写' in mark:
                item['strokes'] = structures[i + 1].split(u'：')[1].strip()
        yield item
