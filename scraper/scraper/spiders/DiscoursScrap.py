import scrapy
from bs4 import BeautifulSoup
class DiscoursScrapper(scrapy.Spider):
    name = 'discours'
    def start_requests(self):
        url=['https://www.vie-publique.fr/discours?page=0']
        yield scrapy.Request(url=url[0],callback =self.get_all_discours)
    def get_all_discours(self,response):
        for row in response.css('div.views-row'):
            link = row.css('a.link-multiple::attr(href)').extract()
            if link:
                link = 'https://www.vie-publique.fr'+link[0]
                rq = response.follow(link,callback=self.get_discours)
                yield rq
        if int(response.request.url.split('=')[1]) < 11636:
            lien = 'https://www.vie-publique.fr/discours?page='+str(int(response.request.url.split('=')[1])+1)
            yield scrapy.Request(url = lien,callback=self.get_all_discours)
    def get_discours(self,response):
        titre = response.xpath('//*[@id="block-ldf-content"]/div/div[1]/div[1]/div/div/h1/text()').extract()
        if titre:
            titre = titre[0].replace('\n','')
        else:
            titre = ''
        prenomnom= response.xpath('//*[@id="block-ldf-content"]/div/div[1]/div[1]/div/div/div[2]/ul/li[1]/a/text()').extract()
        if prenomnom:
            prenom = prenomnom[0].split(' ')[0]
            nom = prenomnom[0].strip(' ').split(' ')[1:]
        else:
            prenom = ''
            nom = ''
        fonction = response.xpath('//*[@id="block-ldf-content"]/div/div[1]/div[1]/div/div/div[2]/ul/li[1]/text()').extract()
        if fonction:
            fonction = fonction[0].replace('\n','').replace('  ','')
        else:
            fonction = ''
        dt = response.xpath('//*[@id="block-ldf-content"]/div/div[1]/div[1]/div/div/div[1]/p[1]/span/time/@datetime').extract()
        if dt:
            dt = dt[0]
        else:
            dt =''
        if response.xpath('//*[@id="block-ldf-content"]/div/div[1]/div[2]/div/div/div'):
            tags = response.xpath('//*[@id="block-ldf-content"]/div/div[1]/div[2]/div/div/div/ul/li/a/text()').extract()
        else:
            tags = ''
        if response.xpath('//*[@id="block-ldf-content"]/div/div[1]/div[1]/div/div/div[2]/div/div/ul/li'):
            themes = response.xpath('//*[@id="block-ldf-content"]/div/div[1]/div[1]/div/div/div[2]/div/div/ul/li/text()').extract()
        else:
            themes= ''
        if response.css('span.field--name-field-texte-integral'):
            text = response.css('span.field--name-field-texte-integral').extract()[0]
            para = text.split('\n')
            para = [x + ' ' for x in para if x!='']
            discours = []
            for p in para:
                if not p.startswith('Source'):
                    discours.append(p)
            text = ' '.join(discours)
        else:
            text = ''
        text = BeautifulSoup(text,'lxml').text
        unique_id  = response.request.url.split('/')[4].split('-')[0]
        yield {
            'Id':unique_id.encode("utf-8") ,
            'Titre':titre.encode("utf-8") ,
            'Type':titre.split(' ')[0].lower().encode('utf-8'),
            'Theme':themes,
            'Prenom':prenom.encode("utf-8") ,
            'Nom':nom,
            'Fonction':fonction.encode("utf-8") ,
            'Date':dt.encode("utf-8") ,
            'Tags':tags ,
            'Texte': text.encode("utf-8") ,
            'Lien':response.request.url
        }