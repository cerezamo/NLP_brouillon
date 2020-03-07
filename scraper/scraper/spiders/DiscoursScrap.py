import scrapy
from bs4 import BeautifulSoup
import re
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
        if int(response.request.url.split('=')[1]) < 11639:
            lien = 'https://www.vie-publique.fr/discours?page='+str(int(response.request.url.split('=')[1])+1)
            yield scrapy.Request(url = lien,callback=self.get_all_discours)
    def get_discours(self,response):
        titre = response.xpath('//*[@id="block-ldf-content"]/div/div[1]/div[1]/div/div/h1/text()').extract()
        if titre:
            titre = titre[0].replace('\n','')
            titre = re.sub('  +', '',titre)
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
            themes = response.xpath('//*[@id="block-ldf-content"]/div/div[1]/div[1]/div/div/div[2]/div/div/ul/li/a/text()').extract()
        else:
            themes= ''
        if titre =='':
            try:
                if response.xpath('//*[@id="block-ldf-content"]/div/div[1]/div[1]/div/div/div[1]/p[1]/span/time/text()').extract_first():
                    dat = response.xpath('//*[@id="block-ldf-content"]/div/div[1]/div[1]/div/div/div[1]/p[1]/span/time/text()').extract_first()
                    titre = ''.join(titre[0][titre[0].find(prenomnom[0])+len(prenomnom[0])+2:titre[0].find(dat)].split(',')[:-1]).capitalize()
                else:
                    titre=''
            except:
                titre = ''
        if response.css('span.field--name-field-texte-integral'):
            if titre.split(' ')[0].lower() == 'interview':
                text = response.xpath('//*[@id="block-ldf-content"]/div/div[1]/div[2]/div/div/span/p/text()').extract()
                text = [word.replace('\n','').replace('\xa0','').replace('\x85','').replace('\x96','').replace('\x92',"'").replace('\x80','') for word in text]
                text = [word for word in text if len(word) >0]
                a = set([word.replace('\n','').split(',')[0] for word in text if word.replace('\n','').isupper()])
                dic = {x: [[],[]] for x in a}
                lst_intervenants = [word.replace('\n','').split(',')[0] for word in text if word.replace('\n','').isupper()]
                lst_discours = [word for word in text if not word.replace('\n','').isupper() and not word.replace('\n','').startswith('Source')]
                if len(lst_intervenants) == len(lst_discours):
                    for i in range(len(lst_discours)):
                        interv = lst_intervenants[i]
                        dic[interv][0].append(i) 
                        dic[interv][1].append(lst_discours[i])
                    text= dic
                else:
                    try:
                        inter = list(a)
                        inter = [inte.lower() for inte in inter]
                        lst_intervenants =  [word.replace('\n','').split(',')[0].upper() for word in text if word.replace('\n','').lower() in inter or word.replace('\n','').split(',')[0].lower() in inter ]
                        lst_discours = [word for word in text if not word.replace('\n','').isupper() and not word.replace('\n','').startswith('Source') and not word.replace('\n','').lower() in inter and not word.replace('\n','').startswith("/")]
                        for i in range(len(lst_discours)):
                            interv = lst_intervenants[i]
                            dic[interv][0].append(i) 
                            dic[interv][1].append(lst_discours[i])
                        text= dic
                    except:
                        text = response.css('span.field--name-field-texte-integral').extract()[0]
                        para = text.split('\n')
                        para = [x + ' ' for x in para if x!='']
                        discours = ['ERREUR SUR LITW']
                        for p in para:
                            if not p.startswith('Source'):
                                discours.append(p)
                        text = ' '.join(discours)
                        text = BeautifulSoup(text,'lxml').text
                        text = text.encode("utf-8")
            else:
                text = response.css('span.field--name-field-texte-integral').extract()[0]
                para = text.split('\n')
                para = [x + ' ' for x in para if x!='']
                discours = []
                for p in para:
                    if not p.startswith('Source'):
                        discours.append(p)
                text = ' '.join(discours)
                text = BeautifulSoup(text,'lxml').text
                text = text.encode("utf-8")
        else:
            text = ''
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
            'Tags':tags,
            'Texte': text ,
            'Lien':response.request.url
        }