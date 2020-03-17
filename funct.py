#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from nltk.tokenize import word_tokenize
from spacy.tokenizer import Tokenizer

def cleanToken(x):
    """
        Fonction permettant de nettoyer et de tokenizer un texte
    """
    import string
    pct = string.punctuation+'...'+'\x92'+'«'+'»'+'``'+"''"+'``'
    x = x.replace('\xa0','').replace('\x85','').replace('\x96','')
    x = "".join(filter(lambda y: not y.isdigit(), x))
    sw = list(fr_stop)
    tokens = [str(w).lower() for w in word_tokenize(x, language='french')]
    tokens = [w for w in tokens if w not in pct]
    tokens = [w for w in tokens if w not in sw]
    return tokens
def count_punct(tokens):
    """
        Permet de compter la ponctuation
    """
    import string
    pct = string.punctuation +'...'+'\x92'+'«'+'»'+'``'
    cpt = 0
    for x in tokens:
        if x in pct:
            cpt+=1
    return cpt
def count_stopwords(tokens):
    """
        compte le nombre de stopwords à l'aide de spacy à partir de tokens
    """
    sw = list(fr_stop)
    return len([word for word in tokens if word.lower() in sw])
def Hapaxlegomena(tokens):
    """
        Compte le nombre de mot unique
    """
    s = pd.DataFrame(Counter(tokens).items(),columns=['Mot','nb'])
    return len(s[s.nb ==1])
def Hapaxdislegomena(tokens):
    """
        Compte le nombre de mot présent deux fois seulement
    """
    s = pd.DataFrame(Counter(tokens).items(),columns=['Mot','nb'])
    return len(s[s.nb ==2])
def extractPos(x):
    """
        Extrait le POS de chaque texte. A voir s'il y a un moyen de pas avoir à taper toutes les variables à la main
    """
    doc = nlp(x)
    lst_pos = [token.pos_ for token in doc]
    c = Counter(lst_pos)
    return [c['NOUN'],c['DET'],c['PUNCT'],c['ADJ'],c['ADP'],c['PRON'],c['VERB'],c['CCONJ'],c['NUM'],c['PROPN'],c['ADV'],c['SCONJ'],c['AUX'],c['INTJ']]

def nbArt(x):
    """
        Nombre d'article dans un document
    """
    doc = nlp(x)
    lst_pos = [token.tag_ for token in doc if token.tag_.split('|')[-1] == 'PronType=Art']
    return len(lst_pos)
def f_mesure(NbToken,nbnom,nbadj,nbprep,nbart,nbpro,nbverb,nbadv,nbint):
    """
        Inspiré par Heylighen and Dewaele, 2002
        Définit ici : https://www.cs.uic.edu/~liub/publications/EMNLP-2010-blog-gender.pdf comme étant
        F = 0.5 * [(freq.nom + freq.adjectif + freq.preposition + freq.article) - (freq.pronom + freq.verbe + freq.adverbe +
        freq.interjection) +100]
    """
    nbnom,nbadj,nbprep,nbart,nbpro,nbverb,nbadv,nbint = nbnom/NbToken,nbadj/NbToken,nbprep/NbToken,nbart/NbToken,nbpro/NbToken,nbverb/NbToken,nbadv/NbToken,nbint/NbToken
    nbnom,nbadj,nbprep,nbart,nbpro,nbverb,nbadv,nbint = nbnom*100,nbadj*100,nbprep*100,nbart*100,nbpro*100,nbverb*100,nbadv*100,nbint*100
    return 0.5*((nbnom + nbadj + nbprep + nbart)-(nbpro + nbverb + nbadv + nbint)+100)
def print_nine_dist(lst,df):
    """
        Input : 
            lst : liste de variables à print
            df : dataframe contenant ces valeurs
        Output : 
            Jusqu'à 9 displot des variables (3*3)
    """
    plt.figure(figsize=(10,14))
    for j in range(1,10):
        if len(lst)>j-1:
            var = lst[j-1]
        else:
            break
        q99= df[var].quantile(0.99)
        plt.subplot(330+j)
        sns.distplot(df[df.sexe == 2][var],color = 'red')
        sns.distplot(df[df.sexe == 1][var])
        plt.xlim(left=-0.1,right=q99)
    plt.subplots_adjust(wspace = 1)
    plt.show()
def NbSyllables(x):
    """
        Input : document
        Output : nbVoyelles
    """
    voyelles= 'aeiouy'
    return len([word for word in x if word.lower() in voyelles])
def flesh_reading_ease(ASL,ASW):
    """
        Score de lisibilité du texte : Plus il est élévé plus il est facile à comprendre
        FRE  = 206.835 - (1.015 × ASL) - (84.6 × ASW)
        ASL = NbToken/NbPhrases
        ASW = NbSyllables/NbToken
    """
    return 206.835 - (1.015 * ASL) - (84.6 * ASW)
def sent_detector_mano(x):
    """
        Détection de phrase à la main.
        Input : document
        Output : liste de phrases
        Problème avec les phrases finissant par : entrainant souvent une liste. 
        De même avec ;. Tentative réalisée
        
    """
    lst =[]
    phrase = []
    i = 0
    for caractere in x: 
        if not (caractere == ' ' and len(phrase) == 0) :
            phrase.append(caractere)
        if caractere in '?!.:;':
            if caractere == ':':
                if x[i+1].isupper() or x[i+2].isupper() or x[i+1] == '-' or x[i+2] == '-':
                    lst.append(''.join(phrase))
                    phrase = []
            elif caractere == ';':
                if x[i+1].isupper() or x[i+2].isupper() or x[i+1] == '-' or x[i+2] == '-':
                    lst.append(''.join(phrase))
                    phrase = []
            elif phrase != '.' or phrase != '?' or phrase != '!':
                lst.append(''.join(phrase))
                phrase = []
        i+=1
    return lst
def extraire_nb_mot(x):
    """
        Input : Document
        Output : nombre de mot
    """
    import string
    pct = string.punctuation+'...'+'\x92'+'«'+'»'+'``'+"''"+'``'
    x = x.replace('\xa0','').replace('\x85','').replace('\x96','')
    x = "".join(filter(lambda y: not y.isdigit(), x))
    tokens = [str(w).lower() for w in word_tokenize(x, language='french')]
    tokens = [w for w in tokens if w not in pct]
    return len(tokens)

def remove_source(x):
    """
      Input : Texte 
      Output : Texte sans le dernier paragraphe
    """
    x =  x[:x.find('Source:')]
    return x[:x.find('Source http')]


def Pron_Type(x,nlp):
    """
    Input : 
          x : Texte
          nlp : Spacy pre entrained modele
    Output: 
          Nombre de pronom personnel première personne du singulier
    """
    sing1=[]
    doc = nlp(x)  
    for token in doc:
        attributes = ' '.join(' '.join(token.tag_.split('|')).split('__'))
        if 'PRON' in attributes and 'Number=Sing' in attributes and 'Person=1' in attributes:
            sing1.append(token)
    return len(sing1)

def Pron_Type_Plur(x,nlp):
    """
    Input : 
          x : Texte
          nlp : Spacy pre entrained modele
    Output: 
          Nombre de pronom perso PLUR 1
    """
    plu1=[]
    doc = nlp(x)  
    for token in doc:
        attributes = ' '.join(' '.join(token.tag_.split('|')).split('__'))
        if 'PRON' in attributes and 'Number=Plur' in attributes and 'Person=1' in attributes:
            plu1.append(token)
    return len(plu1)

def Verb_Tens(x,nlp):
    """
      Input:
        x : texte
        nlp : nlp : Spacy pre entrained modele
      Output : 
        Nombre de verbes à 4 temps différents
    """
    present=[]
    passe=[]
    futur=[]
    imp=[]
    doc = nlp(x)  
    for token in doc:
        attributes = ' '.join(' '.join(token.tag_.split('|')).split('__'))
        if 'AUX' in attributes:
            if 'Tense=Pres' in attributes:
                present.append(token)
            if 'Tense=Imp' in attributes:
                imp.append(token)
            if 'Tense=Fut' in attributes:
                futur.append(token)
            if 'Tense=Past' in attributes:
                passe.append(token)
        elif 'VERB' in attributes:
            if 'Tense=Pres' in attributes:
                present.append(token)
            if 'Tense=Imp' in attributes:
                imp.append(token)
            if 'Tense=Fut' in attributes:
                futur.append(token)
            if 'Tense=Past' in attributes:
                passe.append(token)
    return len(present), len(passe), len(futur), len(imp)

def Quest(x,nlp):
    """
      Input:
        x : texte
        nlp : nlp : Spacy pre entrained modele
      Output : 
        Nombre de '?'
    """
    doc = nlp(x) 
    quest = [w for w in doc if str(w) in ['?']]
    return len(quest)

def Excl(x,nlp):
    """ Input:
        x : texte
        nlp : nlp : Spacy pre entrained modele
      Output : 
        Nombre de '!'
    """
    doc = nlp(x) 
    excl = [w for w in doc if str(w) in ['!']]
    return len(excl)

