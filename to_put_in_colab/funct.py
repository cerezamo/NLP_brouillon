#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import multiprocessing as mp
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from nltk.tokenize import word_tokenize
from spacy.tokenizer import Tokenizer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from collections import Counter
import spacy
import string
from sklearn.metrics import (
    recall_score,
    accuracy_score,
    precision_score,
    roc_auc_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    scorer)
nlp = spacy.load('fr_core_news_md') 

feel = pd.read_csv('https://raw.githubusercontent.com/cerezamo/NLP_project_MHMP/master/to_put_in_colab/FEEL.csv',sep=';')
feel.set_index('id',inplace=True)
def cleanToken(x):
    """
        Fonction permettant de nettoyer et de tokenizer un texte
    """
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

def FastCleaner(x,lst):
  """
    Input :
      x : Tokens
      lst : liste de mot à enlever
    Output : 
      liste de tokens cleaner
  """
  return [word for word in x if word not in lst]

def cleanTokenLemme(x,lst =[]):
    """
        Input : 
          x : texte
          lst : lst de mot a clean en plus
        Output : 
          tokens tout cleaned
        Fonction permettant de nettoyer et de tokenizer un texte tout en le lemmatizant
    """
    pct = string.punctuation+'...'+'\x92'+'«'+'»'+'``'+"''"+'``'
    x = [word.lemma_ for word in nlp(x)]
    x = ' '.join(x)
    x =str(x)
    x = x.replace('\xa0','').replace('\x85','').replace('\x96','')
    x = "".join(filter(lambda y: not y.isdigit(), x))
    sw = list(fr_stop)
    tokens = [str(w).lower() for w in word_tokenize(x, language='french')]
    tokens = [w for w in tokens if w not in pct]
    tokens = [w for w in tokens if w not in sw]
    tokens = [w for w in tokens if not len(w) <=2]
    tokens = [w for w in tokens if w not in lst]
    return tokens
def check_polarity(tokens):
    """
        Analyse de sentiment de chaque mots
        Input : Tokens
        Output :  (%positive,%negative,%non trouvé) en fréquen
    """
    pos = feel[feel.polarity == 'positive'].word.values
    neg = feel[feel.polarity == 'negative'].word.values
    nb_pos = [word for word in tokens if word in pos]
    nb_neg = [word for word in tokens if word in neg]
    return [len(nb_pos)/len(tokens),len(nb_neg)/len(tokens),1 - (len(nb_pos) + len(nb_neg))/len(tokens)]
def extraction_emotion(tokens):
    """
        Permet d'extraire le nombre de mot classé dans 6 émotions différentes.
        Input : Tokens
        Output : (Joie,peur,tristesse,colère,surprise,dégoût) en fréquence
    """
    joie = [w for w in tokens if w in feel[feel.joy == 1].word.values]
    peur = [w for w in tokens if w in feel[feel.fear == 1].word.values]
    tristesse = [w for w in tokens if w in feel[feel.sadness == 1].word.values]
    colere = [w for w in tokens if w in feel[feel.anger == 1].word.values]
    surprise = [w for w in tokens if w in feel[feel.surprise == 1].word.values]
    degout = [w for w in tokens if w in feel[feel.disgust == 1].word.values]
    return [len(joie)/len(tokens),len(peur)/len(tokens),len(tristesse)/len(tokens),len(colere)/len(tokens),len(surprise)/len(tokens),len(degout)/len(tokens)]
def predictions(model,X_train,X_test,y_train,y_test):
    """
        Input :
            model : Algorithme de sklearn avec les paramètres choisit ou par défaut
            X_train,X_test,y_train,y_test : dataset découpé à l'aide de train_test_split
        Output : 
            Classification_report + Confusion_matrix + ROC_curve + (si possible feature importance)
    """
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    pred_prob = model.predict_proba(X_test)
    print(model)
    print ("Classification report :",classification_report(y_test,pred))
    print ("Accuracy : ",accuracy_score(y_test,pred))
    cm = confusion_matrix(y_test,pred)
    ROC = roc_auc_score(y_test,pred) 
    print ("AUC : ",ROC)
    fpr,tpr,thresholds = roc_curve(y_test,pred_prob[:,1])
    plt.figure(figsize=(12,10))
    plt.subplot(221)
    sns.heatmap(cm/np.sum(cm), annot=True, 
            fmt='.2%', cmap='Blues').set_title('Matrice de confusion')
    plt.subplot(222)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % ROC)
    plt.plot([0,1],[0,1],color='red')
    plt.title('Courbe ROC')
    plt.show()
def model_report(model,X_train,X_test,y_train,y_test) :
  """
      Input : 
          model : Algorithme de sklearn
          X_train,X_test,y_train,y_test : dataset découpé à l'aide de train_test_split
      Output : 
          DataFrame avec AccScore,RecallScore,Precision,F1+auc
  """
  model.fit(X_train,y_train)
  predictions  = model.predict(X_test)
  accuracy     = accuracy_score(y_test,predictions)
  recallscore  = recall_score(y_test,predictions)
  precision    = precision_score(y_test,predictions)
  roc_auc      = roc_auc_score(y_test,predictions)
  f1score      = f1_score(y_test,predictions)     
  df = pd.DataFrame({ "Accuracy_score"  : [accuracy],
                    "Recall_score"    : [recallscore],
                    "Precision"       : [precision],
                    "f1_score"        : [f1score],
                    "Area_under_curve": [roc_auc],
                    })
  return df
def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors
def add_features(df):
    from multiprocessing import Pool
    nlp = spacy.load('fr_core_news_md') 
    df['NbCleanToken'] = df.Texte.apply(lambda x : len(cleanToken(x)))
    df['NbSyllables'] = df.Texte.apply(NbSyllables)
    df['NbMot'] = df.Texte.apply(extraire_nb_mot)
    df['Phrases'] = df.Texte.apply(sent_detector_mano)
    df['NbPhrases'] = df.Texte.apply(lambda x:len(sent_detector_mano(x)))
    df['CleanToken'] = df.Texte.apply(cleanToken)
    df.CleanToken = df.apply(lambda row : FastCleaner(row.CleanToken,cleanFast),axis=1)
    df['NbCleanToken']=df.CleanToken.apply(len)
    df['NbPonct'] = df.Texte.apply(count_punct)
    df['NbSw'] = df.Token.apply(count_stopwords)
    df['Hapaxlegomena']=df.CleanToken.apply(Hapaxlegomena)
    df['Hapaxdislegomena']= df.CleanToken.apply(Hapaxdislegomena)
    df['UniqueWordTx']= df.CleanToken.apply(lambda x:len(set(x))/len(x))
    #df['RateCleanRaw'] = df.NbCleanToken/df.NbToken
    df['NbNom'],df['NbDet'],df['NbPunct'],df['NbAdj'],df['NbAdp'],df['NbPron'],df['NbVerb'],df['NbCconj'],df['NbNum'],df['NbPropn'],df['NbAdv'],df['NbSCONJ'],df['NbAUX'],df['NbIntj']=zip(*df.Texte.apply(extractPos))
    df['NbArt']= df.Texte.apply(nbArt)
    df['F_mesure'] = df.apply(lambda row: f_mesure(row.NbToken,row.NbNom,row.NbAdj,row.NbAdp,row.NbArt,row.NbPron,row.NbVerb,row.NbAdv,row.NbIntj),axis=1)
    df['PronJe']=df.apply(lambda row : Pron_Type(row.Texte,nlp),axis=1)
    df['PronNous']=df.apply(lambda row : Pron_Type_Plur(row.Texte,nlp),axis=1)
    df['NbPres'],df['NbPast'],df['NbFut'],df['NbImp']  = zip(*df.apply(lambda row : Verb_Tens(row.Texte,nlp),axis=1))
    df['NbQuest']= df.apply(lambda row : Quest(row.Texte,nlp),axis=1)
    df['NbExcl']= df.apply(lambda row  : Excl(row.Texte,nlp),axis=1)
    return df
def para_df(df,func,n_cores = mp.cpu_count()):
    from multiprocessing import Pool
    df_split = np.array_split(df,n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func,df_split))
    pool.close()
    pool.join()
    return df
def add_features2(df):
    """
    Sert pour la parallélisation
    Input : DataFrame
    Output : DataFrame avec les nouvelles colonnes crée
  """
    df['CleanTokensLemme'] = df.apply(lambda row : cleanTokenLemme(row.Texte,cleanFast),axis=1)
    df['PolPos'],df['PolNeg'],df['PolUnk'] = zip(*df.CleanTokensLemme.apply(check_polarity))
    df['FreqJoie'],df['FreqPeur'],df['FreqSad'],df['FreqColere'],df['FreqSurprise'],df['FreqDegout'] = zip(*df.CleanTokensLemme.apply(extraction_emotion))
    return df
def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df
def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids]
    else:
        D = Xtr

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)
def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=25):
    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. '''
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs
def plot_tfidf_classfeats_h(dfs):
    ''' Plot the data frames returned by the function plot_tfidf_classfeats(). '''
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    x = np.arange(len(dfs[0]))
    colors = ["#a86900","#c24e81"]
    j=0
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=14)
        if df.label == 1:
          s = 'Femme'
        else:
          s = 'Homme'
        ax.set_title(s, fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center', color= colors[j])
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
        j+=1
    plt.show()


    

def sent_detector_mano(x):
    """
        Détection de phrase à la main.
        Input : document
        Output : liste de phrases
        Problème avec les phrases finissant par : entrainant souvent une liste. 
        De même avec ;. Tentative réalisée
        
    """
    import pandas as pd
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
def split_document_to_limit(MAX_TOKENS,df):
  import pandas as pd
  lst= []
  for index,row in df.iterrows():
    identifiant = row.Id
    label = row.sexe
    phrase = []
    for token in row.Texte.split(' '):
      if len(phrase) < MAX_TOKENS:
        phrase.append(token)
      else:
        lst += [(identifiant,label,' '.join(phrase),len(phrase))]
        phrase = []
    if len(phrase)>1:
      lst += [(identifiant,label,' '.join(phrase),len(phrase))]
  return pd.DataFrame(lst,columns=['Id','Label','Texte','Length'])
def split_document_to_limit_phrases(MAX_TOKENS,df):
  import pandas as pd
  lst= []
  for index,row in df.iterrows():
      identifiant = row.Id
      label = row.sexe
      phrase = ''
  for phrases in sent_detector_mano(row.Texte):
      if len(phrase.split(' ')) + len(phrases.split(' ')) < MAX_TOKENS:
          phrase+= " " + phrases
      else:
      lst += [(identifiant,label,phrase,len(phrase.split(' ')))]
      phrase = ''
  lst += [(identifiant,label,phrase,len(phrase.split(' ')))]
  phrase = ''
  return pd.DataFrame(lst,columns=['Id','Label','Texte','Length'])
