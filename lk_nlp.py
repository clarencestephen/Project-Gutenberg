def token(text):
    nopundig = text.translate(pun).translate(dig).lower()
    tokens = word_tokenize(nopundig)
    stops = [i for i in tokens if i not in swords]
    lems = [lem.lemmatize(i) for i in stops]
    return lems

def vec_nmf(text, vec='cv', vec_max=0.9, vec_min=0.05, vec_ngram=(1, 2), nmf_ncomps=10, nmf_nwords=10):
    if vec == 'cv':
        vec = CountVectorizer(tokenizer=token, max_df=vec_max, min_df=vec_min, ngram_range=vec_ngram)
    
    elif vec == 'tfidf':
        vec = TfidfVectorizer(tokenizer=token, max_df=vec_max, min_df=vec_min, ngram_range=vec_ngram)
    cnt = vec.fit_transform(text)
    
    nmf = NMF(n_components=nmf_ncomps)
    top = nmf.fit_transform(cnt)
    
    df = pd.DataFrame(nmf.components_, columns=vec.vocabulary_.keys())
    words = df.apply(lambda i: list(i.nlargest(nmf_nwords).index), axis=1).apply(pd.Series)
    comps = df.apply(lambda i: list(i.nlargest(nmf_nwords)), axis=1).apply(pd.Series)
    
    _, ax = plt.subplots(figsize=(nmf_nwords * 2, nmf_ncomps * 0.6));
    sns.heatmap(comps, annot=words, fmt='', ax=ax, cmap='Blues')
    
    return vec, cnt, nmf, top
	
vec_nmf(text);