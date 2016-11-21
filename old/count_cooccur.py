def method2(reviews):
	from collections import defaultdict
	# remember to include the other import from the previous post
	 
	com = defaultdict(lambda : defaultdict(int))

	terms_only = []
	for review in reviews:
	    terms_only += [term for term in getWords(review) if term not in stop]
 
    # Build co-occurrence matrix
    for i in range(len(terms_only)-1):            
        for j in range(i+1, len(terms_only)):
            w1, w2 = sorted([terms_only[i], terms_only[j]])                
            if w1 != w2:
                com[w1][w2] += 1

def getWords(text):
    return re.compile('\w+').findall(text)