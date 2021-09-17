import numpy as np
#los documentos deben ser siempre vectores de strings

def cantidadApariciones(termino, doc):
	res = 0
	for t in doc:
		if (t == termino): res = res +1
	return res
	
def cantidadDocumentosQueContienen(document_set, term):
	#res = 1 #para evitar divisiones por 0 puede llegar a ser util
	res = 0
	for d in document_set:
		if (term in d): res = res+1
	return res

def term_frecuency(doc, term): 
	return cantidadApariciones(term, doc)/ len(doc)

def inverse_term_frecuency(document_set, term):
	return np.log(len(document_set)/cantidadDocumentosQueContienen(document_set, term))

def tfidf(document_set, doc, term): #Esta es la posta
	return term_frecuency(doc, term) * inverse_term_frecuency(document_set, term)

D = [["this" ,"is" ,"a", "a" ,"sample"], ["this" ,"is" ,"another","another" ,"example","example","example"]]
print(term_frecuency(D[0], "this"))
print(term_frecuency(D[1], "example"))
print(inverse_term_frecuency(D, "example"))
print(tfidf(D, D[0], "example"), tfidf(D, D[1], "example"))