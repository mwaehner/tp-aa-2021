import numpy as np
#los documentos deben ser siempre vectores de strings

def inverse_term_frecuency_calculator(document_set): #esto segurísimo se puede hacer más rápido pero paja
	terms = [item for sublist in document_set for item in sublist] #"aplano la lista de strings"
	print(len(terms))
	for i in range (0,100): print(terms[i])
	terms = set(terms) # saco repetidos
	print(terms)
	print(len(terms))
	res = {}
	i = 0
	for t in terms:
		res[t] = inverse_term_frecuency(document_set, t)
		print (i)
		i = i+1
	print(res)
	return res

def vectorizar(document_set):
	res = []
	inverse_term_frecuency_dic = inverse_term_frecuency_calculator(document_set) #esto no depende del documento en particular
	for d in document_set:
		d_vector = []
		for t in inverse_term_frecuency_dic.keys():
			d_vector.append(inverse_term_frecuency_dic[t] * term_frecuency(d, t))
		res.append(d_vector)
	return res

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

D = [["this" ,"is" ,"a", "a" ,"sample"], ["this" ,"is" ,"another","another" ,"example","example","example"]] #Esto es para testear que anda (anda)
# print(term_frecuency(D[0], "this"))
# print(term_frecuency(D[1], "example"))
# print(inverse_term_frecuency(D, "example"))
#print(tfidf(D, D[0], "example"), tfidf(D, D[1], "example"))
#print(vectorizar(D))