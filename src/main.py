
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import CountVectorizer

""" #1: Levantamo el datacet """
fake = pd.read_csv("../archive/Fake.csv")
true = pd.read_csv("../archive/True.csv")

# ponemos los dos en uno
fake["label"] = 1
true["label"] = 0
df = pd.concat([fake, true], ignore_index = True)

#print(df.text)

""" #2: Limpiamo el tecsto """


def clean_text(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    # agregar otras...
    return text

df.title = df.title.map(clean_text)
df.text = df.text.map(clean_text)

""" #3: Vectorizamos el texto """

# removemos palabras con muy alta o muy baja frecuencia. ademas, removemos las "stop words" del inglés
# (palabras como 'the', 'a', 'he', 'her', etc.)
MAX_FREQ_THRESHOLD = 0.99
MIN_FREQ_THRESHOLD = 0.01

# chusmear https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
cv = CountVectorizer(
    stop_words='english',
    max_df=MAX_FREQ_THRESHOLD,
    min_df=MIN_FREQ_THRESHOLD  # seguro hay mas parametros piolas para usar
)
data_cv = cv.fit_transform(df.text)
# print(cv.get_feature_names())  # vocabulario
# print(len(cv.get_feature_names())) # tamaño del vocabulario
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = df.index

print(data_dtm)

"""#4: experimentamos sobre la matriz de texto vectorizado (data_dtm) """
# ...
