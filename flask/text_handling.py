from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
from sklearn.neighbors import NearestNeighbors

client = MongoClient('mongodb://localhost:27017/')
db = client.project_4_database

dc_abbreviations = [abbrev for abbrev in db.crs_abbreviations.find()][0]['DC_abbrevs']

stopwords = list(stopwords) + dc_abbreviations


def get_from_mongo(mongo_id, art_name):

    return db.cleaned_pdfs.find({'_id': mongo_id})[0][art_name]


def normalize(sparse_matrix):
    from sklearn.preprocessing import Normalizer
    import numpy as np
    from scipy import sparse

    n = Normalizer()
    n.fit(sparse_matrix.toarray())

    X = n.transform(sparse_matrix.toarray())

    X_sparse_cv = sparse.csr_matrix(X)

    return X_sparse_cv


class recommender():
    def __init__(self, handler):

        self.model_dict = {}
        self.new_article = None
        self.recommendations = []
        self.handler = handler

    def get_recommendations(self, new_article, model, vectorizer, training_vectors, n_neighbors, method,
                            metric='cosine'):

        self.new_article = new_article

        new_vec = model.transform(
            vectorizer.transform([new_article]))

        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm='brute')

        nn.fit(training_vectors)

        results = nn.kneighbors(new_vec)

        self.model_dict[method] = (vectorizer, training_vectors)

        self.recommendations.append(results[1][0])

        return results[1][0]

    def print_results(self, num_result=1):

        num_result -= 1

        print(self.new_article)

        print('\n')

        print('-------------------------------')

        arts = []

        for rec in self.recommendations[num_result]:
            mongo_id = self.handler.doc_ids_sampled[rec]
            art_name = self.handler.doc_names_sampled[rec]

            art = get_from_mongo(mongo_id=mongo_id, art_name=art_name)

            arts.append(art)

        for art in list(set(arts)):
            print(art)

            print('-------------------------------')
            print('\n')


class text_handler():

    def __init__(self):

        self.total_unique = len([article for article in db.cleaned_pdfs.find({'unique' :1})])



    def get_vectorizer(self, vectorizer = 'count', ngram_range = (1 ,2), stop_words = 'english', max_df = 0.6, max_features = 5000):

        if vectorizer == 'count':

            count_vectorizer = CountVectorizer(ngram_range=ngram_range,
                                               stop_words=stop_words,
                                               token_pattern="\\b[a-z][a-z]+\\b",
                                               lowercase=True,
                                               max_df = max_df, max_features = max_features)

            return count_vectorizer

        elif vectorizer == 'tfidf':

            tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range,
                                               stop_words=stop_words,
                                               token_pattern="\\b[a-z][a-z]+\\b",
                                               lowercase=True,
                                               max_df = max_df, max_features = max_features)
            return tfidf_vectorizer

        else:

            print("Type in either 'count' or 'tfidf'")



    def get_all_docs(self):

        all_unique_articles = [article for article in db.cleaned_pdfs.find({'unique' :1})]

        return all_unique_articles



    def get_txt(self, article):

        art_name = [key for key in article][1]

        art_text = article[art_name]

        return (art_name ,art_text)


    def get_all_texts(self):

        unique_arts = self.get_all_docs()

        self.doc_names = []
        self.doc_ids = []

        for art in unique_arts:

            _id = art['_id']

            self.doc_ids.append(_id)


        self.doc_names = []

        for doc in self.doc_ids:

            keys = [key for key in db.cleaned_pdfs.find({'_id': doc})[0]]

            for key in keys:

                if "R" in key:

                    name = key
                    self.doc_names.append(key)

        output = []

        for i ,key in enumerate(self.doc_names):

            output.append(unique_arts[i][key])

        return output