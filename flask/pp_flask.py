import flask
import numpy as np
import pickle
from text_handling import *

#------Call the Text Handler, Recommender, CountVectorizer,  --------#

def unpickle_dis(dis):
    pkl_file = open(dis, 'rb')

    data = pickle.load(pkl_file)

    pkl_file.close()

    return data

NMF = unpickle_dis('nmf.pkl')

X = unpickle_dis('X.pkl')

nmf_data = unpickle_dis('nmf_data.pkl')

Rec = unpickle_dis('rec.pkl')

Handler = unpickle_dis('handler.pkl')

CV = unpickle_dis('cv.pkl')


#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = flask.Flask(__name__)

# Homepage
@app.route("/")
def viz_page():
    """
    Homepage: serve our visualization page, awesome.html
    """
    with open("pyedpapyr.html", 'r') as viz_file:
        return viz_file.read()

# Get an example and return it's score from the predictor model
@app.route("/recommend", methods=["POST"])
def recommend():
    """
    When A POST request with json data is made to this uri,
    Read the example from the json, predict probability and
    send it with a response
    """
    # Get decision score for our example that came with the request

    data = flask.request.json

    recommendations = Rec.get_recommendations(new_article=data['article'], model = NMF, vectorizer=CV,
                       training_vectors = NMF.transform(X), n_neighbors = 10, method = 'nmf_cv')

    article_names = []


    for rec in recommendations:

        article_names.append(Handler.doc_names[rec])

    #rec_string = ' '.join(article_names)

    # Put the result in a nice dict so we can send it as json
    results = {"recommendations": article_names}
    print(results)
    return flask.jsonify(results)

#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
app.run(host='0.0.0.0', port = 8000)
app.run(debug=True)
