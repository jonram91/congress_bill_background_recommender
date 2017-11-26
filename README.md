<h1> Background Report Recommender for Congressional Bills</h1>

This project builds a recommender system that creates a list of
recommended readings for any bill currently being debated or
was previously debated in Congress.

The Flask app allows a user to input any summary text of a Congressional
bill to produce a list of Congressional Research Service reports that will
provide a background on the bill itself and the wider topics contained.

<h2> 3 Sections</h2>

1. Data Acquisition, Cleaning, and MongoDB
2. Modeling
3. Flask App

NB: Be sure to [download the necessary files from this link](https://drive.google.com/file/d/1rBueZfFXD5m89x3GnzBDrrksYpSf_W3D/view?usp=sharing) and save it in the "flask" folder.


<h3> Data Acquisition </h3>

[Using the "wget" package,](http://www.gnu.org/software/wget/) I was able
to download thousands of PDFs from the [Federation of American Scientists'
website.](https://fas.org/sgp/crs/)

```shell
wget -r https://fas.org/sgp/crs/

```
NB: This may take a while.

<h4> Cleaning </h4>

Usually it is notoriously difficult to parse information from PDFs.
Luckily, the [textract](http://textract.readthedocs.io/en/stable/) 
package allows you to parse a PDF in 2 lines of code.

The initial cleaning involved removing stop words (including some custom words)
and symbols that I found in the majority of the PDFs.

Finally, everything was stored in a MongoDB database.

<h3> Modeling </h3>

After doing some initial cleaning, I tokenized the text using 
spacy, lemmatizing the tokens along the way.

I tried several combinations of vectorizers, models, and distance
metrics for the Nearest Neighbors recommender and ultimately decided
on a count vectorizer with the NMF model using cosine distance.

<h3> Flask </h3>

The best NMF model was pickled and used as part of the 
backend of the app.
