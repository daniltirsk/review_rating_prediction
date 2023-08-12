from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
from preprocessing import *

app = Flask(__name__)

app.config["BASE_URL"] = "urlGoesHere"

LGBM_THRESH = 0.556235
BERT_THRESH = 0.384609
BASE_MODEL = "lvwerra/distilbert-imdb"

id2label = {0: "Negative", 1: "Positive"}

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# define lemmatizer and stop_words to be removed
lemmatizer = WordNetLemmatizer()

stop_words = list(set(stopwords.words('english')))
stop_words.remove('no')
stop_words.remove('not')

vectorizer = joblib.load("./models/BoW_lgbm_vectorizer.pkl")
lgbm_model = joblib.load("./models/BoW_lgbm.pkl")

model_path = './models/distilbert-review-rating/checkpoint-7035'
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
bert_model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)


def calc_ratings(reviews, method):
    result = {}
    if method == 'lgbm' or method == 'both':
        reviews_lgbm = preprocess_lgbm(reviews, lemmatizer, stop_words)
        reviews_lgbm = [" ".join(r) for r in reviews_lgbm]
        reviews_lgbm = vectorizer.transform(reviews_lgbm)
        reviews_lgbm = reviews_lgbm.astype(np.float64)
        preds = lgbm_model.predict(reviews_lgbm)
        ratings, labels = process_output(preds, LGBM_THRESH)
        result["lgbm"] = {"ratings": ratings, "labels": labels}

    if method == 'bert' or method == 'both':
        reviews_bert = preprocess_bert(reviews, tokenizer)
        preds = bert_model(**reviews_bert)
        preds = [p[0] for p in preds[0].tolist()]
        ratings, labels = process_output(preds, BERT_THRESH)
        result["bert"] = {"ratings": ratings, "labels": labels}

    return result


@app.route('/', methods=["GET", "POST"])
def get_main():
    results = []

    if request.method == 'POST':
        reviews = [request.form["content"]]
        results = calc_ratings(reviews, method='both')
        results = [{"method": r, "rating": results[r]['ratings'][0], "label": id2label[results[r]['labels'][0]]}
                   for r in results]

    return render_template('main_form.html', results=results)


@app.route('/get_ratings', methods=["POST"])
def get_ratings():
    data = request.get_json()

    method = data['method']
    reviews = data['reviews']

    result = calc_ratings(reviews, method)
    return result


if __name__ == "__main__":
    app.run(debug=True)
