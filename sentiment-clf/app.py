from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from model import NLPModel

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('query')

model = NLPModel()
clf_path = 'lib/models/SentimentClassifier.pkl'
with open(clf_path, 'rb') as f:
    model.clf = pickle.load(f)

vec_path = 'lib/models/TFIDFVectorizer.pkl'
with open(vec_path, 'rb') as f:
    model.vectorizer = pickle.load(f)

class PredictSentiment(Resource):

    def get(self):
        args = parser.parse_args()
        user_query = args['query']

        uq_vectorized = model.vectorizer_transform(np.array([user_query]))
        pred = model.predict(uq_vectorized)
        pred_proba = model.predict_proba(uq_vectorized)

        output = {'args': args,
                  'label':      pred.tolist()[0],
                  'confidence': pred_proba[0]}

        return output


class HealthCheck(Resource):
    def get(self):
        return {'status': 'OK'}

# Setup URL to api resource
api.add_resource(PredictSentiment, '/predict')
api.add_resource(HealthCheck, '/health')


if __name__ == '__main__':
    app.run(debug=True)
