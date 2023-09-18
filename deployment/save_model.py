import bentoml
import sklearn
import pickle

with open('best_model.pkl', 'rb') as file:
    clf = pickle.load(file)

saved_model = bentoml.sklearn.save_model("rf_gender_classifier", clf)