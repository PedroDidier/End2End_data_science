import bentoml
import numpy as np
from bentoml.io import NumpyNdarray

gender_clf_runner = bentoml.sklearn.get("rf_gender_classifier:latest").to_runner()

svc = bentoml.Service("gender_classifier", runners=[gender_clf_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    result = gender_clf_runner.predict.run(input_series)
    return result