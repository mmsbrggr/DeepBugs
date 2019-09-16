import sys
import json
from keras.models import load_model
import numpy as np
import Util
import LearningDataSwappedArgs
import foolbox

name_embedding_size = 200
file_name_embedding_size = 50
type_embedding_size = 5

def prepare_xy_pairs(data_paths, learning_data):
    xs = []
    ys = []
    code_pieces = []  # keep calls in addition to encoding as x,y pairs (to report detected anomalies)

    for code_piece in Util.DataReader(data_paths):
        learning_data.code_to_xy_pairs(code_piece, xs, ys, name_to_vector, type_to_vector, node_type_to_vector,
                                       code_pieces)
    xs = np.array(xs)
    ys = np.array(ys)
    return [xs, ys, code_pieces]


if __name__ == '__main__':

    # Load word embeddings
    with open("token_to_vector.json") as f:
        name_to_vector = json.load(f)
    with open("type_to_vector.json") as f:
        type_to_vector = json.load(f)
    with open("node_type_to_vector.json") as f:
        node_type_to_vector = json.load(f)

    # Load model from file
    model_file = sys.argv[1]
    model = load_model(model_file)

    # Load data to predict
    learning_data = LearningDataSwappedArgs.LearningData()
    learning_data.resetStats()
    xs, ys, code_pieces = prepare_xy_pairs(["data/playground/calls_input.json"], learning_data)

    criterion = foolbox.criteria.OriginalClassProbability(0.5)
    fmodel = foolbox.models.KerasModel(model, bounds=(-10, 10))
    attack = foolbox.attacks.FGSM(fmodel, criterion)
    adversarial = attack(xs[1], 0)

    # Predict and report to user
    # result = model.predict(xs)
    # print(result)