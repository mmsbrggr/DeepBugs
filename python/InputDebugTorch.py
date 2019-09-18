import sys
import json
import numpy as np
import Util
import torch
import torch.nn as nn
import LearningDataSwappedArgs
from tools.projected_gradient_descent import projected_gradient_descent, get_distance, get_most_similar_from_dict

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
    model = torch.load(model_file)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    # Load data to predict
    learning_data = LearningDataSwappedArgs.LearningData()
    learning_data.resetStats()
    xs, ys, code_pieces = prepare_xy_pairs(["data/playground/calls_input.json"], learning_data)

    x = torch.from_numpy(xs[:1]).float()
    y = torch.tensor([[0]]).float()
    adv = projected_gradient_descent(model, x.clone(), y.clone(), nn.BCELoss(), 1000, 0.5, 5)

    x_pred = model(x).numpy()[0]
    adv_pred = model(adv).numpy()[0]

    x = x.numpy()[0]
    adv = adv.numpy()[0]

    assert(np.array_equal(x[:200], adv[:200]))
    assert(np.array_equal(x[600:], adv[600:]))

    x_arg1 = x[200:400]
    x_arg2 = x[400:600]
    adv_arg1 = adv[200:400]
    adv_arg2 = adv[400:600]

    dinf, deu, dcos = get_distance(x, adv)
    dinf_arg1, deu_arg1, dcos_arg1 = get_distance(x_arg1, adv_arg1)
    dinf_arg2, deu_arg2, dcos_arg2 = get_distance(x_arg2, adv_arg2)

    print("x prediction: %1.5f" % x_pred)
    print("adv prediction: %1.5f" % adv_pred)
    print()
    print("Inf distance: %3.5f" % dinf)
    print("Euclidean distance: %3.5f" % deu)
    print("Cosine distance: %3.5f" % dcos)
    print()
    print("Inf distance arg1: %3.5f" % dinf_arg1)
    print("Euclidean distance arg1: %3.5f" % deu_arg1)
    print("Cosine distance arg1: %3.5f" % dcos_arg1)
    print()
    print("Inf distance arg2: %3.5f" % dinf_arg2)
    print("Euclidean distance arg2: %3.5f" % deu_arg2)
    print("Cosine distance arg2: %3.5f" % dcos_arg2)

    np.savetxt("adversarial_input.txt", adv)

    # Load word embeddings to find most similar arguments
    with open("token_to_vector.json") as f:
        name_to_vector = json.load(f)

    adv_arg1_name = get_most_similar_from_dict(name_to_vector, adv_arg1, [code_pieces[0].arguments[0]])
    adv_arg2_name = get_most_similar_from_dict(name_to_vector, adv_arg2, [code_pieces[0].arguments[1]])

    print()
    print(adv_arg1_name)
    print(adv_arg2_name)

    # Predict and report to user
    # with torch.no_grad():
    #    xs = torch.from_numpy(xs).float()
    #    result = model(xs)
    #    print(result)
