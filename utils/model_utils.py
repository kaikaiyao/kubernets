import copy
import os
import pickle


def save_finetuned_model(model, path, filename):
    model_cpu = copy.deepcopy(model).cpu()
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    with open(save_path, 'wb') as f:
        pickle.dump(model_cpu, f)


def load_finetuned_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def clone_model(model):
    return copy.deepcopy(model)