from PIL import Image
import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic

def get_loaders(filepath):
