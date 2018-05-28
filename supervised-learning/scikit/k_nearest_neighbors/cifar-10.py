
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


data = unpickle("../../../data/CIFAR-10/cifar-10-batches-py/data_batch_1")

print(data)



