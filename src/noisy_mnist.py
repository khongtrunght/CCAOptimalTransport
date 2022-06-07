import numpy as np
np.set_printoptions(suppress=True)


def fetch(url):
    import requests
    import gzip
    import os
    import hashlib
    import numpy
    fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            dat = f.read()
    else:
        with open(fp, "wb") as f:
            dat = requests.get(url).content
            f.write(dat)
    return numpy.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()


X_train = fetch(
    "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_train = fetch(
    "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch(
    "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_test = fetch(
    "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]


np.save("data/X_train.npy", X_train)
np.save("data/Y_train.npy", Y_train)
np.save("data/X_test.npy", X_test)
np.save("data/Y_test.npy", Y_test)
