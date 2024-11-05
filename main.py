import numpy as np
import struct
import sklearn.decomposition as skd
from sklearn.neighbors import KNeighborsClassifier
from pyscript import Element
from js import document, window, console
import pickle

# Disable warnings by pyscript appearing in the browser.
import warnings
warnings.filterwarnings("ignore")



with open('train-images.idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    Xtraindata = np.transpose(data.reshape((size, nrows*ncols)))

with open('train-labels.idx1-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    ytrainlabels = data.reshape((size,)) # (Optional)

with open('t10k-images.idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    Xtestdata = np.transpose(data.reshape((size, nrows*ncols)))

with open('t10k-labels.idx1-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    ytestlabels = data.reshape((size,)) # (Optional)
        


pca = skd.PCA()
pca.fit(np.transpose(Xtraindata))

pcak = skd.PCA(n_components=41).fit(np.transpose(Xtraindata))

xtraink = pcak.transform(np.transpose(Xtraindata))
xtestk = pcak.transform(np.transpose(Xtestdata))

KNNCL = KNeighborsClassifier(n_neighbors=3)
KNNCL.fit(xtraink, ytrainlabels)

def get_number():

    print("Training Score: {}".format(KNNCL.score(xtraink, ytrainlabels)))
    print("Testing Score: {}".format(KNNCL.score(xtestk, ytestlabels)))

    document.querySelector(".prediction").hidden = False
    document.querySelector(".result").innerText = "Testing Score: {}".format(KNNCL.score(xtestk, ytestlabels))

    console.log("Training Score: {}".format(KNNCL.score(xtraink, ytrainlabels)))