import sys
import os
import time
import pickle

import dataBase.DataBase as DataBase
import dataBase.Logger as Logger

def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def main():
    bookshelfDir = "data-gen/outputs/v2.61/bookshelf"
    # pyGFile = "data-gen/outputs/v2.61/00000110.pickle"
    fileId = 0
    for i in range(100, 7501, 100):
        pyGFile = f"data-gen/outputs/v2.61/{i:08d}.pickle"
        Logger.printPlace(f"deal with PyG file {pyGFile} ...")
        data = load_pickle(pyGFile)
        for j in range(len(data)):
            fileName = f"CircuitGen{fileId:04d}"
            dataBase = DataBase.DataBase()
            dataBase.readpyG(data[j][0],data[j][1])
            auxFile,nodeInThisLayer = dataBase.generateBookShelf(bookshelfDir, fileName, curlayer = -1, moveMacro = True)
            fileId += 1
        # dataBase.auxFileName = auxFileName


if __name__ == "__main__":
    main()