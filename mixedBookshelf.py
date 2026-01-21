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

trainSetName = "CircuitGen"
trainSetNameFixed = "CircuitGenFixedMacro"

def main():
    bookshelfDir = f"data-gen/outputs/v2.61/{trainSetName}"
    bookshelfDir2 = f"data-gen/outputs/v2.61/{trainSetNameFixed}"
    # pyGFile = "data-gen/outputs/v2.61/00000110.pickle"

    fileId = 0
    for i in range(100, 2501, 100):  # 7501
        pyGFile = f"data-gen/outputs/v2.61/{i:08d}.pickle"
        # pyGFile = f"/home/pc/data/cjq/work/chipdiffusion-main/data-gen/outputs/v2.61/{i:08d}.pickle"
        Logger.printPlace(f"deal with PyG file {pyGFile} ...")
        data = load_pickle(pyGFile)
        for j in range(len(data)):
            fileName = f"{trainSetName}{fileId:04d}"
            fileName2 = f"{trainSetNameFixed}{fileId:04d}"
            dataBase = DataBase.DataBase()
            dataBase.readpyG(data[j][0],data[j][1])
            auxFile,nodeInThisLayer = dataBase.generateBookShelf(bookshelfDir, fileName, curlayer = -1, moveMacro = True)
            auxFile,nodeInThisLayer = dataBase.generateBookShelf(bookshelfDir2, fileName2, curlayer = -1, moveMacro = False)
            fileId += 1
        # dataBase.auxFileName = auxFileName


if __name__ == "__main__":
    main()