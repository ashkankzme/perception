import json


def saveObjectsToJsonFile(objects, fileName):
    with open(fileName, 'w') as outfile:
        json.dump([ob.__dict__ for ob in objects], outfile, indent=4)


def saveToJsonFile(data, fileName):
    with open(fileName, 'w') as outfile:
        json.dump(data, outfile, indent=4)


def loadObjectsFromJsonFile(fileName): # todo does this work?
    with open(fileName) as json_file:
        data = json.load(json_file)
        # data = json.load(json_file, object_hook=HumanWorker._decode)

    return data


def getFoldIndices(datasetSize, foldCount, foldIdx):
    foldSize = datasetSize // foldCount
    foldStartIdx = foldSize * foldIdx
    foldEndIdx = min(foldStartIdx + foldSize, datasetSize-1) if foldIdx != foldCount-1 else datasetSize-1

    return foldStartIdx, foldEndIdx