import random
import math
from utils import saveObjectsToJsonFile, loadObjectsFromJsonFile

'''
Trajectory format, with an example:
WorkerID: XYZ
Gender: A/B/C/NA
Race: ...
...
reactions: [
    {
        "headline": "headline1",
        "reaction": "reaction1",
    }
]
perceptions: [
    {
        "headline": "headline1",
        "perception": "perception1",
    }
]

current headline: 'headline1'
what is the perceived label?
'''
# the trajectories contain three parts; header -- trajectories -- question


class Trajectory(object):
    def __init__(self, inputFrames, header, query, prediction):
        self.inputFrames = inputFrames
        self.header = header
        self.query = query
        self.prediction = prediction


    @staticmethod
    def choiceOf(k, n):
        return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))

    @staticmethod
    def formatInput(inputs):
        formatedInput = []
        for i, trajectory in enumerate(inputs):
            dataPoint = {'X': '', 'y': ''}
            formattedTrajectory = '<header> ' + trajectory.header + ' </header>\n'
            for j, frame in enumerate(trajectory.inputFrames):
                formattedTrajectory += f' <frame_{j}> ' + frame + f' </frame_{j}>\n'

            formattedTrajectory += ' <query> ' + trajectory.query + ' </query>'
            dataPoint['X'] = formattedTrajectory
            dataPoint['y'] = trajectory.prediction
            formatedInput.append(dataPoint)

        return formatedInput

    @staticmethod
    def generateTrajectorySequencesFromMRFDataset(workers, trajectoryWindowSize, sampleSizePerWorker, outputFilename):
        random.seed(1372)
        trajectorySequences = []
        for worker in workers:
            workerHeader = 'Worker ID: ' + worker['id'] + '\n' + \
                           'Age: ' + str(worker['age']) + '\n' + \
                           'Gender: ' + str(worker['gender']) + '\n' + \
                           'Education: ' + str(worker['education']) + '\n' + \
                           "Race: " + ", ".join(worker['race'] if len(worker['race']) else ['unknown']) + '\n' + \
                           "Media Diet: " + ", ".join(worker['mediaConsumptionRegimen'] if len(worker['mediaConsumptionRegimen']) else ['unknown']) + '\n'

            maxPossibleTrajectories = Trajectory.choiceOf(trajectoryWindowSize['max'], len(worker['annotatedFrames']))
            maxPossibleTrajectories = int(maxPossibleTrajectories)
            workerSampleSize = min(sampleSizePerWorker, maxPossibleTrajectories) # todo: this is a hacky way to limit the number of trajectories per worker
            for i in range(workerSampleSize):
                K = random.randint(trajectoryWindowSize['min'], trajectoryWindowSize['max'])
                sampledIndices = random.sample(range(len(worker['annotatedFrames'])), K + 1)
                sampledIndices = sorted(sampledIndices)
                sampledFrames = [worker['annotatedFrames'][i] for i in sampledIndices[:-1]]

                sampledFrames = [
                    'Headline: ' + frame['Input.sentence'] + '\n' +
                    'Reader\'s Reaction: ' + ", ".join(frame["reaction"]) + '\n' +
                    'Writer\'s Intent: ' + ", ".join(frame["intent"]) + '\n' +
                    'Perceived Label: ' + frame["perceivedLabel"] + '\n'

                    for frame in sampledFrames
                ]

                nextFrame = worker['annotatedFrames'][sampledIndices[-1]]

                query = 'Headline: ' + nextFrame['Input.sentence'] + '\n' + \
                        'Perceived Label: ?\n' + 'Reader\'s Reactions: ?\n' + 'Writer\'s Intent: ?\n'

                prediction = 'Perceived Label: ' + nextFrame["perceivedLabel"] + '\n' + \
                             'Reader\'s Reactions: '+ ", ".join(nextFrame["reaction"]) + '\n' + \
                             'Writer\'s Intent: ' + ", ".join(nextFrame["intent"]) + '\n'


                trajectory = Trajectory(sampledFrames, workerHeader, query, prediction)
                trajectorySequences.append(trajectory)

        trajectorySequences = Trajectory.formatInput(trajectorySequences)
        saveObjectsToJsonFile(trajectorySequences, outputFilename)


if __name__ == '__main__':
    outputPath = '../data/trajectories/bigrui/'
    workers = loadObjectsFromJsonFile('../data/mrf_turk_processed.json')
    # workers = sorted(workers, key=lambda x: len(x['annotatedFrames']), reverse=True)
    workers = [worker for worker in workers if
               len(worker['annotatedFrames']) >= 20]  # throwing out workers with less than 10 annotated frames
    random.seed(1372)
    random.shuffle(workers)
    samplingRate = 100000
    trainTestCutOffIndex = int(len(workers) * 0.9)
    trainWorkers, testWorkers = workers[:trainTestCutOffIndex], workers[trainTestCutOffIndex:]
    Trajectory.generateTrajectorySequencesFromMRFDataset(trainWorkers, {'min': 4, 'max': 8}, samplingRate, outputPath + 'train_trajectories.json')
    Trajectory.generateTrajectorySequencesFromMRFDataset(testWorkers, {'min': 4, 'max': 8}, samplingRate, outputPath + 'test_trajectories.json')
