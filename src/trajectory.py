import random
from src.utils import saveObjectsToJsonFile

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


    def toInputFormat(self):
        return f'{self.header}\n{self.inputFrames}\n{self.query}\n'


    def toOutputFormat(self):
        return f'{self.prediction}'

    @staticmethod
    def generateTrajectorySequencesFromMRFDataset(workers, trajectoryWindowSize, sampleSizePerWorker, outputFilename):
        random.seed(1372)
        trajectorySequences = []
        for worker in workers:
            workerHeader = 'Worker ID: ' + worker['id'] + '\n' + \
                           'Age: ' + str(worker['age']) + '\n' + \
                           'Gender: ' + str(worker['gender']) + '\n' + \
                           'Education: ' + str(worker['education']) + '\n' + \
                           "Race: " + ", ".join(worker['race']) + '\n' + \
                           "Media Diet: " + ", ".join(worker['mediaConsumptionRegimen']) + '\n'

            for i in range(sampleSizePerWorker):
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
                        'Reader\'s Reactions: ?\n' + 'Writer\'s Intent: ?\n' + 'Perceived Label: ?\n'

                prediction = 'Reader\'s Reactions: '+ ", ".join(nextFrame["reaction"]) + '\n' + \
                             'Writer\'s Intent: ' + ", ".join(nextFrame["reaction"]) + '\n' + \
                             'Perceived Label: ' + nextFrame["perceivedLabel"] + '\n'

                trajectory = Trajectory(sampledFrames, workerHeader, query, prediction)
                trajectorySequences.append(trajectory)

        saveObjectsToJsonFile(trajectorySequences, outputFilename)