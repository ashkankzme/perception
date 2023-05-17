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
        for workerID in workers:
            workerHeader = 'Worker ID: ' + workerID + '\n' + \
                           'Age: ' + str(workers[workerID]['demographics']['age']) + '\n' + \
                           'Gender: ' + str(workers[workerID]['demographics']['gender']) + '\n' + \
                           'Education: ' + str(workers[workerID]['demographics']['education']) + '\n' + \
                           'Race: ' + str(workers[workerID]['demographics']['race']) + '\n' + \
                           'Media Diet: ' + str(workers[workerID]['mediaConsumptionRegimen']) + '\n'

            for i in range(sampleSizePerWorker):
                K = random.randint(trajectoryWindowSize['min'], trajectoryWindowSize['max'])
                sampledIndices = random.sample(range(len(workers[workerID]['frames'])), K + 1)
                sampledIndices = sorted(sampledIndices)
                sampledFrames = [workers[workerID]['frames'][i] for i in sampledIndices[:-1]]

                sampledFrames = [
                    'Headline: ' + frame['headline'] + '\n' +
                    f'Reader\'s Reaction: {", ".join(frame["reaction"])}\n' +
                    f'Writer\'s Intent: {", ".join(frame["intent"])}\n' +
                    f'Perceived Label: {frame["perceivedLabel"]}\n'

                    for frame in sampledFrames
                ]

                nextFrame = workers[workerID]['frames'][sampledIndices[-1]]

                query = 'Headline: ' + nextFrame['headline'] + '\n' + \
                        'Reader\'s Reactions: ?\n' + 'Writer\'s Intent: ?\n' + 'Perceived Label: ?\n'

                prediction = f'Reader\'s Reactions: {", ".join(nextFrame["reaction"])}\n' + \
                             f'Writer\'s Intent: {", ".join(nextFrame["reaction"])}\n' + \
                             f'Perceived Label: {nextFrame["perceivedLabel"]}\n'

                trajectory = Trajectory(sampledFrames, workerHeader, query, prediction)
                trajectorySequences.append(trajectory)

        saveObjectsToJsonFile(trajectorySequences, outputFilename)