import random
import math
from utils import saveToJsonFile, loadObjectsFromJsonFile

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
    def getAgeGroup(ageCode):
        if ageCode == "1":
            return "18-24"
        elif ageCode == "2":
            return "25-34"
        elif ageCode == "3":
            return "35-44"
        elif ageCode == "4":
            return "45-54"
        elif ageCode == "5":
            return "55-64"
        elif ageCode == "6":
            return "65+"
        elif ageCode.lower() == "unknown":
            return "Unknown"

    @staticmethod
    def getEducationLevel(educationCode):
        if educationCode == "1":
            return "Some High School"
        elif educationCode == "2":
            return "High School"
        elif educationCode == "3":
            return "Some College"
        elif educationCode == "4":
            return "Associate's Degree"
        elif educationCode == "5":
            return "Bachelor's Degree"
        elif educationCode == "6":
            return "Master's Degree"
        elif educationCode == "7":
            return "Professional Degree"
        elif educationCode == "8":
            return "Doctoral Degree"
        elif educationCode.lower() == "unknown":
            return "Unknown"

    @staticmethod
    def getGender(genderCode):
        if genderCode == "1":
            return "Female"
        elif genderCode == "2":
            return "Male"
        elif genderCode == "3":
            return "Non-Binary"
        elif genderCode == "4":
            return "Other"
        elif genderCode.lower() == "unknown":
            return "Unknown"

    @staticmethod
    def formatInput(inputs):
        formatedInput = []
        for i, trajectory in enumerate(inputs):
            dataPoint = {'X': '', 'y': ''}
            formattedTrajectory = ''
            formattedTrajectory += '<query>' + trajectory.query + '</query>'
            formattedTrajectory += '<header>' + trajectory.header + '</header>\n'
            for j, frame in enumerate(trajectory.inputFrames):
                formattedTrajectory += f'<frame_{j}>' + frame + f'</frame_{j}>\n'

            dataPoint['X'] = formattedTrajectory
            dataPoint['y'] = trajectory.prediction
            formatedInput.append(dataPoint)

        return formatedInput

    @staticmethod
    #@deprecated
    def _generateTrajectorySequencesFromMRFDataset(workers, trajectoryWindowSize, sampleSizePerWorker, outputFilename):
        random.seed(1372)
        trajectorySequences = []
        for worker in workers:
            workerHeader = 'Worker ID: ' + worker['id'] + '\n' + \
                           'Age: ' + str(worker['age']) + '\n' + \
                           'Gender: ' + str(worker['gender']) + '\n' + \
                           'Education: ' + str(worker['education']) + '\n' + \
                           "Race: " + ", ".join(worker['race'] if len(worker['race']) else ['unknown']) + '\n' + \
                           "Media Diet: " + ", ".join(
                worker['mediaConsumptionRegimen'] if len(worker['mediaConsumptionRegimen']) else ['unknown']) + '\n'

            maxPossibleTrajectories = Trajectory.choiceOf(trajectoryWindowSize['max'], len(worker['annotatedFrames']))
            maxPossibleTrajectories = int(maxPossibleTrajectories)
            workerSampleSize = min(sampleSizePerWorker,
                                   maxPossibleTrajectories)  # todo: this is a hacky way to limit the number of trajectories per worker
            for i in range(workerSampleSize):
                K = random.randint(trajectoryWindowSize['min'], trajectoryWindowSize['max'])
                sampledIndices = random.sample(range(len(worker['annotatedFrames'])), K + 1)
                sampledIndices = sorted(sampledIndices)
                sampledFrames = [worker['annotatedFrames'][i] for i in sampledIndices[:-1]]

                sampledFrames = [
                    'Headline: ' + frame['Input.sentence'] + '\n' +
                    # 'Reader\'s Reaction: ' + ", ".join(frame["reaction"]) + '\n' +
                    # 'Writer\'s Intent: ' + ", ".join(frame["intent"]) + '\n' +
                    'Perceived Label: ' + frame["perceivedLabel"] + '\n'

                    for frame in sampledFrames
                ]

                nextFrame = worker['annotatedFrames'][sampledIndices[-1]]

                query = 'Headline: ' + nextFrame['Input.sentence'] + '\n' + \
                        'Perceived Label: ?\n' + 'Reader\'s Reactions: ?\n' + 'Writer\'s Intent: ?\n'

                prediction = 'Perceived Label: ' + nextFrame["perceivedLabel"] + '\n'  # + \
                # 'Reader\'s Reactions: '+ ", ".join(nextFrame["reaction"]) + '\n' + \
                # 'Writer\'s Intent: ' + ", ".join(nextFrame["intent"]) + '\n'

                trajectory = Trajectory(sampledFrames, workerHeader, query, prediction)
                trajectorySequences.append(trajectory)

        trajectorySequences = Trajectory.formatInput(trajectorySequences)
        saveToJsonFile(trajectorySequences, outputFilename)

    @staticmethod
    def generateTrajectorySequencesFromMRFDataset(workers, trajectoryWindowSize, outputFilename, labelsOnly=False):
        random.seed(1372)
        trajectorySequences = []
        for worker in workers:
            workerHeader = 'Worker ID: ' + worker['id'] + '\n' + \
                           'Age: ' + Trajectory.getAgeGroup(str(worker['age'])) + '\n' + \
                           'Gender: ' + Trajectory.getGender(str(worker['gender'])) + '\n' + \
                           'Education: ' + Trajectory.getEducationLevel(str(worker['education'])) + '\n' + \
                           "Race: " + ", ".join(worker['race'] if len(worker['race']) else ['unknown']) + '\n' + \
                           "Media Diet: " + ", ".join(
                worker['mediaConsumptionRegimen'] if len(worker['mediaConsumptionRegimen']) else ['unknown']) + '\n'

            pivotIndex = 0
            while pivotIndex < len(worker['annotatedFrames']) - trajectoryWindowSize['max']:
                K = random.randint(trajectoryWindowSize['min'], trajectoryWindowSize['max'])

                sampledFrames = worker['annotatedFrames'][pivotIndex: pivotIndex + K]

                sampledFrames = [
                    'Headline: ' + frame['Input.sentence'] + '\n' +
                    ('' if labelsOnly else 'Reader\'s Reaction: ' + ", ".join(frame["reaction"]) + '\n') +
                    ('' if labelsOnly else 'Writer\'s Intent: ' + ", ".join(frame["intent"]) + '\n') +
                    'Perceived Label: ' + frame["perceivedLabel"] + '\n'
                    for frame in sampledFrames
                ]

                nextFrame = worker['annotatedFrames'][pivotIndex + K]

                query = 'Headline: ' + nextFrame['Input.sentence'] + '\n' + \
                        'Perceived Label: ?\n' + \
                        ('' if labelsOnly else 'Reader\'s Reactions: ?\n') + \
                        ('' if labelsOnly else 'Writer\'s Intent: ?\n')

                prediction = 'Perceived Label: ' + nextFrame["perceivedLabel"] + '\n' + \
                             ('' if labelsOnly else 'Reader\'s Reactions: '+ ", ".join(nextFrame["reaction"]) + '\n') + \
                             ('' if labelsOnly else 'Writer\'s Intent: ' + ", ".join(nextFrame["intent"]) + '\n')

                trajectory = Trajectory(sampledFrames, workerHeader, query, prediction)
                trajectorySequences.append(trajectory)

                # advancing pivotIndex by K + 1
                # +1 is to prevent leakage in training data,
                # we need to skip the next frame which contains the label for a training sample)
                pivotIndex += K + 1

        trajectorySequences = Trajectory.formatInput(trajectorySequences)
        saveToJsonFile(trajectorySequences, outputFilename)

    @staticmethod
    def generateQueriesFromMRFDataset(workers, outputFilename, labelsOnly=False):
        random.seed(1372)
        trajectorySequences = []
        for worker in workers:
            workerHeader = 'Worker ID: ' + worker['id'] + '\n' + \
                           'Age: ' + Trajectory.getAgeGroup(str(worker['age'])) + '\n' + \
                           'Gender: ' + Trajectory.getGender(str(worker['gender'])) + '\n' + \
                           'Education: ' + Trajectory.getEducationLevel(str(worker['education'])) + '\n' + \
                           "Race: " + ", ".join(worker['race'] if len(worker['race']) else ['unknown']) + '\n' + \
                           "Media Diet: " + ", ".join(
                worker['mediaConsumptionRegimen'] if len(worker['mediaConsumptionRegimen']) else ['unknown']) + '\n'

            for frame in worker['annotatedFrames']:
                query = 'Headline: ' + frame['Input.sentence'] + '\n' + \
                        'Perceived Label: ?\n' + \
                        ('' if labelsOnly else 'Reader\'s Reactions: ?\n') + \
                        ('' if labelsOnly else 'Writer\'s Intent: ?\n')

                prediction = 'Perceived Label: ' + frame["perceivedLabel"] + '\n' + \
                             ('' if labelsOnly else 'Reader\'s Reactions: ' + ", ".join(frame["reaction"]) + '\n') + \
                             ('' if labelsOnly else 'Writer\'s Intent: ' + ", ".join(frame["intent"]) + '\n')

                trajectory = Trajectory([], workerHeader, query, prediction)
                trajectorySequences.append(trajectory)

        trajectorySequences = Trajectory.formatInput(trajectorySequences)
        saveToJsonFile(trajectorySequences, outputFilename)

# if __name__ == '__main__':
#     outputPath = '../data/trajectories/bigrui/'
#     workers = loadObjectsFromJsonFile('../data/mrf_turk_processed.json')
#     # workers = sorted(workers, key=lambda x: len(x['annotatedFrames']), reverse=True)
#     workers = [worker for worker in workers if
#                len(worker['annotatedFrames']) >= 20]  # throwing out workers with less than 10 annotated frames
#     random.seed(1372)
#     random.shuffle(workers)
#     trainTestCutOffIndex = int(len(workers) * 0.9)
#     trainWorkers, testWorkers = workers[:trainTestCutOffIndex], workers[trainTestCutOffIndex:]
#     Trajectory.generateTrajectorySequencesFromMRFDataset(trainWorkers, {'min': 4, 'max': 8}, outputPath + 'train_trajectories.json')
#     Trajectory.generateTrajectorySequencesFromMRFDataset(testWorkers, {'min': 4, 'max': 8}, outputPath + 'test_trajectories.json')
