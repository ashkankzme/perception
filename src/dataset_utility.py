import csv
import os
import random

from humanWorker import HumanWorker
from trajectory import Trajectory
from utils import saveObjectsToJsonFile, loadObjectsFromJsonFile


class MRFDatasetUtility(object):
    @staticmethod
    # Read TSV file line by line, return as list of objects with header as keys
    def readTSV(path):
        with open(path, 'r') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t')
            return [row for row in reader]

    @staticmethod
    # Read all .csv files in input directory, pushing rows as objects into a list with CSV header as keys. Return list of objects. CSV files are encoded as ISO-8859-1.
    def readCSVFiles(directory):
        data = []
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                with open(os.path.join(directory, filename), 'r', encoding='ISO-8859-1') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        data.append(row)
        return data

    # Input: list of objects
    # Output: Return a dict, containing all objects in the list, grouped by "WorkerId"
    def groupByWorkerId(data):
        grouped = {}
        for row in data:
            if row['WorkerId'] not in grouped:
                grouped[row['WorkerId']] = []
            grouped[row['WorkerId']].append(row)
        return grouped


    # Input: a dict of HITS mapped to their corresponding WorkerIds
    # Output: a new dict of WorkerIds mapped to their corresponding HITS, while only preserving columns with predetermined special names
    @staticmethod
    def filterColumns(groupedMTurkData):
        filteredGroupedMTurkData = {}
        specialColumnNames = [
            'WorkerId',
            'Answer.10mo2a', # reader intent
            'Answer.10mo2ab',
            'Answer.10mo2b',
            'Answer.10mo2c',
            'Answer.10react.no',
            'Answer.10react.yes',
            'Answer.2imply.no',  # q4 (first under writer intent)
            'Answer.2imply.yes',
            'Answer.mo2a', # q4 (first under writer intent)
            'Answer.mo2b',
            'Answer.mo2c',
            'Answer.3imply.no',
            'Answer.3imply.yes',
            'Answer.3mo2a',
            'Answer.3mo2b',
            'Answer.3mo2c',
            'Answer.4imply.no',
            'Answer.4imply.yes',
            'Answer.4mo2a',
            'Answer.4mo2b',
            'Answer.4mo2c',
            'Answer.5imply.no',
            'Answer.5imply.yes',
            'Answer.5mo2a',
            'Answer.5mo2b',
            'Answer.5mo2c',
            'Answer.6imply.no',
            'Answer.6imply.yes',
            'Answer.6mo2a',
            'Answer.6mo2b',
            'Answer.6mo2c',
            'Answer.7imply.no',
            'Answer.7imply.yes',
            'Answer.7mo2a',
            'Answer.7mo2b',
            'Answer.7mo2c',
            'Answer.8imply.no',
            'Answer.8imply.yes',
            'Answer.8mo2a',
            'Answer.8mo2b',
            'Answer.8mo2c',
            'Answer.age',
            'Answer.ed',
            'Answer.eth1.White',
            'Answer.eth2.Hispanic/Latino',
            'Answer.eth3.Black/African-American',
            'Answer.eth4.Native American',
            'Answer.eth5.Asian/Pacific Islander',
            'Answer.eth6.Other',
            'Answer.gender',
            'Answer.likert.agree', # this is likelyhood to share
            'Answer.likert.disagree',
            'Answer.likert.neutral',
            'Answer.likert.strong_agree',
            'Answer.likert.strong_disagree',
            'Answer.news1.Twitter',
            'Answer.news10.Daily Mail',
            'Answer.news11.Washington Post',
            'Answer.news12.Reuters',
            'Answer.news13.Breitbart News Network',
            'Answer.news14.NPR',
            'Answer.news15.BBC',
            'Answer.news16.GUARDIAN',
            'Answer.news17.Facebook',
            'Answer.news17.Other',
            'Answer.news18.Reddit',
            'Answer.news19.4chan',
            'Answer.news2.Yahoo-ABC News Network',
            'Answer.news20.Vox',
            'Answer.news21.youtube',
            'Answer.news3.CNN',
            'Answer.news4.Huffington Post',
            'Answer.news5.CBS News',
            'Answer.news6.USAToday',
            'Answer.news7.BuzzFeed',
            'Answer.news8.New York Times',
            'Answer.news9.Fox News Digital Network',
            'Answer.otherbox',
            'Answer.real.mis',
            'Answer.real.realnews',
            'HITId',
            'LifetimeApprovalRate',
            'SubmitTime',
            'Input.label',
            'Input.sentence'
        ]
        for workerId in groupedMTurkData:
            filteredGroupedMTurkData[workerId] = []
            for hit in groupedMTurkData[workerId]:
                # if hit['AssignmentStatus'] != 'Approved': # TODO double check what the criteria for an approved HIT is
                #     continue

                filteredHit = {}
                for key in hit:
                    if key in specialColumnNames and key not in ['Answer.mo2a', 'Answer.mo2b', 'Answer.mo2c']:
                        filteredHit[key] = hit[key]
                    elif key in specialColumnNames:
                        newKey = key.replace('Answer.', 'Answer.2')
                        filteredHit[newKey] = hit[key]

                for key in specialColumnNames:
                    if key not in hit:
                        if key in ['Answer.mo2a', 'Answer.mo2b', 'Answer.mo2c']:
                            newKey = key.replace('Answer.', 'Answer.2')
                            filteredHit[newKey] = ''
                        else:
                            filteredHit[key] = ''

                filteredGroupedMTurkData[workerId].append(filteredHit)

            filteredGroupedMTurkData[workerId] = sorted(filteredGroupedMTurkData[workerId], key=lambda k: k['SubmitTime'])

        return filteredGroupedMTurkData


    @staticmethod
    def getMostRepeatedAnswer(columnId, hits):
        answers = {}
        for hit in hits:
            if hit[columnId] not in answers:
                answers[hit[columnId]] = 0
            answers[hit[columnId]] += 1
        return max(answers, key=answers.get)


    @staticmethod
    def transformMTurkData(refinedGroupedMTurkData):
        # get text columns most repeated values
        intentTexts = [
            'Answer.10mo2a',
            'Answer.10mo2ab',
            'Answer.10mo2b',
            'Answer.10mo2c',
            ]

        reactionTexts = [
            'Answer.2mo2a',  # q4 (first under writer intent)
            'Answer.2mo2b',
            'Answer.2mo2c',
            'Answer.3mo2a',
            'Answer.3mo2b',
            'Answer.3mo2c',
            'Answer.4mo2a',
            'Answer.4mo2b',
            'Answer.4mo2c',
            'Answer.5mo2a',
            'Answer.5mo2b',
            'Answer.5mo2c',
            'Answer.6mo2a',
            'Answer.6mo2b',
            'Answer.6mo2c',
            'Answer.7mo2a',
            'Answer.7mo2b',
            'Answer.7mo2c',
            'Answer.8mo2a',
            'Answer.8mo2b',
            'Answer.8mo2c',
        ]

        mostRepeatedAnswers = {}
        refinedHits = [hit for workerId in refinedGroupedMTurkData for hit in refinedGroupedMTurkData[workerId]]
        for column in intentTexts + reactionTexts:
            columnMostRepeatedAnswer = MRFDatasetUtility.getMostRepeatedAnswer(column, refinedHits)
            mostRepeatedAnswers[column] = columnMostRepeatedAnswer

        # iterate over groupedData, for each row, remove all columns that equal the corresponding value in mostRepeatedAnswers
        workers = []
        for workerId in refinedGroupedMTurkData:
            for hit in refinedGroupedMTurkData[workerId]:
                hit['reaction'] = []
                if hit['Answer.10react.yes']:
                    if len(hit['Answer.10mo2a']) > 0 and hit['Answer.10mo2a'] != mostRepeatedAnswers['Answer.10mo2a']:
                        hit['reaction'].append(hit['Answer.10mo2a'])

                    if len(hit['Answer.10mo2ab']) > 0 and hit['Answer.10mo2ab'] != mostRepeatedAnswers['Answer.10mo2ab']:
                        hit['reaction'].append(hit['Answer.10mo2ab'])

                    if len(hit['Answer.10mo2b']) > 0 and hit['Answer.10mo2b'] != mostRepeatedAnswers['Answer.10mo2b']:
                        hit['reaction'].append(hit['Answer.10mo2b'])

                    if len(hit['Answer.10mo2c']) > 0 and hit['Answer.10mo2c'] != mostRepeatedAnswers['Answer.10mo2c']:
                        hit['reaction'].append(hit['Answer.10mo2c'])

                hit['intent'] = []
                for column in reactionTexts:
                    if column not in hit:
                        continue

                    if len(hit[column]) > 0 and hit[column] == mostRepeatedAnswers[column]:
                        del hit[column]
                        # additionalColumns = [_c for _c in hit.keys() if _c[:8] == column[:8] and _c != column] # removing imply columns
                        # for additionalColumn in additionalColumns:
                        #     del hit[additionalColumn]

                    elif hit[column[:8] + 'imply.yes'].lower() == 'true': # todo test if actually works
                        hit['intent'].append(hit[column])

                hit['perceivedLabel'] = 'real news' if hit['Answer.real.realnews'] == 'true' else 'misinformation' if hit['Answer.real.mis'] == 'true' else None
                hit['actualLabel'] = hit['Input.label']

            workerDemographics = {}
            workerDemographics['age'] = list(set([hit['Answer.age'] for hit in refinedGroupedMTurkData[workerId]]))
            workerDemographics['gender'] = list(set([hit['Answer.gender'] for hit in refinedGroupedMTurkData[workerId]]))
            workerDemographics['education'] = list(set([hit['Answer.ed'] for hit in refinedGroupedMTurkData[workerId]]))
            workerDemographics['race'] = {
                'white': list(set([hit['Answer.eth1.White'].lower() for hit in refinedGroupedMTurkData[workerId]])),
                'latino': list(set([hit['Answer.eth2.Hispanic/Latino'].lower() for hit in refinedGroupedMTurkData[workerId]])),
                'black': list(set([hit['Answer.eth3.Black/African-American'].lower() for hit in refinedGroupedMTurkData[workerId]])),
                'american': list(set([hit['Answer.eth4.Native American'].lower() for hit in refinedGroupedMTurkData[workerId]])),
                'asian': list(set([hit['Answer.eth5.Asian/Pacific Islander'].lower() for hit in refinedGroupedMTurkData[workerId]])),
                'other': list(set([hit['Answer.eth6.Other'].lower() for hit in refinedGroupedMTurkData[workerId]]))
            }

            workerMediaConsumptionRegimen = {
                # 'Answer.news1.Twitter',
                'twitter': list(set([hit['Answer.news1.Twitter'].lower() for hit in refinedGroupedMTurkData[workerId]])),
                # 'Answer.news10.Daily Mail',
                "dailyMail": list(set([hit['Answer.news10.Daily Mail'].lower() for hit in refinedGroupedMTurkData[workerId]])),
                # 'Answer.news11.Washington Post',
                "washingtonPost": list(set([hit['Answer.news11.Washington Post'].lower() for hit in refinedGroupedMTurkData[workerId]])),
                # 'Answer.news12.Reuters',
                "reuters": list(set([hit['Answer.news12.Reuters'].lower() for hit in refinedGroupedMTurkData[workerId]])),
                # 'Answer.news13.Breitbart News Network',
                "breitbart": list(set([hit['Answer.news13.Breitbart News Network'].lower() for hit in refinedGroupedMTurkData[workerId]])),
                # 'Answer.news14.NPR',
                "npr": list(set([hit['Answer.news14.NPR'].lower() for hit in refinedGroupedMTurkData[workerId]])),
                # 'Answer.news15.BBC',
                "bbc": list(set([hit['Answer.news15.BBC'].lower() for hit in refinedGroupedMTurkData[workerId]])),
                # 'Answer.news16.GUARDIAN',
                "guardian": list(set([hit['Answer.news16.GUARDIAN'].lower() for hit in refinedGroupedMTurkData[workerId]])),
                # 'Answer.news17.Facebook',
                "facebook": list(set([hit['Answer.news17.Facebook'].lower() for hit in refinedGroupedMTurkData[workerId]])),
                # 'Answer.news17.Other',
                "other": list(set([hit['Answer.news17.Other'].lower() for hit in refinedGroupedMTurkData[workerId]])),
                # 'Answer.news18.Reddit',
                "reddit": list(set([hit['Answer.news18.Reddit'].lower() for hit in refinedGroupedMTurkData[workerId]])),
                # 'Answer.news19.4chan',
                "4chan": list(set([hit['Answer.news19.4chan'].lower() for hit in refinedGroupedMTurkData[workerId]])),
                # 'Answer.news2.Yahoo-ABC News Network',
                "yahoo": list(set([hit['Answer.news2.Yahoo-ABC News Network'].lower() for hit in refinedGroupedMTurkData[workerId]])),
                # 'Answer.news20.Vox',
                "vox": list(set([hit['Answer.news20.Vox'].lower() for hit in refinedGroupedMTurkData[workerId]])),
                # 'Answer.news21.youtube',
                "youtube": list(set([hit['Answer.news21.youtube'].lower() for hit in refinedGroupedMTurkData[workerId]])),
                # 'Answer.news3.CNN',
                "cnn": list(set([hit['Answer.news3.CNN'].lower() for hit in refinedGroupedMTurkData[workerId]])),
                # 'Answer.news4.Huffington Post',
                "huffingtonPost": list(set([hit['Answer.news4.Huffington Post'].lower() for hit in refinedGroupedMTurkData[workerId]])),
                # 'Answer.news5.CBS News',
                "cbs": list(set([hit['Answer.news5.CBS News'].lower() for hit in refinedGroupedMTurkData[workerId]])),
                # 'Answer.news6.USAToday',
                "usaToday": list(set([hit['Answer.news6.USAToday'].lower() for hit in refinedGroupedMTurkData[workerId]])),
                # 'Answer.news7.BuzzFeed',
                "buzzFeed": list(set([hit['Answer.news7.BuzzFeed'].lower() for hit in refinedGroupedMTurkData[workerId]])),
                # 'Answer.news8.New York Times',
                "newYorkTimes": list(set([hit['Answer.news8.New York Times'].lower() for hit in refinedGroupedMTurkData[workerId]])),
                # 'Answer.news9.Fox News Digital Network',
                "fox": list(set([hit['Answer.news9.Fox News Digital Network'].lower() for hit in refinedGroupedMTurkData[workerId]])),
            }

            if len(refinedGroupedMTurkData[workerId]) > 0:
                workers.append(
                    HumanWorker(workerId, workerDemographics, workerMediaConsumptionRegimen, refinedGroupedMTurkData[workerId][-1]['LifetimeApprovalRate'])
                )
                workers[-1].setFrames([hit for hit in refinedGroupedMTurkData[workerId]
                                       if 'Input.sentence' in hit and len(hit['Input.sentence'])
                                       and (len(hit['intent']) or hit['reaction'])
                                       and 'perceivedLabel' in hit and hit['perceivedLabel']]) # todo fix formatting, remove unwanted rows

        return workers


    @staticmethod
    def loadAndCleanMRFDataset(inputPath, outputFilename):
        mTurkData = MRFDatasetUtility.readCSVFiles(inputPath)
        print("Number of HITs completed: ", len(mTurkData))
        groupedMTurkData = MRFDatasetUtility.groupByWorkerId(mTurkData)
        refinedGroupedMTurkData = MRFDatasetUtility.filterColumns(groupedMTurkData)
        workers = MRFDatasetUtility.transformMTurkData(refinedGroupedMTurkData)
        # todo remove hits with no intent or reaction (low quality headlines)
        # persisting processed data
        saveObjectsToJsonFile(workers, outputFilename) # todo test serialization/deserialization
        return workers


    @staticmethod
    def generateTrajectoriesFromMRFDataset(outputPath, labelsOnly=False):
        # workers = mrfdu.loadAndCleanMRFDataset('../data/mturk_data/', '../data/mrf_turk_processed.json')
        workers = loadObjectsFromJsonFile('../data/mrf_turk_processed.json')
        workers = [worker for worker in workers if len(worker['annotatedFrames']) >= 100]  # throwing out workers with less than 10 annotated frames

        topWorkerIDs = [worker['id'] for worker in sorted(workers, key=lambda x: len(x['annotatedFrames']), reverse=True)[:10]]
        shuffleWorkers = [worker for worker in workers if worker['id'] not in topWorkerIDs]

        random.seed(1373)
        random.shuffle(shuffleWorkers)

        trainWorkers = [worker for worker in workers if worker['id'] in topWorkerIDs] + workers[:-11]
        evalWorkers, testWorkers = workers[-11:-10], workers[-10:]
        Trajectory.generateTrajectorySequencesFromMRFDataset(trainWorkers, {'min': 4, 'max': 8}, outputPath+'train_trajectories.json', labelsOnly=labelsOnly)
        Trajectory.generateTrajectorySequencesFromMRFDataset(evalWorkers, {'min': 4, 'max': 8}, outputPath+'eval_trajectories.json', labelsOnly=labelsOnly)
        Trajectory.generateTrajectorySequencesFromMRFDataset(testWorkers, {'min': 4, 'max': 8}, outputPath+'test_trajectories.json', labelsOnly=labelsOnly)

    @staticmethod
    def generateLeaveOneOutTrajectories(outputPath, labelsOnly=False, testWorkerIds=None):
        # workers = mrfdu.loadAndCleanMRFDataset('../data/mturk_data/', '../data/mrf_turk_processed.json')
        if testWorkerIds is None:
            testWorkerIds = []
        workers = loadObjectsFromJsonFile('../data/mrf_turk_processed.json')
        workers = [worker for worker in workers if len(worker['annotatedFrames']) >= 100]  # throwing out workers with less than 10 annotated frames

        # trainWorkers = [worker for worker in workers if worker['id'] not in testWorkerIds]
        testWorkers = [worker for worker in workers if worker['id'] in testWorkerIds]
        Trajectory.generateTrajectorySequencesFromMRFDataset(workers, {'min': 4, 'max': 8}, outputPath + 'train_trajectories.json', labelsOnly=labelsOnly)
        Trajectory.generateTrajectorySequencesFromMRFDataset(testWorkers, {'min': 4, 'max': 8}, outputPath + 'test_trajectories.json', labelsOnly=labelsOnly)


    @staticmethod
    def getActualLabelForHeadlinesFromMRFPublicDataset(headlines):
        headlineLabels = {}
        frames = MRFDatasetUtility.readCSVFiles('../data/mrf_v1/')
        for frame in frames:
            headlineLabels[frame['headline'].strip().lower()] = frame['gold_label']

        return [headlineLabels[headline.strip().lower()] for headline in headlines]