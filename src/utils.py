import csv
import os


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
            'Answer.10mo2a',
            'Answer.10mo2ab',
            'Answer.10mo2b',
            'Answer.10mo2c',
            'Answer.10react.no',
            'Answer.10react.yes',
            'Answer.2imply.no',
            'Answer.2imply.yes',
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
            'Answer.mo2a',
            'Answer.mo2b',
            'Answer.mo2c',
            'Answer.likert.agree',
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
            'LifetimeApprovalRate'
        ]
        for workerId in groupedMTurkData:
            filteredGroupedMTurkData[workerId] = []
            for hit in groupedMTurkData[workerId]:
                filteredHit = {}
                for key in hit:
                    if key in specialColumnNames:
                        filteredHit[key] = hit[key]
                filteredGroupedMTurkData[workerId].append(filteredHit)
        return filteredGroupedMTurkData
