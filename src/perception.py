import os
import random
import csv


# Read TSV file line by line, return as list of objects with header as keys
def readTSV(path):
    with open(path, 'r') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        return [row for row in reader]


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
# Output: a new dict of WorkerIds mapped to their corresponding HITS, while only preserving the columns that are most important
# List of columns that are important:
# Answer.10mo2a
# Answer.10mo2b
# Answer.10mo2ab
# Answer.10mo2c
# Answer.2imply.yes
# Answer.mo2a
# Answer.mo2b
# Answer.mo2c
# Answer.3imply.yes
# Answer.3mo2a
# Answer.3mo2b
# Answer.3mo2c
# Answer.4imply.yes
# Answer.4mo2a
# Answer.4mo2b
# Answer.4mo2c
# Answer.5imply.yes
# Answer.5mo2a
# Answer.5mo2b
# Answer.5mo2c
# Answer.6imply.yes
# Answer.6mo2a
# Answer.6mo2b
# Answer.6mo2c
# Answer.7imply.yes
# Answer.7mo2a
# Answer.7mo2b
# Answer.7mo2c
# Answer.8imply.yes
# Answer.8mo2a
# Answer.8mo2b
# Answer.8mo2c
# Answer.likert.agree
# Answer.likert.disagree
# Answer.likert.neutral
# Answer.likert.strong_agree
# Answer.likert.strong_disagree
# Answer.real.mis
# Answer.real.realnews
def filterColumns(groupedMTurkData):
    filtered = {}
    for workerId in groupedMTurkData:
        filtered[workerId] = []
        for row in groupedMTurkData[workerId]:
            filtered[workerId].append({
                'Answer.10mo2a': row['Answer.10mo2a'],
                'Answer.10mo2b': row['Answer.10mo2b'],
                'Answer.10mo2ab': row['Answer.10mo2ab'],
                'Answer.10mo2c': row['Answer.10mo2c'],
                'Answer.2imply.yes': row['Answer.2imply.yes'],
                'Answer.mo2a': row['Answer.mo2a'],
                'Answer.mo2b': row['Answer.mo2b'],
                'Answer.mo2c': row['Answer.mo2c'],
                'Answer.3imply.yes': row['Answer.3imply.yes'],
                'Answer.3mo2a': row['Answer.3mo2a'],
                'Answer.3mo2b': row['Answer.3mo2b'],
                'Answer.3mo2c': row['Answer.3mo2c'],
                'Answer.4imply.yes': row['Answer.4imply.yes'],
                'Answer.4mo2a': row['Answer.4mo2a'],
                'Answer.4mo2b': row['Answer.4mo2b'],
                'Answer.4mo2c': row['Answer.4mo2c'],
                'Answer.5imply.yes': row['Answer.5imply.yes'],
                'Answer.5mo2a': row['Answer.5mo2a'],
                'Answer.5mo2b': row['Answer.5mo2b'],
                'Answer.5mo2c': row['Answer.5mo2c'],
                'Answer.6imply.yes': row['Answer.6imply.yes'],
                'Answer.6mo2a': row['Answer.6mo2a'],
                'Answer.6mo2b': row['Answer.6mo2b'],
                'Answer.6mo2c': row['Answer.6mo2c'],
                'Answer.7imply.yes': row['Answer.7imply.yes'],
                'Answer.7mo2a': row['Answer.7mo2a'],
                'Answer.7mo2b': row['Answer.7mo2b'],
                'Answer.7mo2c': row['Answer.7mo2c'],
                'Answer.8imply.yes': row['Answer.8imply.yes'],
                'Answer.8mo2a': row['Answer.8mo2a'],
                'Answer.8mo2b': row['Answer.8mo2b'],
                'Answer.8mo2c': row['Answer.8mo2c'],
                'Answer.likert.agree': row['Answer.likert.agree'],
                'Answer.likert.disagree': row['Answer.likert.disagree'],
                'Answer.likert.neutral': row['Answer.likert.neutral'],
                'Answer.likert.strong_agree': row['Answer.likert.strong_agree'],
                'Answer.likert.strong_disagree': row['Answer.likert.strong_disagree'],
                'Answer.real.mis': row['Answer.real.mis'],
                'Answer.real.realnews': row['Answer.real.realnews']
            })
    return filtered


if __name__ == '__main__':
    mTurkData = readCSVFiles('../data/mturk_data/')
    print("Number of HITs completed: ", len(mTurkData))
    groupedMTurkData = groupByWorkerId(mTurkData)
    refinedGroupedMTurkData = filterColumns(groupedMTurkData)
    print("A random sample of 10 HITS completed by a random worker: ", random.sample(refinedGroupedMTurkData[random.choice(list(refinedGroupedMTurkData.keys()))], 10))

    # print all rows in mTurkData where the value for 'Input.reason' is longer than 30 characters and 'LifetimeApprovalRate' is over 90%
    all_column_names = list(set([key for row in mTurkData for key in row.keys()]))
    print(sorted(all_column_names))
