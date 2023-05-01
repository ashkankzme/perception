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


if __name__ == '__main__':
    # Read in the data
    # data = readTSV('../data/mrf_v1/train_annot_all.tsv')

    # Print out the data
    # print(data[:5])
    mTurkData = readCSVFiles('../data/mturk_data/')
    print("Number of HITs completed: ", len(mTurkData))
    groupedMTurkData = groupByWorkerId(mTurkData)



    # select a random sample of 50 keys from groupedMTurkData, and print the length of the value list for each key
    # print("Random sample of 50 workers:")
    # for key in random.sample(list(groupedMTurkData.keys()), 50):
    #     example = random.sample(groupedMTurkData[key], 1)[0]
    #     # print(example['Input.sentence'], len(groupedMTurkData[key]))
    #     print(example['Input.sentence'])
    #
    # print("---------------")

    # sort the keys by the length of the value list, and print the first 50 keys descending
    # print("Top 50 workers by number of HITs completed:")
    # for key in sorted(groupedMTurkData, key=lambda k: len(groupedMTurkData[k]), reverse=True)[:50]:
    #     print(len(groupedMTurkData[key]))

    # print all rows in mTurkData where the value for 'Input.reason' is longer than 30 characters and 'LifetimeApprovalRate' is over 90%
    # all_column_names = list(set([key for row in mTurkData for key in row.keys()]))
    # for idx, row in enumerate(mTurkData):
    #     for column in all_column_names:
    #         if column not in row:
    #             mTurkData[idx][column] = ''

    # print(len([row for row in mTurkData if len(row['Input.reason']) > 20]))
    # print(len(set([row['Input.reason'].lower() for row in mTurkData if len(row['Input.reason']) > 20])))
    # for row in mTurkData:
    #     if len(row['Input.reason']) > 10 and len(row['LifetimeApprovalRate']) > 0 and float(row['LifetimeApprovalRate'].split('%')[0]) >= 90:
    #         print('WorkerId: {}'.format(row['WorkerId']))
    #         print('sentence: {}'.format(row['Input.sentence']))
    #         print('reason: {}'.format(row['Input.reason']))
    #         print('-----------------')
    # print(sorted(all_column_names))