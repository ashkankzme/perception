from dataset_utility import MRFDatasetUtility as mrfdu

if __name__ == '__main__':
    headlines = ['ct heart scan radiation cancer risk ?.', 'breast cancer more veggies not better.']
    print(mrfdu.getActualLabelForHeadlinesFromMRFPublicDataset(headlines))