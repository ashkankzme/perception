from utils import loadObjectsFromJsonFile, saveToJsonFile


def generateSentimentAnalysisDataForUsersWDemographics(outputPath):
    workerIdsWDemographics = loadObjectsFromJsonFile('../data/worker_ids_with_demographics.json')
    workers = loadObjectsFromJsonFile('../data/mrf_turk_processed.json')
    workers = [worker for worker in workers if len(worker['annotatedFrames']) >= 100]  # throwing out workers with less than 10 annotated frames
    demographicWorkers = [worker for worker in workers if worker['id'] in workerIdsWDemographics]

    headlinesMappedToReactions = {}
    for worker in demographicWorkers:
        for frame in worker['annotatedFrames']:
            if len(frame['reaction']) == 0:
                continue

            headline = frame['Input.sentence']
            if headline not in headlinesMappedToReactions:
                headlinesMappedToReactions[headline] = []
            reactionsConcatenated = ", ".join(frame['reaction'])
            headlinesMappedToReactions[headline].append({
                'workerId': worker['id'],
                'reaction': reactionsConcatenated
            })

    # keep only the headlines that have at least 2 reactions
    headlinesMappedToReactions = {headline: reactions for headline, reactions in headlinesMappedToReactions.items() if len(reactions) >= 2}
    saveToJsonFile(headlinesMappedToReactions, outputPath + 'headlines_mapped_to_reactions.json')

    headlinesMappedToIntentions = {}
    for worker in demographicWorkers:
        for frame in worker['annotatedFrames']:
            if len(frame['intent']) == 0:
                continue

            headline = frame['Input.sentence']
            if headline not in headlinesMappedToIntentions:
                headlinesMappedToIntentions[headline] = []
            intentsConcatenated = ", ".join(frame['intent'])
            headlinesMappedToIntentions[headline].append({
                'workerId': worker['id'],
                'intent': intentsConcatenated
            })

    # keep only the headlines that have at least 2 reactions
    headlinesMappedToIntentions = {headline: intents for headline, intents in headlinesMappedToIntentions.items() if len(intents) >= 2}
    saveToJsonFile(headlinesMappedToIntentions, outputPath + 'headlines_mapped_to_intentions.json')