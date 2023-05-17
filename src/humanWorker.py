from worker import Worker
from collections import namedtuple


class HumanWorker(Worker):

        def __init__(self, id, demographics, mediaConsumptionRegimen, approvalRating):
            super().__init__(id)
            # self.demographics = demographics

            self.age = HumanWorker.removeDefaultValues(demographics['age'], '0')
            self.age = self.age[0] if len(self.age) and len(self.age[0]) else 'unknown'

            self.gender = HumanWorker.removeDefaultValues(demographics['gender'], '0')
            self.gender = self.gender[0] if len(self.gender) and len(self.gender[0]) else 'unknown'

            self.education = HumanWorker.removeDefaultValues(demographics['education'], '0')
            self.education = self.education[0] if len(self.education) and len(self.education[0]) else 'unknown'

            self.race = demographics['race']
            self.race = [racialGroup for racialGroup in self.race if 'true' in self.race[racialGroup]]
            self.race = self.race if len(self.race) > 0 else []

            self.mediaConsumptionRegimen = [media for media in mediaConsumptionRegimen if 'true' in mediaConsumptionRegimen[media]]
            self.mediaConsumptionRegimen = self.mediaConsumptionRegimen if len(self.mediaConsumptionRegimen) > 0 else []

            self.approvalRating = approvalRating
            self.annotatedFrames = []


        def addFrame(self, frame):
            self.annotatedFrames.append(frame)


        def setFrames(self, frames): # todo parse the data first and then create Frame objects out of them
            self.annotatedFrames = frames

        @staticmethod
        def _decode(humanWorkerDict):
            return namedtuple('X', humanWorkerDict.keys())(*humanWorkerDict.values())


        @staticmethod
        def removeDefaultValues(attribute, defaultValue):
            return [x for x in attribute if x != defaultValue]