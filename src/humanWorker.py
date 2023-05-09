from worker import Worker
from collections import namedtuple


class HumanWorker(Worker):

        def __init__(self, id, demographics, mediaConsumptionRegimen, approvalRating):
            super().__init__(id)
            # self.demographics = demographics

            self.age = HumanWorker.removeDefaultValues(demographics['age'], '0')
            self.age = self.age[0] if len(self.age) else None

            self.gender = HumanWorker.removeDefaultValues(demographics['gender'], '0') if len(demographics['gender']) > 1 else demographics['gender'][0] if len(demographics['gender'][0]) else None

            self.education = HumanWorker.removeDefaultValues(demographics['education'], '0') if len(demographics['education']) > 1 else demographics['education'][0] if len(demographics['education'][0]) else None

            self.race = demographics['race']

            self.mediaConsumptionRegimen = mediaConsumptionRegimen
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