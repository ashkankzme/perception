from .worker import Worker
from collections import namedtuple


class HumanWorker(Worker):

        def __init__(self, id, demographics, mediaConsumptionRegime, approvalRating):
            super().__init__(id)
            self.ageBracket = ageBracket
            self.education = education
            self.mediaConsumptionRegimen = mediaConsumptionRegimen
            self.ethnicity = ethnicity
            self.gender = gender
            self.approvalRating = approvalRating
            self.annotatedFrames = []


        def addFrame(self, frame):
            self.annotatedFrames.append(frame)


        def setFrames(self, frames): # todo parse the data first and then create Frame objects out of them
            self.annotatedFrames.extend(frames)

        @staticmethod
        def _decode(humanWorkerDict):
            return namedtuple('X', humanWorkerDict.keys())(*humanWorkerDict.values())
