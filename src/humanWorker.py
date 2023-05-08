from .worker import Worker
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


        def addFrames(self, frames):
            self.annotatedFrames.extend(frames)
