'''
Trajectory format, with an example:
WorkerID: XYZ
Gender: A/B/C/NA
Race: ...
...
reactions: [
    {
        "headline": "headline1",
        "reaction": "reaction1",
    }
]
perceptions: [
    {
        "headline": "headline1",
        "perception": "perception1",
    }
]

current headline: 'headline1'
what is the perceived label?
'''
# the trajectories contain three parts; header -- trajectories -- question


class MRFSequence(object):
    def __init__(self):
        self.trajectories = []
        self.header = {}
        self.question = {}

    def add_trajectory(self, trajectory):
        self.trajectories.append(trajectory)

    def add_header(self, header):
        self.header = header

    def add_question(self, question):
        self.question = question

    def __str__(self): # todo fix this to match the intended format
        return str(self.header) + '\n' + str(self.trajectories) + '\n' + str(self.question)