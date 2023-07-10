import re

class DemographicUtils:
    emptyHeader = '''
    Worker ID: Unknown\nAge: Unknown\nGender: Unknown\nEducation: Unknown\nRace: unknown\nMedia Diet: unknown\n
    '''.strip()
    emptyDemographicHeader = '''
    Age: Unknown\nGender: Unknown\nEducation: Unknown\nRace: unknown\nMedia Diet: unknown\n
    '''.strip()
    @staticmethod
    def maskDemographicHeaderAttributes(trajectory):
        headerStartIdx = trajectory.find('<header>')
        ageAttributeIdx = trajectory.find('Age: ', headerStartIdx)
        headerEndIdx = trajectory.find('</header>')
        return trajectory[:ageAttributeIdx] + DemographicUtils.emptyDemographicHeader + trajectory[headerEndIdx:]

    @staticmethod
    def maskHeaderAttributes(trajectory):
        headerStartIdx = trajectory.find('<header>')
        headerEndIdx = trajectory.find('</header>')
        return trajectory[:headerStartIdx] + DemographicUtils.emptyHeader + trajectory[headerEndIdx:]

    @staticmethod
    def extractWorkerId(trajectory):
        workerIdTagIdx = trajectory.find('Worker ID: ')
        endWorkerIdIdx = trajectory.find('\nAge: ', workerIdTagIdx)
        return trajectory[workerIdTagIdx + len('Worker ID: '): endWorkerIdIdx]

    @staticmethod
    def extractHeader(trajectory):
        headerStartIdx = trajectory.find('<header>')
        headerEndIdx = trajectory.find('</header>')
        return trajectory[headerStartIdx + len('<header>'): headerEndIdx]