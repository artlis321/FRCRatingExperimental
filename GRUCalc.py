import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg
import numpy as np
import requests
from scipy.stats import norm

class Main():
    def __init__(self):
        self.teams_json = [] # json from TBA for list of all teams
        self.matches_json = [] # json from TBA for list of all matches

        self.teamKeysAll = [] # list of keys for all teams at event
        self.teamKeys = [] # list of keys for all teams that have had a match
        self.teamInds = {} # dictionary that converts a team key to an index

        self.numTeams = 0 # number of teams that have had a match
        self.numMatches = 0 # number of matches (each side counts as its own 'match')

        self.matchBool = [] # boolean values for which matches are included in calculation
        # ignore later matches to check predictive value
        # ignore matches with technical errors

        self.participation = np.zeros( (0,0) ) # participation matrix P, each row is a match with 1's for teams present
        self.score = [] # score vector s, one value for each match (how many points did the alliance score?)
        
        self.average = []
        self.variance = []

    def loadTBA(self,eventString):
        with open("TBAAuthKey.txt","r") as keyFile: # gets read key from file
            key = keyFile.readline() 
        baseURL = 'http://www.thebluealliance.com/api/v3/'
        header = {'X-TBA-Auth-Key':key}
        
        def getTBA(url): # defines function for pulling data from TBA
            return requests.get(baseURL+url,headers=header).json()

        self.teamsAPI = getTBA("event/"+eventString+"/teams") # pulls data about teams
        self.matchesAPI = getTBA("event/"+eventString+"/matches") # pulls data about matches
        self.matchesAPI = [match for match in self.matchesAPI if match['alliances']['red']['score'] != -1] # removes matches that weren't played yet
        self.matchBool = np.array([True for i in range(2*len(self.matchesAPI))]) # (unused at the moment) which results are to be used in calculation

        self.updateBaseVals()

        self.average = np.array([10.0 for i in range(self.numTeams)])
        self.variance = np.array([100.0 for i in range(self.numTeams)])

    def valFromNum(self,num):
        key = 'frc'+str(num)
        ind = self.teamInds[key]
        return self.average[ind],self.variance[ind]

    def winStr(self,redAvg,redVar,blueAvg,blueVar):
        if blueAvg > redAvg:
            return f"Blue wins {norm.cdf((blueAvg-redAvg)/(blueVar+redVar)**0.5):0.2f}"
        if redAvg > blueAvg:
            return f"Red wins {norm.cdf((redAvg-blueAvg)/(blueVar+redVar)**0.5):0.2f}"
        else:
            return "A draw!?"

    def sortedTeamVals(self):
        struct = np.array([('',0,0) for _ in range(self.numTeams)],dtype=[('key',object),('avg',float),('var',float)])
        for i in range(self.numTeams):
            key = self.teamKeys[i]
            ind = self.teamInds[key]
            avg = self.average[ind]
            var = self.variance[ind]
            struct[i] = (key,avg,var)
        struct = np.flip(struct[np.argsort(struct['avg'])])
        return struct

    def updateBaseVals(self):
        self.teamKeysAll = [t['key'] for t in self.teamsAPI] # converts api to list
        self.numMatches = sum(self.matchBool) # counts number of match results to be used for calc

        tempNumTeams = len(self.teamKeysAll) # gets list of all teams
        tempTeamInds = {self.teamKeysAll[i]:i for i in range(tempNumTeams)} # assigns each team an index for list comprehension

        self.participation = np.zeros( (self.numMatches,tempNumTeams) ) # creates participation matrix (rows are matches, columns are teams)
        self.score = np.empty( (self.numMatches) ) # creates score vector (value for each match result)

        # takes data from matchesAPI and separates it into the two sides
        matchesSplit = []
        for match in self.matchesAPI:
            matchesSplit.append(match['alliances']['red'])
            matchesSplit.append(match['alliances']['blue'])

        # gets teams and self.scores from the matches
        index = 0
        for i in range(len(self.matchBool)): 
            if self.matchBool[i]:
                teams = matchesSplit[i]['team_keys']
                for t in teams:
                    self.participation[index][ tempTeamInds[t] ] = 1
                self.score[index] = matchesSplit[i]['score']
                index += 1

        self.teamKeys = self.teamKeysAll.copy()
        for t in range(tempNumTeams-1,-1,-1):
            numMatches = np.sum(self.participation[:,t])
            if numMatches == 0:
                del self.teamKeys[t]
                self.participation = np.delete(self.participation,t,1)
        self.numTeams = len(self.teamKeys)
        self.numMatches = len(self.participation)
        self.teamInds = { self.teamKeys[i]:i for i in range(self.numTeams) }

    def logNormalizedGaussian(self,average,variance,x):
        return -(x-average)**2/(2*variance)-0.5*np.log(2*np.pi*variance) # np.log( np.exp(-(x-average)**2/(2*variance)) / (2*np.pi*variance)**(0.5) )

    def relLogProb(self,average,variance):
        matchAverages = np.matmul(self.participation,average)
        matchVariances = np.matmul(self.participation,variance)
        logProbAll = self.logNormalizedGaussian(matchAverages,matchVariances,self.score)
        logProb = np.sum(logProbAll)
        return logProb

    def gradAll(self):
        sCalc = np.matmul(self.participation,self.average)-self.score
        vCalc = np.matmul(self.participation,self.variance)
        gradAvg = np.zeros(self.numTeams)
        gradVar = np.zeros(self.numTeams)

        for j in range(self.numMatches):
            gradAvg -= self.participation[j] * sCalc[j] / vCalc[j]
            gradVar += self.participation[j] * (sCalc[j]**2 / (2*vCalc[j]**2) - 0.5/(2*np.pi*vCalc[j]))

        return gradAvg,np.linalg.norm(gradAvg),gradVar,np.linalg.norm(gradVar)

    def step(self,mult):
        gradAvg,lenAvg,gradVar,lenVar = self.gradAll()
        self.average += mult*gradAvg / lenAvg**0.9
        self.variance += mult*gradVar / lenVar**0.9
        self.average = np.abs(self.average)
        self.variance = np.abs(self.variance)
        #print(self.relLogProb(self.average,self.variance))

    def multiStep(self,num):
        for i in range(num):
            self.step()