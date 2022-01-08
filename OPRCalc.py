import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg
import numpy as np
import requests
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm

class GRUState(object):
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

    def loadTBA(self,authkey,eventString):
        baseURL = 'http://www.thebluealliance.com/api/v3/'
        header = {'X-TBA-Auth-Key':authkey}
        
        def getTBA(url):
            return requests.get(baseURL+url,headers=header).json()

        self.teamsAPI = getTBA("event/"+eventString+"/teams")
        self.matchesAPI = getTBA("event/"+eventString+"/matches")
        self.matchBool = np.array([True for i in range(2*len(self.matchesAPI))])

        self.updateBaseVals()

    def updateBaseVals(self):
        self.teamKeysAll = [t['key'] for t in self.teamsAPI]
        self.numMatches = sum(self.matchBool)

        tempNumTeams = len(self.teamKeysAll)
        tempTeamInds = {self.teamKeysAll[i]:i for i in range(tempNumTeams)}

        self.participation = np.zeros( (self.numMatches,tempNumTeams) )
        self.score = np.empty( (self.numMatches) )

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
        self.teamInds = { self.teamKeys[i]:i for i in range(self.numTeams) }

    def relLogProb(self,average,variance):
        matchAverages = np.matmul(self.participation,average)
        matchVariances = np.matmul(self.participation,variance)
        logProbAll = self.logNormalizedGaussian(matchAverages,matchVariances,self.score)
        logProb = np.sum(logProbAll)
        return logProb

    def gradientLogProb(self,average,variance):
        initialProb = self.relLogProb(average,variance)
        averageGrad = np.empty(self.numTeams)
        varianceGrad = np.empty(self.numTeams)
        for i in range(self.numTeams):  
            testAverage = average.copy()
            testVariance = variance.copy()
            averageStep = np.sqrt(variance[i])/1e5
            testAverage[i] += averageStep
            newAvgProb = self.relLogProb(testAverage,variance)
            averageGrad[i] = (newAvgProb - initialProb)/averageStep
            testAverage[i] = average[i]
            varianceStep = variance[i]/1e5
            testVariance[i] += varianceStep
            newVarProb = self.relLogProb(average,testVariance)
            varianceGrad[i] = (newVarProb-initialProb)/varianceStep
        slope = (sum(varianceGrad**2)+sum(averageGrad**2))**0.5
        return averageGrad,varianceGrad,slope

    def maximizeProbability(self,startAverage,startVariance,steps,plot=False):
        averages = [startAverage]
        variances = [startVariance]
        probabilities = [self.relLogProb(startAverage,startVariance)]
        multipliers = [10]
        gradientMult = np.e**10
        for i in range(steps):
            averageGrad,varianceGrad,slope = self.gradientLogProb(averages[-1],variances[-1])
            newAvg = averages[-1]+averageGrad*gradientMult/slope
            newVar = np.abs(variances[-1]+varianceGrad*gradientMult/slope)
            newProb = self.relLogProb(newAvg,newVar)
            if newProb <= probabilities[-1]:
                gradientMult *= 1/np.e
                averages.append(averages[-1])
                variances.append(variances[-1])
                probabilities.append(probabilities[-1])
                if gradientMult == 0:
                    break
                multipliers.append(np.log(gradientMult))
            else:
                averages.append(newAvg)
                variances.append(newVar)
                probabilities.append(newProb)
                multipliers.append(np.log(gradientMult))
        x = range(len(averages))
        averages = np.array(averages)
        variances = np.array(variances)**(0.5)
        if plot:
            plt.plot(x,probabilities)
            plt.plot(x,multipliers)
            for t in [10]:
                plt.errorbar(x,averages[:,t],variances[:,t])
            plt.show()
        return averages[-1],variances[-1],probabilities[-1]

    def logNormalizedGaussian(self,average,variance,x):
        return -(x-average)**2/(2*variance)-0.5*np.log(2*np.pi*variance) # np.log( np.exp(-(x-average)**2/(2*variance)) / (2*np.pi*variance)**(0.5) )


fromFile = False

if fromFile:
    teamsAPI = []
    matchesAPI = []
else:
    baseURL = 'http://www.thebluealliance.com/api/v3/'
    header = {'X-TBA-Auth-Key':'y979sLSaQhjmZIGv5UDsIt4Oh5p2yZQiwGaKkKECky1LT4SEV2vpGfGNfEGLAbav'}
    eventString = '2019cmptx'

s = GRUState()
s.loadTBA('y979sLSaQhjmZIGv5UDsIt4Oh5p2yZQiwGaKkKECky1LT4SEV2vpGfGNfEGLAbav','2019cmptx')
bestAvg = []
bestVar = []
bestProb = -200
probabilities = []
for i in tqdm(range(1000)):
    avg,var,prob = s.maximizeProbability(np.zeros(s.numTeams), ((np.random.random(s.numTeams)+1)*10)**2, 2000)
    probabilities.append(prob)
    if prob > bestProb:
        bestAvg = avg
        bestVar = var
        bestProb = prob
        print(avg)
        print(var)
        print(prob)
        print("_____________________")
        print("")
plt.hist(probabilities,bins=20)
plt.show()