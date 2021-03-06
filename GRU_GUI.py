
from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget)
from GRUCalc import Main
import numpy as np

GRU = Main()
app = QApplication([])
window = QWidget()
window.setWindowTitle("FRC Gaussian Rating Utility")
layout = QGridLayout()
##### Load Event
layout.addWidget(QLabel("Event Name"),1,0)

eventLine = QLineEdit("")
layout.addWidget(eventLine,1,1)

eventButton = QPushButton("Load")
layout.addWidget(eventButton,1,2)

eventOut = QLineEdit("")
layout.addWidget(eventOut,1,3)
##### Calculate values
layout.addWidget(QLabel("steps , multiplier"),2,0)

stepLine = QLineEdit("")
layout.addWidget(stepLine,2,1)

stepButton = QPushButton("Run")
layout.addWidget(stepButton,2,2)

stepOut = QLineEdit("")
layout.addWidget(stepOut,2,3)
##### Display progress of calculation
stepProgress = QProgressBar()
layout.addWidget(stepProgress,3,0,1,4)
##### Get value for specific team
layout.addWidget(QLabel("# of team"),4,0)

teamLine = QLineEdit("")
layout.addWidget(teamLine,4,1)

teamButton = QPushButton("Get")
layout.addWidget(teamButton,4,2)

teamOut = QLineEdit("")
layout.addWidget(teamOut,4,3)
##### Input of team values
gameButton = QPushButton("Predict Game")
layout.addWidget(gameButton,5,0,1,2)

fileButton = QPushButton("File Output")
layout.addWidget(fileButton,5,2)

fileName = QLineEdit("")
layout.addWidget(fileName,5,3)

layout.addWidget(QLabel("Red 1"),6,0)
red1 = QLineEdit("")
layout.addWidget(red1,6,1)

layout.addWidget(QLabel("Blue 1"),6,2)
blue1 = QLineEdit("")
layout.addWidget(blue1,6,3)

layout.addWidget(QLabel("Red 2"),7,0)
red2 = QLineEdit("")
layout.addWidget(red2,7,1)

layout.addWidget(QLabel("Blue 2"),7,2)
blue2 = QLineEdit("")
layout.addWidget(blue2,7,3)

layout.addWidget(QLabel("Red 3"),8,0)
red3 = QLineEdit("")
layout.addWidget(red3,8,1)

layout.addWidget(QLabel("Blue 3"),8,2)
blue3 = QLineEdit("")
layout.addWidget(blue3,8,3)

layout.addWidget(QLabel("Red Out"),9,0)
redOut = QLineEdit("")
layout.addWidget(redOut,9,1)

layout.addWidget(QLabel("Blue Out"),9,2)
blueOut = QLineEdit("")
layout.addWidget(blueOut,9,3)

gameOut = QLineEdit("")
layout.addWidget(gameOut,10,0,1,4)
#####
window.setLayout(layout)

def loadTBA():
    print("loadTBA")
    eventKey = eventLine.text()
    try:
        GRU.loadTBA(eventKey)
        eventOut.setText(f"Loaded {GRU.numMatches} results")
    except Exception as e:
        eventOut.setText(str(e))

def runSteps():
    print("runSteps")
    try:
        line = stepLine.text()
        if ',' in line:
            steps,mult = line.split(",")
            steps = int(steps)
            mult = float(mult)
        else:
            steps = int(stepLine.text())
            mult = 1
        for i in range(steps):
            GRU.step(mult)
            stepProgress.setValue(int((i+1)/steps*100))
        stepOut.setText(f"log prob:{GRU.relLogProb(GRU.average, GRU.variance):.2f}")
    except Exception as e:
        stepOut.setText(str(e))

def getTeam():
    print("getTeam")
    try:
        teamNum = teamLine.text()
        avg,var = GRU.valFromNum(teamNum)
        out = f"{avg:.2f}+-{np.sqrt(var):.2f}"
        teamOut.setText(out)
    except Exception as e:
        teamOut.setText(str(e))

def fileOut():
    print("fileOut")
    try:
        struct = GRU.sortedTeamVals()
        with open(fileName.text(),'w') as fOut:
            for i in range(len(struct)):
                s = struct[i]
                fOut.write(f"#{i+1:<2} : {s['key']:8}|=>\t{s['avg']:.2f}+-{s['var']**0.5:.2f}\n")
    except Exception as e:
        fileName.setText(str(e))

def calcGame():
    print("calcGame")
    try:
        redAvg = 0
        redVar = 0
        teamNums = [red1.text(),red2.text(),red3.text()]
        for num in teamNums:
            if num != '':
                avg,var = GRU.valFromNum(num)
                redAvg += avg
                redVar += var
        redOut.setText(f"{redAvg:.2f}+-{redVar**0.5:.2f}")

        blueAvg = 0
        blueVar = 0
        teamNums = [blue1.text(),blue2.text(),blue3.text()]
        for num in teamNums:
            if num != '':
                avg,var = GRU.valFromNum(num)
                blueAvg += avg
                blueVar += var
        blueOut.setText(f"{blueAvg:.2f}+-{blueVar**0.5:.2f}")

        gameOut.setText(GRU.winStr(redAvg,redVar,blueAvg,blueVar))
    except Exception as e:
        redOut.setText(str(e))

eventButton.clicked.connect(loadTBA)
stepButton.clicked.connect(runSteps)
teamButton.clicked.connect(getTeam)
fileButton.clicked.connect(fileOut)
gameButton.clicked.connect(calcGame)

window.show()
app.exec()