
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
layout.addWidget(QLabel("# of steps to run"),2,0)

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
layout.addWidget(QLabel("Game Prediction"),5,0)

gameButton = QPushButton("Calculate")
layout.addWidget(gameButton,5,1)

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
#####
window.setLayout(layout)

def loadTBA():
    print("loadTBA")
    eventKey = eventLine.text()
    try:
        GRU.loadTBA(eventKey)
        GRU.average = np.array([10.0 for i in range(GRU.numTeams)])
        GRU.variance = np.array([10.0 for i in range(GRU.numTeams)])
        eventOut.setText("Done!")
    except Exception as e:
        eventOut.setText(str(e))

def runSteps():
    print("runSteps")
    try:
        steps = int(stepLine.text())
        for i in range(steps):
            GRU.step()
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
        redOut.setText(f"{redAvg:.2f}+-{np.sqrt(redVar):.2f}")

        blueAvg = 0
        blueVar = 0
        teamNums = [blue1.text(),blue2.text(),blue3.text()]
        for num in teamNums:
            if num != '':
                avg,var = GRU.valFromNum(num)
                blueAvg += avg
                blueVar += var
        blueOut.setText(f"{blueAvg:.2f}+-{np.sqrt(blueVar):.2f}")
    except Exception as e:
        redOut.setText(str(e))

eventButton.clicked.connect(loadTBA)
stepButton.clicked.connect(runSteps)
teamButton.clicked.connect(getTeam)
gameButton.clicked.connect(calcGame)

window.show()
app.exec()