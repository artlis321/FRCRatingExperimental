Running GRU_GUI gives you a visual interface with which one may predict results from FRC games using data from the blue alliance.

Event Name : string corresponding to the event (from the TBA url)
- press 'Load' to load the data
- output box displays number of results loaded / potential error

steps, multiplier : string corresponding to number of steps made in the calculation
- optional multiplier may be added for finer adjustments of the gradient descent
- output box displays the logarithm of the relative probability after descent is complete, higher is better

\# of team : input the number of a team and press 'get' to see their average and stdev

file output : lets you output a file with the sorted average and stdev of all teams

the rest lets you predict the outcome of a full game by inputting numbers of several teams