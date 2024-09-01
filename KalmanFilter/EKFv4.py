import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, subprocess

subprocess.run("cls", shell = True)

'''
This is the 4th version of finding dte. This uses speedometer reading as the input.

To initialize the object, give initial soc, consumption ratio(cr) and dte; all at current time.
Predict step requires current speed reading from speedometer.
Update step requires only soc reading at current time.

'''

class EKFv4:
    def __init__(self, soc, cr, dte) -> None:
        # cr = consumption ratio. It is defined as (1-soc)/distance travelled.
        self.X = np.array([[soc], [0]])
        self.P = np.eye(2)
        self.cr = cr
        self.delta_t = 1
        self.v_t0 = 0
        self.Z = np.array([[soc], [dte]])

    def predict(self, v_t1):
        # Prediction step
        X = self.X
        cr = self.cr
        delta_t = self.delta_t 
        # The speedometer data is in kmph. We convert it to mps and find distance travelled.
        delta_d = (v_t1 + self.v_t0)/2 * delta_t * 5/18000
        self.delta_d = delta_d
        change = np.array([[-1*cr*delta_d], [delta_d]])
        print("change=\n", change)
        F = np.array([[1, 0], [0, 1]])
        self.X_predict = np.add(F @ X, change)
        print("X_predict=\n",self.X_predict)
        self.P_predict = F @ self.P @ F.T + np.array([[1, 0], [0, 1]])
        #print("P_predict=\n", self.P_predict)
        self.v_t0 = v_t1

    def update(self, soc_t):
        # Update step
        dte = self.Z[1][0]
        Z = np.array([[soc_t],[dte]])
        X = self.X_predict
        P = self.P_predict
        cr = self.cr
        soc = self.X[0][0]
        H = np.array([[1, 0],[1/(2*cr), soc/(1-soc)]])
        Z_estimate = H @ X
        print("Z_estimate=\n",Z_estimate)
        S = H @ P @ H.T + np.array([[1, 0],[0, 1]])
        #print("S = \n", S)
        K = P @ H.T @ np.linalg.inv(S)
        print("K=\n", K)
        self.X = X + K @ (Z - Z_estimate)
        self.P = (np.eye(2) - K @ H) @ P
        self.updateVariables()
        print("crude=\n",soc_t/self.cr)
        return soc_t/self.cr
    def updateVariables(self):
        X = self.X
        cr = self.cr
        soc = X[0][0]
        distance_travelled = X[1][0]
        delta_d = self.delta_d
        new_cr = (1/cr) + (delta_d/(1-soc))
        new_cr = 1/new_cr
        print("new_cr=\n", new_cr)
        self.cr = new_cr
        self.Z[1][0] = soc/new_cr
        
    @property
    def state(self):
        return self.X
    
    @property
    def DTE(self):
        Z = self.Z
        dte = Z[1][0]
        return dte
    
def dotted_line():
    print("------------------------------------------------------------------------")

EKF = EKFv4(0.999, 0.01, 100)
print(EKF.state)
speeds =[  1, 10,   30,   30,   30,   30,   30,   30,   40,  40 ,   50,   50,   50]
soc_m = [99.9,99.9,99.99,99.99,99.98,99.99,99.98,99.97,99.98,99.97,99.97,99.96,99.96]
ans = []
crude = []
for index, soc_t in enumerate(soc_m):
    EKF.predict(speeds[index])
    crude.append(EKF.update(soc_t/100))
    print("new state is\n", EKF.state)
    print("new DTE is\n", EKF.DTE)
    dotted_line()
    ans.append(EKF.DTE)

# plt.plot(speeds)
# dotted_line()
print(ans, crude)
# plt.show()
