import numpy as np

class EKFv2:
    def __init__(self, soc, distance_travelled) -> None:
        self.X = np.array([[soc], [distance_travelled]])
        self.P = np.eye(2)

    def predict(self, delta_d):
        # Prediction step
        X = self.X
        soc, distance_travelled = X[0][0], X[1][0]
        cr = (1-soc)/distance_travelled
        change = np.array([[-1*cr*delta_d], [delta_d]])
        F = np.array([[1, 0], [0, 1]])
        V = np.array([[-1*delta_d, -1*cr],[0, 1]])
        self.X_predict = np.add(X, change)
        #print("X_predict=\n",self.X_predict)
        self.P_predict = F @ self.P @ F.T + V @ self.P @ V.T
        #print("P_preddict=\n", self.P_predict)
    
    def update(self, soc_m, distance_travelled):
        # Update step
        DTE_m = 100 - distance_travelled
        X, P = self.X_predict, self.P
        Z = np.array([[soc_m],[DTE_m]])
        #print("Z=\n",Z)
        soc, d = X[0][0], X[1][0]
        H = np.array([[1, 0], [d/(2*(1-soc)), soc/(2*(1-soc))]])
        #print("H=\n", H)
        #R = np.array([[0], [0]])
        Z_estimate = H @ X
        print("Z_estimate=\n",Z_estimate)
        S = H @ P @ H.T + np.array([[1, 0],[0, 1]])
        #print("S = \n", S)
        K = P @ H.T @ np.linalg.inv(S)
        #print("K=\n", K)
        self.X = X + K @ (Z - Z_estimate)
        self.P = (np.eye(2) - K @ H) @ P

    @property
    def state(self):
        return self.X
    
    @property
    def consumption_factor(self):
        return self.X[0][0]/self.X[1][0]
    
    @property
    def DTE(self):
        X = self.X
        soc, distance_travelled = X[0][0], X[1][0]
        return soc*distance_travelled/(1-soc)

EKF = EKFv2(0.505, 49.5)
print(EKF.state)
measurements = [[0.5, 49.99], [0.495, 50.3], [0.49, 51.1], [0.485, 51.4]]
for measurement in measurements:
    EKF.predict(delta_d=0.5)
    EKF.update(measurement[0], measurement[1])
    print("new state is\n", EKF.state)
    print("new DTE is\n", EKF.DTE)
    print("-------------------------------------------------------------------")