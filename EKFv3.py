import numpy as np

class EKFv3:
    def __init__(self, soc, distance_travelled) -> None:
        self.X = np.array([[soc], [distance_travelled]])
        self.P = np.eye(2)
        self.d_0 = distance_travelled

    def predict(self):
        # Prediction step
        X = self.X
        soc, distance_travelled = X[0][0], X[1][0]
        delta_d = distance_travelled - self.d_0
        
        cr = (1-soc)/distance_travelled
        change = np.array([[-1*cr*delta_d], [delta_d]])
        F = np.array([[1, 0], [0, 1]])
        V = np.array([[-1*delta_d, -1*cr],[0, 1]])
        self.X_predict = np.add(X, change)
        print("X_predict=\n",self.X_predict)
        self.P_predict = F @ self.P @ F.T + V @ self.P @ V.T
        print("P_preddict=\n", self.P_predict)
    
    def update(self, soc_m, distance_travelled):
        # Update step
        DTE_m = self.d_0 - distance_travelled
        X, P = self.X_predict, self.P
        Z = np.array([[soc_m],[DTE_m]])
        print("Z=\n",Z)
        soc, d = X[0][0], X[1][0]
        H = np.array([[1, 0], [d/(2*(1-soc)), soc/(2*(1-soc))]])
        print("H=\n", H)
        #R = np.array([[0], [0]])
        Z_estimate = H @ X
        print("Z_estimate=\n",Z_estimate)
        S = H @ P @ H.T + np.array([[1, 0],[0, 1]])
        print("S = \n", S)
        K = P @ H.T @ np.linalg.inv(S)
        print("K=\n", K)
        self.X = X + K @ (Z - Z_estimate)
        self.P = (np.eye(2) - K @ H) @ P
        self.d_0 = distance_travelled

    @property
    def state(self):
        return self.X
    
    @property
    def DTE(self):
        X = self.X
        soc, distance_travelled = X[0][0], X[1][0]
        return soc*distance_travelled/(1-soc)


EKF = EKFv3(0.9, 10)
print(EKF.state)
measurements = [[.84, 12.5], [.835, 13], [82, 13.5], [81, 14], [.805, 14.5]]
for measurement in measurements:
    EKF.predict()
    EKF.update(measurement[0], measurement[1])
    print("new state is\n", EKF.state)
    print("new DTE is\n", EKF.DTE)
    print("-------------------------------------------------------------------")


