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
        print("X_predict=\n",self.X_predict)
        self.P_predict = F @ self.P @ F.T + V @ self.P @ V.T
    
    def update(self, Z):
        # Update step
        X, P = self.X_predict, self.P
        self.Z = np.array(Z)
        print("Z=\n",Z)
        soc, d = X[0][0], X[1][0]
        H = np.array([[1, 0], [d/(2*(1-soc)), soc/(2*(1-soc))]])
        print("H=\n", H)
        R = np.array([[0.0], [0.0]])
        Z_estimate = H @ X  + R
        print("Z_estimate=\n",Z_estimate)
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        print("K=\n", K)
        self.X = X + K @ (Z - Z_estimate)
        self.P = (np.eye(2) - K @ H) @ P

    @property
    def state(self):
        return self.X
    


EKF = EKFv2(0.5, 50)
print(EKF.state)
EKF.predict(0.5)
EKF.update([[0.495], [50.5]])
print("new state is\n", EKF.state)

