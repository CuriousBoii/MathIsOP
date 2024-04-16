import numpy as np
class KF:
    def __init__(self, initial_x, initial_v, accel_variance):
        self.x = np.array([initial_x, initial_v])
        self.P = np.eye(2)
        print(self.P)
        self.accel_variance = accel_variance
        print("Kalman Filter Initialized")

    def predict(self, dt:float):
        F = np.array([[1, dt], [0, 1]])
        new_x = F.dot(self.x)
        G = np.array([0.5* dt**2,dt]).reshape(2,1)
        new_P = F.dot(self.P).dot(F.T) + G.dot(G.T) * self.accel_variance
        self.x = new_x
        self.P = new_P
    
    def update(self, meas_value:float, meas_variance:float):
        z = np.array([meas_value])
        R = np.array([meas_variance])
        H = np.array([1,0]).reshape([1,2])
        y = z - H.dot(self.x)
        S  = H.dot(self.P).dot(H.T) + R
        K = self.P.dot(H.T).dot(np.linalg.inv(S))

        new_x = self.x + K.dot(y)
        new_P = (np.eye(2) - K.dot(H)).dot(self.P)
        self.x = new_x
        self.P = new_P

    @property
    def getP(self):
        return self.P
    @property
    def pos(self):
        return self.x[0]
    @property
    def vel(self):
        return self.x[1]
    
kf = KF(2, 5, 1.2)

kf.predict(1)
print(kf.pos)
print(kf.getP)
kf.update(0.1, 0.01)

