from unittest import TestCase 
from Distance_Velocity import KF
import numpy as np
class TestKF(TestCase):
    def test_can_construct(self):
        x = 0.2
        v = 2.3
        kf = KF(initial_x=x, initial_v=v,accel_variance = 1.2)
        self.assertAlmostEqual(kf.pos, x)
        self.assertAlmostEqual(kf.vel, v)
    
    def test_can_predict(self):
        x = 0.2
        v = 2.3
        kf = KF(initial_x=x, initial_v=v, accel_variance=1.2)
        dt = 0.1
        predicted_state = kf.predict(dt)
        self.assertEqual(kf.getP.shape, (2,2))

    def test_can_update(self):
        x = 0.2
        v = 2.3
        kf = KF(initial_x=x, initial_v=v, accel_variance=1.2)
        det_before = np.linalg.det(kf.getP)
        kf.update(meas_value=0.1, meas_variance=0.01)
        det_after = np.linalg.det(kf.getP)
        self.assertLess(det_after, det_before)

