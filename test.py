# Created by jikangwang at 3/17/19


from __future__ import absolute_import, division, print_function

from tensorflow import keras
import numpy as np

p1 =np.array([2,3,4])
p2=np.array([4,5,6])

print(np.sum(abs(p1-p2))/len(p1))