import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('LRSample.csv')
print(data.shape)
data.head()

X=data['X'].values
Y=data['Y'].values
mean_x = np.mean(X)
mean_y = np.mean(Y)
t = len(X)
numer = 0
denom = 0
for i in range(t):
  numer += ( X[i] - mean_x ) * ( Y[i] - mean_y )
  denom += ( X[i] - mean_x ) ** 2
  m = numer / denom
  c = mean_y - ( m * mean_x )
print("Slope(m) = ", m)
print("Intercept(c) = ", c )

y= m*X +c
#Plotting the line
plt.plot(X,y,color="red",label='Regression Line')
plt.scatter(X,Y, color="blue", label='Scatter Plot')
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()


