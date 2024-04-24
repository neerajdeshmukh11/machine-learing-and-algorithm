from sklearn.cluster import KMeans
import numpy as np
x=np.array([[1.713,1.586],[0.180, 1.786],[0.353,1.240],
            [0.940, 1.566],[1.486, 0.759],[1.266, 1.106],[1.540, 0.419],[0.459, 1.799],[0.773, 0.186]])
y=np.array([0,1,1,0,1,0,1,1,1])
kmeans=KMeans(n_clusters=3, random_state=0).fit(x,y)
print("The input data is:\nVAR1\tVAR2\tClass")
i=0
for val in x:
  print(val[0],"\t", val[1],"\t",y[i])
  i+=1
print("="*20)
print("The test data to predict:")
test_data=[]
VAR1=float(input("Enter the value for VAR1:"))
VAR2=float(input("Enter the value for VAR2:"))
test_data.append(VAR1)
test_data.append(VAR2)
print("="*20)
print("The predicted class is:", kmeans.predict([test_data]))
