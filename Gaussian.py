import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data_size = 20
x = np.linspace(0,1,data_size)
noise = np.random.uniform(low=1.0,high=0.1,size=data_size )*0.2

y=np.sin(2.0*np.pi*x)+noise
plt.scatter(x,y)

x_line = np.linspace(0,1,1000)
sin_x = np.sin(2.0*np.pi*x_line)
plt.plot(x_line,sin_x,'red')

poly = PolynomialFeatures(degree=3)
poly.fit(x.reshape(-1,1))
x_poly_3 = poly.transform(x.reshape(-1,1))

lin_reg_3 = LinearRegression().fit(x_poly_3,y)

x_line_poly_3 = poly.fit_transform(x_line.reshape(-1,1))
plt.plot(x_line,lin_reg_3.predict(x_line_poly_3))
plt.show()