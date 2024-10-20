import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('dataset_km_miles.csv')
print(data.head()) 

plt.scatter(data['km'], data['miles'], color='blue')
plt.xlabel('Kilometers')
plt.ylabel('Miles')
plt.title('Kilometers x Miles')
plt.grid(True)
plt.show()

