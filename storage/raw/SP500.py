import pandas as pd
import matplotlib.pyplot as plt
import os

csv = pd.read_csv(os.path.join(os.path.dirname(__file__), 'SP500.csv'))

plt.plot(csv['date'], csv['price'])
plt.show()
