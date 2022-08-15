import matplotlib.pyplot as plt

fig=plt.imread('fig22.png')
ax = fig.add_subplot(121)
ax.set_xticks(np.arange(0, 30+1, 5))
plt.savefig('fig22redigert.png')