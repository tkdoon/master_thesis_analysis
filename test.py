import matplotlib.pyplot as plt

fig=plt.figure(figsize=(16,12))
plt.plot([1,2,3],[1,2,3])
fig_hold=fig
plt.plot([2,4,6],[1,2,3])
plt.ylabel("konnntiha")
plt.show()
# fig_hold.show()
print(fig)
print(fig_hold)