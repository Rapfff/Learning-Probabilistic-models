import matplotlib.pyplot as plt

#log = [float(i) for i in LOGLIKELI]
#ttime = [float(i) for i in TIME]

fig, ax1 = plt.subplots()
#xx = [10**1,10**2,10**3,10**4]
#xx = [10**2,10**3,10**4,10**5]
xx = [1,2,3,4,5,6,7,8]
yy = [-10.0920,-6.2974,-4.5939,-3.0973,-2.9740,-2.7233,-2.5135,-2.5135]
color = 'tab:red'
ax1.set_xlabel('number of sequences')
ax1.set_ylabel('loglikelihood')
ax1.plot(xx, yy, color=color)
plt.show()

