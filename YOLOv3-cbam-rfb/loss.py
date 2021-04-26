import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
data1_loss =np.loadtxt("results1.txt") 
#data2_loss = np.loadtxt("valid_SCRCA_records.txt") 


x = data1_loss[:,0]
y = data1_loss[:,5]


#x1 = data2_loss[:,0]
#y1 = data2_loss[:,1]

fig = plt.figure(figsize = (7,5))    #figsize是图片的大小`
ax1 = fig.add_subplot(1, 1, 1) # ax1是子图的名字`

pl.plot(x,y,'g-',label=u'Dense_Unet(block layer=5)')
# ‘'g‘'代表“green”,表示画出的曲线是绿色，“-”代表画的曲线是实线，可自行选择，label代表的是图例的名称，一般要在名称前面加一个u，如果名称是中文，会显示不出来，目前还不知道怎么解决。
#p2 = pl.plot(x1, y1,'r-', label = u'RCSCA_Net')
pl.legend()
#显示图例
#p3 = pl.plot(x2,y2, 'b-', label = u'SCRCA_Net')
pl.legend()
pl.xlabel(u'iters')
pl.ylabel(u'loss')
plt.savefig("train_results_loss.png")
plt.title('Compare loss for different models in training')
plt.show()

