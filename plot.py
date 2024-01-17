from pandas import read_csv


import matplotlib
import matplotlib.pyplot as pyplot

dataset=read_csv("/opt/data/private/xwj/ML/data/test_set.csv",header=0,index_col=0)
values=dataset.values
groups=[0,1,2,3,4,5,6]
i=1

pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups),1,i)
    pyplot.plot(values[:,group])
    pyplot.title(dataset.columns[group],y=0.5,loc='right')
    i+=1

pyplot.savefig("xwj/ML/ett_plot.jpg",dpi=500)
pyplot.close()