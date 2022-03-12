import matplotlib.pyplot as plt
import numpy as np


Data_Categories =   ['Airplane',    'Bottle',   'Car',      'Sofa',     'Bench',    'Cellphone',    'Bike',     'Mean']
ours = {
    'Loss' :        [0.471,         1.248,       0.957,     1.755,       1.301,      1.340,          1.831,     1.274],
    'Time' :        [40.328,        44.556,      38.063,    40.959,      45.661,     49.283,         39.969,    42.688]
}
Ours_New = {
    'Loss' :        [0.450,         1.354,       0.975,     1.780,      1.438,      1.297,           1.765,     1.294],
    'Time' :        [20.081,        17.517,      17.475,    19.921,     17.584,     17.932,          17.340,    18.264]
}
PSGN = {
    'Loss' :        [0.465,         1.482,       0.998,     1.833,      2.170,      1.353,           1.989,     1.47],
    'Time' :        [439.627,       445.045,     431.720,   420.135,    419.151,   405.803,        452.540,   430.574]
}
Pixel2Point_InitialPC = {
    'Loss' :        [0.484,         1.514,       1.034,     1.869,       1.516,      3.444,          1.986,      1.692],
    'Time' :        [50.739,        63.831,      54.053,    57.779,     45.532,     49.454,         48.471,     52.837]
}
Pixel2Point = {
    'Loss' :        [0.463,         1.379,       1.008,     1.879,       1.444,      1.306,          1.967,      1.349],
    'Time' :        [41.648,        38.414,      38.258,    41.264,      37.027,     34.390,         36.822,     38.260]
}


###################### Compare our models to others' ##################

def showLossPlot():
    plt.plot(Data_Categories, Ours_New['Loss'], label='Ours', linestyle= '--', linewidth=4.5)    
    plt.plot(Data_Categories, Pixel2Point['Loss'], label='Pixel2Point (w/o PC)', linestyle=':', linewidth=4.5)
    plt.plot(Data_Categories, Pixel2Point_InitialPC['Loss'], label='Pixel2Point (w PC)', linestyle=':', linewidth=4.5)
    plt.plot(Data_Categories, PSGN['Loss'], label='PSGN', linestyle='-.', linewidth=4.5)
    plt.legend(prop={'size':20})
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    plt.xlabel('Data Category', fontweight ='bold', fontsize=20)
    plt.ylabel('Chamfer Loss', fontweight ='bold', fontsize=20)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig('/home/ahz/Desktop/My papers/1/PlotLossResult.png')    
    plt.show()


def showTimePlot():
    plt.plot(Data_Categories, Ours_New['Time'], label='Ours', linestyle= '--', linewidth=4.5)    
    plt.plot(Data_Categories, Pixel2Point['Time'], label='Pixel2Point (w/o PC)', linestyle=':', linewidth=4.5)
    plt.plot(Data_Categories, Pixel2Point_InitialPC['Time'], label='Pixel2Point (w PC)', linestyle=':', linewidth=4.5)
    plt.plot(Data_Categories, PSGN['Time'], label='PSGN', linestyle='-.', linewidth=4.5)
    plt.legend(prop={'size':20})
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    plt.xlabel('Data Category', fontweight ='bold', fontsize=20)
    plt.ylabel('Computational Time (ms)', fontweight ='bold', fontsize=20)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig('/home/ahz/Desktop/My papers/1/PlotTimeResult.png')

    plt.show()


def showLossHistogram():
    # set width of bar
    barWidth = 0.15

    # Set position of bar on X axis
    br1 = np.arange(len(Ours_New['Loss']))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    # Make the plot
    plt.bar(br1, Ours_New['Loss'], color ='r', width = barWidth, label ='Ours', alpha=1)
    plt.bar(br2, Pixel2Point['Loss'], color ='b', width = barWidth, label ='Pixel2Point (w/o PC)', alpha=1)
    plt.bar(br3, PSGN['Loss'], color ='g', width = barWidth, label ='PSGN', alpha=1)
    plt.bar(br4, Pixel2Point_InitialPC['Loss'], color ='y', width = barWidth, label ='Pixel2Point (w PC)', alpha=1)

    # Adding Xticks
    plt.legend(prop={'size':20})
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    plt.xlabel('Data Category', fontweight ='bold', fontsize = 20)
    plt.ylabel('Chamfer Loss', fontweight ='bold', fontsize = 20)
    plt.xticks([r + 1.5*barWidth for r in range(len(Ours_New['Loss']))], Data_Categories)
    plt.grid(axis='y', linewidth=0.5)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig('/home/ahz/Desktop/My papers/1/HistogramLossResult.png')
    plt.show()


def showTimeHistogram():
    # set width of bar
    barWidth = 0.15

    # Set position of bar on X axis
    br1 = np.arange(len(Ours_New['Time']))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    # Make the plot
    plt.bar(br1, Ours_New['Time'], color ='r', width = barWidth, label ='Ours', alpha=1)
    plt.bar(br2, Pixel2Point['Time'], color ='b', width = barWidth, label ='Pixel2Point (w/o PC)', alpha=1)
    plt.bar(br3, Pixel2Point_InitialPC['Time'], color ='y', width = barWidth, label ='Pixel2Point (w PC)', alpha=1)
    plt.bar(br4, PSGN['Time'], color ='g', width = barWidth, label ='PSGN', alpha=1)    

    # Adding Xticks
    plt.legend(prop={'size':20})
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    plt.xlabel('Data Category', fontweight ='bold', fontsize = 20)
    plt.ylabel('Computational Time (ms)', fontweight ='bold', fontsize = 20)
    plt.xticks([r + 1.5*barWidth for r in range(len(Ours_New['Time']))], Data_Categories)
    plt.grid(axis='y', linewidth=0.5)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig('/home/ahz/Desktop/My papers/1/HistogramTimeResult.png')    
    plt.show()






###################### Compare our models to each other ##################

def showLossPlot_Ours():
    plt.plot(Data_Categories, Ours_New['Loss'], label='Ours (w/o Pool)', linestyle= '--', linewidth=4.5)    
    plt.plot(Data_Categories, ours['Loss'], label='Ours (w Pool)', linestyle= ':', linewidth=4.5)
    plt.legend(prop={'size':20})
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    plt.xlabel('Data Category', fontweight ='bold', fontsize=20)
    plt.ylabel('Chamfer Loss', fontweight ='bold', fontsize=20)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig('/home/ahz/Desktop/My papers/1/PlotLossResult_Ours.png')

    plt.show()

def showTimePlot_Ours():
    plt.plot(Data_Categories, Ours_New['Time'], label='Ours (w/o Pool)', linestyle= '--', linewidth=4.5)    
    plt.plot(Data_Categories, ours['Time'], label='Ours (w Pool)', linestyle= ':', linewidth=4.5)
    plt.legend(prop={'size':20})
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    plt.xlabel('Data Category', fontweight ='bold', fontsize=20)
    plt.ylabel('Computational Time (ms)', fontweight ='bold', fontsize=20)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig('/home/ahz/Desktop/My papers/1/PlotTimeResult_Ours.png')

    plt.show()

def showLossHistogram_Ours():
    # set width of bar
    barWidth = 0.15

    # Set position of bar on X axis
    br1 = np.arange(len(ours['Loss']))
    br2 = [x + barWidth for x in br1]

    # Make the plot
    plt.bar(br1, Ours_New['Loss'], color ='r', width = barWidth, label ='Ours (w/o Pool)', alpha=0.7)
    plt.bar(br2, ours['Loss'], color ='b', width = barWidth, label ='Ours (w Pool)', alpha=0.7)

    # Adding Xticks
    plt.legend(prop={'size':20})
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    plt.xlabel('Data Category', fontweight ='bold', fontsize = 20)
    plt.ylabel('Chamfer Loss', fontweight ='bold', fontsize = 20)
    plt.xticks([r + 1.5*barWidth for r in range(len(Ours_New['Loss']))], Data_Categories)
    plt.grid(axis='y', linewidth=0.5)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig('/home/ahz/Desktop/My papers/1/HistogramLossResult_Ours.png')

    plt.show()

def showTimeHistogram_Ours():
    # set width of bar
    barWidth = 0.15

    # Set position of bar on X axis
    br1 = np.arange(len(Ours_New['Time']))
    br2 = [x + barWidth for x in br1]

    # Make the plot
    plt.bar(br1, Ours_New['Time'], color ='r', width = barWidth, label ='Ours (w/o Pool)', alpha=0.7)
    plt.bar(br2, ours['Time'], color ='b', width = barWidth, label ='Ours (w Pool)', alpha=0.7)

    # Adding Xticks
    plt.legend(prop={'size':20})
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    plt.xlabel('Data Category', fontweight ='bold', fontsize = 20)
    plt.ylabel('Computational Time (ms)', fontweight ='bold', fontsize = 20)
    plt.xticks([r + 1.5*barWidth for r in range(len(Ours_New['Time']))], Data_Categories)
    plt.grid(axis='y', linewidth=0.5)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig('/home/ahz/Desktop/My papers/1/HistogramTimeResult_Ours.png')


    plt.show()


if __name__ == "__main__":
    # showLossPlot()
    # showTimePlot()
    showLossHistogram()
    showTimeHistogram()
    
    # showLossPlot_Ours()
    # showTimePlot_Ours()
    showLossHistogram_Ours()
    showTimeHistogram_Ours()

