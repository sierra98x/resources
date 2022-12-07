import csv

import matplotlib.pyplot as plt

exp1 = ('crossTransp-Results',[(12,'wd','WD & B','-'),(15,'wd','WD & B',':'),(18,'wd','WD & B','-.'),(20,'wd','WD & B','--')])
exp2 = ('constCostTransp-Results',[(12,'wd','WD & B','-'),(15,'wd','WD & B',':'),(18,'wd','WD & B','-.'),(20,'wd','WD & B','--')])
exp3 = ('vanishTransp-Results',[(12,'wd','WD','-'),(20,'wd','WD',':'),(12,'b','B','-.'),(20,'b','B','--')])
for (file, lines) in [exp1,exp2,exp3]:
    data = {'wd':{},'b':{}}
    with open(f"./Results/{file}.csv",'r') as f:
        
        reader = csv.reader(f)
        headers = next(reader)
        xValues = []

        for row in reader:
            
            n,m,s,wc,bc = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4])

            xValues.append(m)
            
            if n not in data['wd']:
                data['wd'][n] = {
                    'x': [],
                    'y': []
                }

            if n not in data['b']:
                data['b'][n] = {
                    'x': [],
                    'y': []
                }
            
            data['wd'][n]['x'].append(m)
            data['b'][n]['x'].append(m)
            data['wd'][n]['y'].append(wc/s if wc>0 else 0)
            data['b'][n]['y'].append(bc/s if bc>0 else 0)
    for n, metric, metricLabel, style in lines:

        plt.plot(data[metric][n]['x'],data[metric][n]['y'], label = f'{metricLabel}; n={n}', linestyle=style, marker='.')

    plt.xlabel('Segments')
    plt.ylabel('Ratio of reference instances with erratic pairs')
    plt.xticks([0]+xValues)
    plt.legend()
    plt.savefig(f'./Graphs/{file}-Graph.png',dpi=400)
    plt.cla()




