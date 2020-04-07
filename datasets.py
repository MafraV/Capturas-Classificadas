import glob
import cv2
import csv
import os

images1 = glob.glob("C:/episodes/episode_1 - 1/*.png")
images2 = glob.glob("C:/episodes/episode_1 - 2/*.png")
images3 = glob.glob("C:/episodes/episode_1 - 3/*.png")

episode = [cv2.imread(img) for img in images1+images2+images3] 
names = [img for img in images1+images2+images3]
states = []

size = len(episode)
j=0
k=1
while (j<=size-10):
    path = 'C:/videos/Episode1/Ep1 - '+str(k)
    os.mkdir(path)
    y=j+10
    for i in range(j,y):
        string = os.path.basename(names[i])
        cv2.imwrite(path+'/'+string,episode[i])
    name = names[y-1]
    if (name[-5]=='n'): state = 'open'
    else: state = 'closed'
    states.append(state)
    j+=1
    k+=1
    
    
with open('C:/videos/Episode1/Ep1 - States.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Sequence Number", "Hand State"])
    tam = len(states)
    for i in range(0,tam):
        sequence = 'Ep.1 - '+str(i+1)
        writer.writerow([sequence, states[i]])             