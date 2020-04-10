import glob
import cv2
import csv
import os

#Recebendo todos os caminhos q terminem em .pgm
images1 = glob.glob("C:/episodes/episode_1 - 1/*.pgm")
images2 = glob.glob("C:/episodes/episode_1 - 2/*.pgm")
images3 = glob.glob("C:/episodes/episode_1 - 3/*.pgm")

#Criando uma lista para as imagens e outra para seus respectivos nomes
episode = [cv2.imread(img,-1) for img in images1+images2+images3] 
names = [img for img in images1+images2+images3]
states = []

size = len(episode)
j=0
k=1
while (j<=size-10):
    path = 'C:/videos/Episode1/Ep1 - '+str(k) #criando o nome da pasta
    os.mkdir(path) #criando a pasta
    y=j+10
    for i in range(j,y): #salvando as fotos na pasta criada
        string = os.path.basename(names[i])
        cv2.imwrite(path+'/'+string,episode[i])
    name = names[y-1] #salvando o estado da mão referente a decima foto
    if (name[-5]=='n'): state = 'open'
    else: state = 'closed'
    states.append(state)
    j+=1
    k+=1

#Criando o .csv com os estados da mão de cada sequencia    
with open('C:/videos/Episode1/Ep1 - States.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Sequence Number", "Hand State"])
    tam = len(states)
    for i in range(0,tam):
        sequence = 'Ep.1 - '+str(i+1)
        writer.writerow([sequence, states[i]])             