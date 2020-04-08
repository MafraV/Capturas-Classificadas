import glob
import cv2
import csv
import os
import random

#Carregando os CSVs da sequencias
data1 = csv.reader(open('C:/videos/Episode1/Ep1 - States.csv', 'r'), delimiter=",", quotechar='|')
data2 = csv.reader(open('C:/videos/Episode2/Ep2 - States.csv', 'r'), delimiter=",", quotechar='|')

#Salvando a coluna de estados em arrays
states1 = [row[1] for row in data1]
states2 = [row[1] for row in data2]

#Removendo o titulo da coluna
states1.remove(states1[0])
states2.remove(states2[0])

tam1 = len(states1)
tam2 = len(states2)

#Criando arrays com os caminhos das pastas
sequences1, sequences2 = [], []

k=1
for i in range(1,tam1+1):
    sequences1.append('C:/videos/Episode1/Ep1 - '+str(k))
    k+=1

k=1
for i in range(1,tam2+1):
    sequences2.append('C:/videos/Episode2/Ep2 - '+str(k))
    k+=1

#Randomizando os arrays
random.shuffle(sequences1)
random.shuffle(sequences2)
random.shuffle(sequences1)
random.shuffle(sequences2)
random.shuffle(sequences1)
random.shuffle(sequences2)

train, test = [], []
statesTrain, statesTest = [], []

#Adicionando a sequencia i ao conjunto de treino e a sequencia i+1 ao de teste
for i in range(0,tam1,2):
    seq = sequences1[i]
    if seq[-3]=='-':
        num = int(seq[-1])
    else:
        num = int(seq[-3]+seq[-2]+seq[-1])
    train.append(seq)
    statesTrain.append(states1[num-1])
    
    if (i+1)<tam1:
        seq = sequences1[i+1]
        if seq[-3]=='-':
            num = int(seq[-1])
        else:
            num = int(seq[-3]+seq[-2]+seq[-1])
        test.append(seq)
        statesTest.append(states1[num-1])

for i in range(0,tam2,2):
    seq = sequences2[i]
    if seq[-3]=='-':
        num = int(seq[-1])
    else:
        num = int(seq[-3]+seq[-2]+seq[-1])
    train.append(seq)
    statesTrain.append(states2[num-1])
    
    if (i+1)<tam2:
        seq = sequences2[i+1]
        if seq[-3]=='-':
            num = int(seq[-1])
        else:
            num = int(seq[-3]+seq[-2]+seq[-1])
        test.append(seq)
        statesTest.append(states2[num-1])

tamTrain = len(train)
tamTest = len(test)

#Salvando as fotos das sequencias no conjunto de treino
k=1
for i in range(0,tamTrain):
    direc = train[i]+'/*.png'
    path = 'C:/videos/Train/TrainSet - '+str(k)
    direcs = glob.glob(direc)
    images = [cv2.imread(img) for img in direcs] 
    names = [img for img in direcs]
    os.mkdir(path)
    for j in range(0,9):
        string = os.path.basename(names[j])
        cv2.imwrite(path+'/'+string,images[j])
    k+=1

#Criando o .csv dos estados das sequencias do conjunto de treino
with open('C:/videos/Train/TrainSet - States.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Sequence Number", "Hand State"])
    tam = len(statesTrain)
    for i in range(0,tam):
        sequence = str(i+1)
        writer.writerow([sequence, statesTrain[i]]) 

#Salvando as fotos das sequencias no conjunto de teste
k=1
for i in range(0,tamTest):
    direc = test[i]+'/*.png'
    path = 'C:/videos/Test/TestSet - '+str(k)
    direcs = glob.glob(direc)
    images = [cv2.imread(img) for img in direcs] 
    names = [img for img in direcs]
    os.mkdir(path)
    for j in range(0,9):
        string = os.path.basename(names[j])
        cv2.imwrite(path+'/'+string,images[j])
    k+=1

#Criando o .csv dos estados das sequencias do conjunto de teste
with open('C:/videos/Test/TestSet - States.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Sequence Number", "Hand State"])
    tam = len(statesTest)
    for i in range(0,tam):
        sequence = str(i+1)
        writer.writerow([sequence, statesTest[i]]) 