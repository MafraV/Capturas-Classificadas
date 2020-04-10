import glob
import cv2
import csv
import os
import random

#Carregando os .csv das sequencias
data1 = csv.reader(open('C:/videos/Episode1/Ep1 - States.csv', 'r'), delimiter=",", quotechar='|')
data2 = csv.reader(open('C:/videos/Episode2/Ep2 - States.csv', 'r'), delimiter=",", quotechar='|')
data3 = csv.reader(open('C:/videos/Episode3/Ep3 - States.csv', 'r'), delimiter=",", quotechar='|')
data4 = csv.reader(open('C:/videos/Episode4/Ep4 - States.csv', 'r'), delimiter=",", quotechar='|')
data5 = csv.reader(open('C:/videos/Episode5/Ep5 - States.csv', 'r'), delimiter=",", quotechar='|')
data6 = csv.reader(open('C:/videos/Episode6/Ep6 - States.csv', 'r'), delimiter=",", quotechar='|')

#Salvando a coluna de estados em arrays
states1 = [row[1] for row in data1]
states2 = [row[1] for row in data2]
states3 = [row[1] for row in data3]
states4 = [row[1] for row in data4]
states5 = [row[1] for row in data5]
states6 = [row[1] for row in data6]

#Removendo o titulo da coluna
states1.remove(states1[0])
states2.remove(states2[0])
states3.remove(states1[0])
states4.remove(states2[0])
states5.remove(states1[0])
states6.remove(states2[0])

tam1 = len(states1)
tam2 = len(states2)
tam3 = len(states3)
tam4 = len(states4)
tam5 = len(states5)
tam6 = len(states6)

#Criando arrays com os caminhos das pastas
sequences1, sequences2 = [], []
sequences3, sequences4 = [], []
sequences5, sequences6 = [], []

for i in range(1,tam1+1):
    sequences1.append('C:/videos/Episode1/Ep1 - '+str(i))

for i in range(1,tam2+1):
    sequences2.append('C:/videos/Episode2/Ep2 - '+str(i))
    
for i in range(1,tam3+1):
    sequences3.append('C:/videos/Episode3/Ep3 - '+str(i))

for i in range(1,tam4+1):
    sequences4.append('C:/videos/Episode4/Ep4 - '+str(i)) 

for i in range(1,tam5+1):
    sequences5.append('C:/videos/Episode5/Ep5 - '+str(i))

for i in range(1,tam6+1):
    sequences6.append('C:/videos/Episode6/Ep6 - '+str(i))

#Randomizando os arrays
random.shuffle(sequences1)
random.shuffle(sequences2)
random.shuffle(sequences3)
random.shuffle(sequences4)
random.shuffle(sequences5)
random.shuffle(sequences6)
random.shuffle(sequences1)
random.shuffle(sequences2)
random.shuffle(sequences3)
random.shuffle(sequences4)
random.shuffle(sequences5)
random.shuffle(sequences6)
random.shuffle(sequences1)
random.shuffle(sequences2)
random.shuffle(sequences3)
random.shuffle(sequences4)
random.shuffle(sequences5)
random.shuffle(sequences6)

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

for i in range(0,tam3,2):
    seq = sequences3[i]
    if seq[-3]=='-':
        num = int(seq[-1])
    else:
        num = int(seq[-3]+seq[-2]+seq[-1])
    train.append(seq)
    statesTrain.append(states3[num-1])
    
    if (i+1)<tam3:
        seq = sequences3[i+1]
        if seq[-3]=='-':
            num = int(seq[-1])
        else:
            num = int(seq[-3]+seq[-2]+seq[-1])
        test.append(seq)
        statesTest.append(states3[num-1])
        
for i in range(0,tam4,2):
    seq = sequences4[i]
    if seq[-3]=='-':
        num = int(seq[-1])
    else:
        num = int(seq[-3]+seq[-2]+seq[-1])
    train.append(seq)
    statesTrain.append(states4[num-1])
    
    if (i+1)<tam4:
        seq = sequences4[i+1]
        if seq[-3]=='-':
            num = int(seq[-1])
        else:
            num = int(seq[-3]+seq[-2]+seq[-1])
        test.append(seq)
        statesTest.append(states4[num-1])
        
for i in range(0,tam5,2):
    seq = sequences5[i]
    if seq[-3]=='-':
        num = int(seq[-1])
    else:
        num = int(seq[-3]+seq[-2]+seq[-1])
    train.append(seq)
    statesTrain.append(states5[num-1])
    
    if (i+1)<tam5:
        seq = sequences5[i+1]
        if seq[-3]=='-':
            num = int(seq[-1])
        else:
            num = int(seq[-3]+seq[-2]+seq[-1])
        test.append(seq)
        statesTest.append(states5[num-1])

for i in range(0,tam6,2):
    seq = sequences6[i]
    if seq[-3]=='-':
        num = int(seq[-1])
    else:
        num = int(seq[-3]+seq[-2]+seq[-1])
    train.append(seq)
    statesTrain.append(states6[num-1])
    
    if (i+1)<tam6:
        seq = sequences6[i+1]
        if seq[-3]=='-':
            num = int(seq[-1])
        else:
            num = int(seq[-3]+seq[-2]+seq[-1])
        test.append(seq)
        statesTest.append(states6[num-1])

tamTrain = len(train)
tamTest = len(test)

#Salvando as fotos das sequencias no conjunto de treino
o=1
c=1
for i in range(0,tamTrain):
    direc = train[i]+'/*.pgm'
    if (statesTrain[i]=='open'): 
        path = 'C:/videos/Train/Open/TrainSet - Open - '+str(o)
        o+=1
    else:
        path = 'C:/videos/Train/Closed/TrainSet - Closed - '+str(c)
        c+=1
    direcs = glob.glob(direc)
    images = [cv2.imread(img,-1) for img in direcs] 
    names = [img for img in direcs]
    os.mkdir(path)
    for j in range(0,9):
        string = os.path.basename(names[j])
        cv2.imwrite(path+'/'+string,images[j])

#Salvando as fotos das sequencias no conjunto de teste
o=1
c=1
for i in range(0,tamTest):
    direc = test[i]+'/*.pgm'
    if (statesTest[i]=='open'): 
        path = 'C:/videos/Test/Open/TestSet - Open - '+str(o)
        o+=1
    else:
        path = 'C:/videos/Test/Closed/TestSet - Closed - '+str(c)
        c+=1
    direcs = glob.glob(direc)
    images = [cv2.imread(img,-1) for img in direcs] 
    names = [img for img in direcs]
    os.mkdir(path)
    for j in range(0,9):
        string = os.path.basename(names[j])
        cv2.imwrite(path+'/'+string,images[j])