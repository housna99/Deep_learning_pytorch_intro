import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
class Net(nn.Module):
    def __init__(self,nins,nout): #definir les parametres, cad les couches. ici on va faire 2 couches linéaires 
        super(Net, self).__init__()
        self.nins=nins
        self.nout=nout
        nhid = int((nins+nout)/2)
        self.hidden = nn.Linear(nins, nhid) #1ere couche cachée
        self.out    = nn.Linear(nhid, nout) #2eme couche cachée

    def forward(self, x): #comment l'info se propage #definir les fcts qui creent le graphe G
        x = torch.tanh(self.hidden(x)) #tanh est une activation linéaire
        x = self.out(x)
        return x

def test(model, data, target):
    x=torch.FloatTensor(data)
    y=model(x).data.numpy()
    haty = np.argmax(y,axis=1)
    nok=sum([1 for i in range(len(target)) if target[i]==haty[i]])
    acc=float(nok)/float(len(target))

    return acc

def train(model, data, target, nEpoch, learningRate): #la training loop (iterate) 
    #optim = torch.optim.Adam(model.parameters(), lr=learningRate) #ADAM variante de SGD
    optim=torch.optim.SGD(model.parameters(), lr=learningRate)
    criterion = nn.CrossEntropyLoss()

    x=torch.FloatTensor(data) #convertir en tenseur 
    y=torch.LongTensor(target)
    
    accVector=[]
    lossVector=[]
    epochVector=[x for x in range(nEpoch)]
    nbIterBeforeMaxAcc = 0
    idx = np.arange(nEpoch)
    
    print("Loss    Accuracy")
    for epoch in range(nEpoch): 
        np.random.shuffle(idx)
        tx=x[idx]
        ty=y[idx]

        # forward
        haty = model(tx) #appeler forward on a les logits  
        loss = criterion(haty,ty) #on complète le graphe par la loss

        #determination et affichage accuracy
        acc = test(model, tx ,ty) #data, target
        print(f"{loss.item():.4f}  {acc}")

        #recupération de la loss et de l'accuracy pour affichage
        lossVector.append(loss.item())
        accVector.append(acc)

        #backward
        optim.zero_grad() #mettre a zero tous les gradients 
        loss.backward() #la backpropagation 
        optim.step() #derniere etape modification des théta

        if acc < 1.0:
            nbIterBeforeMaxAcc += 1
    
     
    return epochVector, accVector, lossVector, nbIterBeforeMaxAcc


def genData(n, nsamps):  #generation des données selon deux gaussiennes 
    n0 = int(nsamps*0.8)
    x =  np.random.uniform(size=(n,n))
    y= np.random.uniform(size=(n,n))
  
    f = np.log(x+y) #definition de la fonction f
    y2 = np.ones((nsamps,),dtype='int64')
    y2[:n0] = 0 # optional

    return f, y2


def toytest(): #expérience
    lrs = [0.001, 0.01, 0.1, 1.0]
    nEpoch = 100
    runTimes = []
    nbEpochBeforeMaxAcc = []
    nbOfExperiences = 2

    x,y=genData(100, 500)
    for i in range(nbOfExperiences):
        print(f"Début de l'expérience {i+1}...")
        model = Net(100,10)
        
        print("Données pour l'expérience:")
        print(f"x:   {x} \ny:   {y}")
        start = time.time()
        epochVector, accVector, lossVector, nbIter = train(model, x, y, nEpoch, lrs[i])
        end = time.time()
        runTimes.append(end - start)
        nbEpochBeforeMaxAcc.append(nbIter)
        print("Temps d'apprentissage d'environ: %f s" % runTimes[-1])
        print(f"Nombre d'epochs pour avoir une précision de 100%: {nbIter}")

        #traçage des graphes
        plt.plot(epochVector, accVector, label="Accuracy")
        plt.plot(epochVector, lossVector, label="Loss")
        plt.legend(loc="upper right")
        plt.xlabel('Epochs', fontsize=9)
        plt.title(f"Expérience réalisée avec un taux d'apprentissage de {lrs[i]}, optimizer : SGD", fontsize= 11)
        plt.show()
        print(f"Fin de l'expérience {i+1}.\n")
    plt.subplot(3, 1, 1)
    plt.plot([1, 2, 3, 4], runTimes, label="Temps d'apprentissage")
    plt.legend(loc="upper right", prop={'size': 9})

    plt.ylabel("Temps d'apprentissage en s", fontsize=8)
    plt.title("Évolution du temps d'apprentissage entre les expériences", fontsize=11)

    plt.subplot(3, 1, 3)
    plt.plot([1, 2, 3, 4], nbEpochBeforeMaxAcc, label=f"Nombre d'epochs avant {100}% de précision", color='r')
    plt.legend(loc="upper right", fontsize= 9)
    plt.xlabel("n° d'expérience", fontsize= 9)
    plt.ylabel("Nombre d'epochs", fontsize= 8)
    plt.title("Évolution du nombre d'epochs necessaire entre les expériences", fontsize= 11)
    plt.show()

if __name__ == "__main__":
    toytest()
