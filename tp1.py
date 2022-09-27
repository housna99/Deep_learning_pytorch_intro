import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
class Net(nn.Module):
    def __init__(self,nins,nout): #definir les parametres, cad les couches la on va faire 2 couches linéaires 
        super(Net, self).__init__()
        self.nins=nins
        self.nout=nout
        nhid = int((nins+nout)/2)
        self.hidden = nn.Linear(nins, nhid)
        self.out    = nn.Linear(nhid, nout)

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
    optim = torch.optim.Adam(model.parameters(), lr=learningRate) #ADAM variante de SGD
    criterion = nn.CrossEntropyLoss()

    x=torch.FloatTensor(data) #convertir en tenseur 
    y=torch.LongTensor(target)
    
    accVector=[]
    lossVector=[]
    epochVector=[x for x in range(nEpoch)]
    idx = np.arange(nEpoch)
    print("Loss Accuracy")
    for epoch in range(nEpoch): 
        np.random.shuffle(idx)
        tx=x[idx]
        ty=y[idx]
        optim.zero_grad() #mettre a zero tous les gradients 
        haty = model(tx) #appeler forward on a les logits  
        loss = criterion(haty,ty) #on complète le graphe par la loss
        acc = test(model, tx ,ty) #data, target
        print(str(loss.item())+" "+str(acc))
        lossVector.append(loss.item())
        accVector.append(acc)
        loss.backward() #la backpropagation 
        optim.step() #derniere etape modification des théta
        
    #traçage des graphes
    plt.plot(epochVector, accVector, label="Accuracy")
    plt.plot(epochVector, lossVector, label="Loss")
    plt.legend(loc="upper right")
    plt.xlabel('Epochs')
    plt.title("Expérience réalisée avec un taux d'apprentissage de "+str(learningRate))
    plt.show()

def genData(n, nsamps):  #juste generation des données selon deux gaussiènes 
    n0 = int(nsamps*0.8) #*prior0)
    x =  np.random.uniform(size=(n,n))
    y= np.random.uniform(size=(n,1))
  
    f = np.log(x+y) #definition de la fonction f
    y2 = np.ones((nsamps,),dtype='int64')
    y2[:n0] = 0 # optional

    return f, y2


def toytest(): #expérience
    print("Début de l'expérience...")
    model = Net(100,10)
    x,y=genData(100, 500)
    nEpoch = 100
    lr=1.0
    print("Données pour l'expérience:\nx y")
    print(str(x)+" "+str(y))

    train(model, x, y, nEpoch, lr)

    print("Fin de l'expérience.")

if __name__ == "__main__":
    toytest()
