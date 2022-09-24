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

def train(model, data, target): #la training loop (iterate) 
    optim = torch.optim.Adam(model.parameters(), lr=1.0) #ADAM variante de SGD
    criterion = nn.CrossEntropyLoss()

    x=torch.FloatTensor(data) #convertir en tenseur 
    y=torch.LongTensor(target)
    
    L_acc=[]
    ep=[]
    L_acc2=[]
    idx = np.arange(100)
    for epoch in range(100): 
        
        np.random.shuffle(idx)
        tx=x[idx]
        ty=y[idx]#torch.tensor(y[idx])
        optim.zero_grad() #mettre a zero tt les gradients 
        haty = model(tx) #appeler forward on a les logits  
        loss = criterion(haty,ty) #on complète le graphe par la loss
        acc = test(model, tx ,ty) #data, target)
        print(str(loss.item())+" "+str(acc))
        L_acc2.append(loss.item())
        loss.backward() #la backpropagation 
        optim.step() #derniere etape modification des théta 
        #je shuffle ici
        
        
        L_acc.append(acc)
        #L_acc2.append(str(loss.item()))
        ep.append(epoch)
    plt.plot(ep, L_acc,L_acc2)
    #plt.plot(ep, L_acc2, 'r', label='loss')
    plt.xlabel('Epochs')
    plt.title('with lr=1.0')
    plt.show()

def genData(n,nsamps): #nins, nsamps): #juste generation des données selon deux gaussiènes 
    # prior0 = 0.7
    # mean0  = 0.3
    # var0   = 0.1
    # mean1  = 0.8
    # var1   = 0.01

    n0 = int(nsamps*0.8) #*prior0)
    # x0=var0 * np.random.randn(n0,nins) + mean0
    # x1=var1 * np.random.randn(nsamps-n0,nins) + mean1
    # x = np.concatenate((x0,x1), axis=0)
    # y = np.ones((nsamps,),dtype='int64')
    # y[:n0] = 0
    x =  np.random.uniform(size=(n,n)) #np.random.normal(3, 2.5, size=(n, n))
    y= np.random.uniform(size=(n,1))
   # e = np.random.normal(0.5, 0.8, size=x.shape)
    f = (x+y)**1/2 + np.log(x**2 + np.abs(y)) #np.square(x) + np.exp(y)  + 1  #+ e 
    y2 = np.ones((nsamps,),dtype='int64')
    y2[:n0] = 0 # optional
    print(f.shape)
    print(y2.shape)
    return f,y2
    #return x,y

def toytest(): #expérience
    print('start')
    model = Net(100,10)
    x,y=genData(100, 500)
    print(genData(100, 500))
  
    train(model,x,y)

if __name__ == "__main__":
    toytest()

    #Run it and plot the curves for the loss and for the accuracy
    #Adapt the code to create a test corpus with the same method as for the training corpus and evaluate your model

