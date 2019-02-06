import numpy as np
from core import *


class Sequential(Model):
    def __init__(self):
        self.layers=[]

    def add(self,l):
        self.layers.append(l)


    def forward(self,X,level,ctx):
        for i,l in enumerate(self.layers):
            X=l.forward(X,str(level)+"."+str(i),ctx)
        return X

    def backward(self,err,level,ctx):
        for i,l in reversed(list(enumerate(self.layers))):
            err=l.backward(err,str(level)+"."+str(i),ctx)
        return err

    def predict(self,X):
        return self.forward(X,"",Ctx(False))

    def fit(self,X,Y,batch_size=32,epochs=1,loss=std_loss,lr=0.1,print_stat=print_stat,callbacks=[]):
        for epoch in xrange(epochs):
            batches=mini_batches(X,Y,batch_size)
            lo=0
            for data in batches:
                ctx=Ctx()
                for i,(bx,by) in enumerate(data):
                    r=self.forward(bx,str(id(self)),ctx)
                    l,e,flag=loss(by,r)
                    #print e
                    if not flag:
                        self.backward(e,str(id(self)),ctx)
                    lo+=l
                ctx.doUpdate(lr)
            lo=lo/len(X)
            print_stat(epoch,lo)
    def printMe(self,full=False):
        for l in self.layers:
            l.printMe(full)
            
