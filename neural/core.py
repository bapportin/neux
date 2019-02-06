import numpy as np
import random
import ctc

class Ctx(object):
    def __init__(self,storeData=True):
        self.storeData=storeData
        self.data={}
        self.err={}
        self.states={}
    def putIo(self,layer,level,inp,out):
        if self.storeData:
            self.data[(layer,level)]=(inp,out)

    def getIo(self,layer,level):
        return self.data[(layer,level)]

    def putErr(self,layer,err):
        c,d=self.err.get(layer,(0,0))
        c+=1
        d+=err
        self.err[layer]=(c,d)

    def getErr(self,layer):
        c,d=self.err.get(layer,(1,0))
        return d/c

    def getState(self,layer,level,form):
        ret=self.states.get((layer,level))
        if ret is None:
            ret=np.zeros_like(form)
        return ret

    def setState(self,layer,level,state):
        self.states[(layer,level)]=state

    def doUpdate(self,lr):
        for l in self.err.keys():
            if type(l)==type(()):
                l[0].update(self,lr,l[1])
            else:
                l.update(self,lr)

class Layer(object):
    def forward(self,X,level,ctx):
        raise NotImplementedError
    def backward(self,err,level,ctx):
        raise NotImplementedError
    def update(self,ctx,lr):
        pass
    def printBrd(self,txt):
        l=100
        p0=(l-len(txt))/2
        p1=l-len(txt)-p0
        print ("-"*p0)+txt+("-"*p1)
    def printMe(self,full=False):
        print


class Activation(Layer):
    pass

def std_loss(y_true,y_pred):
    if type(y_true)==type([]):
        g=map(lambda y0,y1: y0-y1,y_true,y_pred)
    else:
        g=y_true-y_pred
    l=np.sum(np.abs(g))
    return l/len(g),g,((np.nan in g) or (np.inf in g))

def ctc_loss(y_true,y_pred):
    return ctc.ctc_loss(y_pred,y_true)

def mini_batches(X,y,batch_size):
    batches=[]
    for i in xrange(((len(X)-1)/batch_size)+1):
        batches.append([])
    for i,x in enumerate(X):
        ib=(i%len(batches))
        if ib==0:
            random.shuffle(batches)
        batches[ib].append((x,y[i]))
    return batches

def print_stat(epoch,loss):
    print "epoch: ",epoch,"loss: ",loss

class Model(Layer):
    def predict(self,X):
        raise NotImplementedError()
    def fit(self,X,Y,batch_size=32,epochs=1,loss=std_loss,lr=0.1,print_stat=print_stat,callbacks=[]):
        pass
        
    
class Base(Layer):
    def __init__(self,size):
        self.b=np.zeros(size,dtype=np.float32)

    def forward(self,X,level,ctx):
        ret=X+self.b
        ctx.putIo(self,level,X,ret)
        return ret
    
    def backward(self,err,level,ctx):
        ctx.putErr(self,err)
        return err

    def update(self,ctx,lr):
        self.b+=ctx.getErr(self)*lr
        #print "update Base",ctx.getErr(self)*lr

    def printMe(self,full=False):
        self.printBrd("Base"+str(self.b.shape))
        if full:
            print self.b
            self.printBrd("Base"+str(self.b.shape))
        
    
class Matrix(Layer):
    def __init__(self,isize,osize):
        self.W=(np.random.randn(isize, osize) * (0.1 / isize)).astype(np.float32)

    def forward(self,X,level,ctx):
        ret=np.dot(X,self.W)
        ctx.putIo(self,level,X,ret)
        return ret

    def backward(self,err,level,ctx):
        inp,out=ctx.getIo(self,level)
        dW=np.outer(inp,err)
        ret=np.dot(err,self.W.T)
        ctx.putErr(self,dW)
        return ret

    def update(self,ctx,lr):
        dW=ctx.getErr(self)
        #print dW
        #print "update Matrix",dW*lr
        self.W+=dW*lr
        
    def printMe(self,full=False):
        self.printBrd("Matrix"+str(self.W.shape))
        if full:
            print self.W
            self.printBrd("Matrix"+str(self.W.shape))


class Dense(Layer):
    def __init__(self,isize,osize,act="Tanh"):
        import activations
        self.w=Matrix(isize,osize)
        self.b=Base(osize)
        self.a=getattr(activations,act)()

    def forward(self,X,level,ctx):
        X1=self.w.forward(X,str(level)+".Dense.Matrix",ctx)
        X2=self.b.forward(X1,str(level)+".Dense.Base",ctx)
        return self.a.forward(X1,str(level)+".Dense.Activation",ctx)

    def backward(self,err,level,ctx):
        e1=self.a.backward(err,str(level)+".Dense.Activation",ctx)
        e2=self.b.backward(e1,str(level)+".Dense.Base",ctx)
        return self.w.backward(e2,str(level)+".Dense.Matrix",ctx)

    def printMe(self,full=False):
        self.printBrd("Dense("+str(self.w.W.shape)+";"+str(self.b.b.shape)+";"+str(self.a.__class__.__name__)+")")
        
