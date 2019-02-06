import numpy as np
from core import *
from activations import *



class SimpleRNN(Layer):
    def __init__(self,isize,osize):
        self.wi=np.random.randn(isize, osize) * (0.1 / (isize+osize))
        self.wh=np.random.randn(osize, osize) * (0.1 / (isize+osize))
        self.bh=np.zeros(osize)

    def forward(self,X,level,ctx):
        h=ctx.getState(self,level,self.bh)
        ret=[]
        io=[]
        for i,x in enumerate(X):
            #h=tanh(wi*x+wh*h+bh)
            io.append((x,h))
            h=np.tanh(np.dot(x,self.wi)+np.dot(h,self.wh)+self.bh)
            ret.append(h)
        ctx.setState(self,level,h)
        ctx.putIo(self,level,io,ret)
        return ret
    
    def backward(self,err,level,ctx):
        #print self.wi.shape,self.wh.shape,self.bh.shape
        inp,out=ctx.getIo(self,level)
        ret=[]
        delta=0
        dbh=dwh=dwi=0
        for t in reversed(range(len(err))):
            #print t,err[t].shape,self.wi.shape,delta
            #error accumulation
            dh=err[t]+delta
            #print dh.shape,out[t].shape
            #derivative
            dh_raw=(1-np.power(out[t],2))*dh
            #constant error
            dbh+=dh_raw
            #hidden matrix error
            dwh+=np.outer(inp[t][1],dh_raw)
            #input matrix error
            dwi+=np.outer(inp[t][0],dh_raw)
            #delta back projection
            delta=np.dot(dh_raw,self.wh.T)
        ctx.putErr((self,0),dwi)
        ctx.putErr((self,1),dwh)
        ctx.putErr((self,2),dbh)
        return ret

    def update(self,ctx,lr,part):
        if part==0:
            self.wi+=ctx.getErr((self,0))*lr
        elif part==1:
            self.wh+=ctx.getErr((self,1))*lr
        elif part==2:
            self.bh+=ctx.getErr((self,2))*lr
        #print "update",part,ctx.getErr((self,part))*lr

    def printMe(self,full=False):
        self.printBrd("SimpleRNN"+str(self.wi.shape)+str(self.wh.shape)+str(self.bh.shape))

class TimeDistributed(Layer):
    def __init__(self,layer):
        self.layer=layer

    def forward(self,X,level,ctx):
        ret=[]
        for i,x in enumerate(X):
            k=str(level)+"."+str(i)
            ret.append(self.layer.forward(x,k,ctx))
        return np.array(ret)

    def backward(self,err,level,ctx):
        ret=[]
        for i,x in reversed(list(enumerate(err))):
            k=str(level)+"."+str(i)
            ret.append(self.layer.backward(x,k,ctx))
        ret.reverse()
        return np.array(ret)
    
    def printMe(self,full=False):
        self.layer.printMe(full)

def sigmoid(x):
    return np.float32(1.0) / (np.float32(1.0) + np.exp(-x))

def gru_forward(X,h,Wz,Uz,bz,Wr,Ur,br,Wh,Uh,bh):
    ret=[]
    io=[]
    for t in xrange(len(X)):
        z=sigmoid(np.dot(X[t],Wz)+np.dot(h,Uz)+bz)
        r=sigmoid(np.dot(X[t],Wr)+np.dot(h,Ur)+br)
        v=np.tanh(np.dot(X[t],Wh)+np.dot(r*h,Uh)+bh)
        hh=z*h + (1-z)*v
        io.append((X[t],h,z,r,v))
        ret.append(hh)
        h=hh
    return h,io,ret

def gru_backward(inp,out,err,Wz,Uz,bz,Wr,Ur,br,Wh,Uh,bh):
    ret=[]
    delta=np.zeros(err[0].shape,dtype=np.float32)
    #initialize delta sorage
    dwz=np.zeros_like(Wz)
    dwr=np.zeros_like(Wr)
    dwh=np.zeros_like(Wh)
    duz=np.zeros_like(Uz)
    dur=np.zeros_like(Ur)
    duh=np.zeros_like(Uh)
    dbz=np.zeros_like(bz)
    dbr=np.zeros_like(br)
    dbh=np.zeros_like(bh)
    for t in range(len(err)-1,-1,-1):
        xt,ht,zt,rt,vt=inp[t]
        #error accumulation
        dh=(err[t]+delta)#.astype(np.float32)
        #print dh.dtype
        #derivative
        dh_hat=dh*(np.float32(1)-zt)
        dh_hat_1=dh_hat*(np.float32(1)-np.power(vt,np.float32(2)))

        #calculate xh deltas
        dwh+=np.outer(xt,dh_hat_1)
        duh+=np.outer(rt*ht,dh_hat_1)
        dbh+=dh_hat_1

        #derive r
        drhp = np.dot(dh_hat_1,Uh.T)
        dr=drhp*ht
        dr_1=dr*rt*(np.float32(1)-rt)

        #calculate xr deltas
        dwr+=np.outer(xt,dr_1)
        dur+=np.outer(ht,dr_1)
        dbr+=dr_1

        #drive z
        dz=dh*(ht-vt)
        dz_1 = dz*zt*(np.float32(1)-zt)

        #calculate xz deltas
        dwz+=np.outer(xt,dz_1)
        duz+=np.outer(ht,dz_1)
        dbz+=dz_1

        dh_fz_inner = np.dot(dz_1,Uz.T)
        dh_fz = dh*zt
        dh_fhh = drhp*rt
        dh_fr = np.dot(dr_1,Ur.T)

        delta=dh_fz_inner + dh_fz + dh_fhh + dh_fr

        ret.append(delta)
    return ret,dwz,dwr,dwh,duz,dur,duh,dbz,dbr,dbh

class GRU(Layer):
    def __init__(self,isize,osize):
        #in matrices
        self.Wz=(np.random.randn(isize, osize) * (1.0 / (isize+osize))).astype(np.float32)
        self.Wr=(np.random.randn(isize, osize) * (1.0 / (isize+osize))).astype(np.float32)
        self.Wh=(np.random.randn(isize, osize) * (1.0 / (isize+osize))).astype(np.float32)
        #h matrices
        self.Uz=(np.random.randn(osize, osize) * (1.0 / (osize+osize))).astype(np.float32)
        self.Ur=(np.random.randn(osize, osize) * (1.0 / (osize+osize))).astype(np.float32)
        self.Uh=(np.random.randn(osize, osize) * (1.0 / (osize+osize))).astype(np.float32)
        #bases
        self.bz=np.zeros(osize,dtype=np.float32)
        self.br=np.zeros(osize,dtype=np.float32)
        self.bh=np.zeros(osize,dtype=np.float32)

    def printMe(self,full=False):
        shapes=[]
        for k,v in self.__dict__.items():
            if len(k)==2 and k[0] in "WUb" and hasattr(v,"shape"):
                shapes.append(v.shape)
        self.printBrd("GRU"+str(shapes))
    
    def forward(self,X,level,ctx):
        h=ctx.getState(self,level,self.bh)
        h,io,ret=gru_forward(X,h,self.Wz,self.Uz,self.bz,self.Wr,self.Ur,self.br,self.Wh,self.Uh,self.bh)
        ctx.setState(self,level,h)
        ctx.putIo(self,level,io,ret)
        return np.array(ret)

    def backward(self,err,level,ctx):
        #print self.wi.shape,self.wh.shape,self.bh.shape
        inp,out=ctx.getIo(self,level)
        ret,dwz,dwr,dwh,duz,dur,duh,dbz,dbr,dbh=gru_backward(inp,out,err,self.Wz,self.Uz,self.bz,self.Wr,self.Ur,self.br,self.Wh,self.Uh,self.bh)            
        #save base errors
        ctx.putErr((self,id(self.bh)),dbh)
        ctx.putErr((self,id(self.br)),dbr)
        ctx.putErr((self,id(self.bz)),dbz)
        #save input matrix errors
        ctx.putErr((self,id(self.Wh)),dwh)
        ctx.putErr((self,id(self.Wr)),dwr)
        ctx.putErr((self,id(self.Wz)),dwz)
        #save hidden matrix errors
        ctx.putErr((self,id(self.Uh)),duh)
        ctx.putErr((self,id(self.Ur)),dur)
        ctx.putErr((self,id(self.Uz)),duz)
        return np.array(ret)
        
    def update(self,ctx,lr,part):
        for k,v in self.__dict__.items():
            if id(v)==part:
                s0=np.sum(np.abs(v))
                v+=lr*ctx.getErr((self,part))
                #print "updated",k,s0,np.sum(np.abs(v))
        

