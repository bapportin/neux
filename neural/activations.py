import numpy as np
from core import Layer
#activations

class Logabs(Layer):
    def forward(self,X,level,ctx):
        ret=np.sign(X)*np.log(np.abs(X)+1)
        ctx.putIo(self,level,X,ret)
        return ret
    def backward(self,err,level,ctx):
        inp,out=ctx.getIo(self,level)
        g=np.exp(-np.abs(out))
        ret=err*g
        return ret
    def printMe(self,full=False):
        self.printBrd("Activation(Logabs)")
        

class Tanh(Layer):
    def forward(self,X,level,ctx):
        ret=np.tanh(X)
        ctx.putIo(self,level,X,ret)
        return ret
    def backward(self,err,level,ctx):
        inp,out=ctx.getIo(self,level)
        g=np.float32(1)-np.power(out,2)
        return err*g
    def printMe(self,full=False):
        self.printBrd("Activation(Tanh)")


class Softmax(Layer):
    def forward(self, x,level,ctx):
        shiftx=x-np.max(x)
        exp = np.exp(shift)
        ret= exp / np.sum(exp, axis=1, keepdims=True)
        ctx.putIo(self,level,x,ret)
        return ret

    def backward(self,err,level,ctx):
        inp,out=ctx.getIo(self,level)
        g=out*(np.float32(1)-out)
        return err*h

    def printMe(self,full=False):
        self.printBrd("Activation(Softmax)")

class Sigmoid(Layer):
    def forward(self,x,level,ctx):
        ret=np.float32(1.0) / (np.float32(1.0) + np.exp(-x))
        ctx.putIo(self,level,x,ret)
        return ret

    def backward(self,err,level,ctx):
        inp,out=ctx.getIo(self,level)
        g=out*(np.float32(1)-out)
        return err*h

    def printMe(self,full=False):
        self.printBrd("Activation(Sigmoid)")


class Relu(Layer):
    def forward(self, x,level,ctx):
        ret=np.maximum(x, np.float32(0), x)
        ctx.putIo(self,level,x,ret)        
        return ret

    def backward(self,err,level,ctx):
        inp,out=ctx.getIo(self,level)
        g=np.float32(1)*(out>np.float32(0))
        return err*g

    def printMe(self,full=False):
        self.printBrd("Activation(Relu)")

        
class LeakyRelu(Layer):
    def forward(self, x,level,ctx):
        ret=np.maximum(x, 0.01 * x, x)
        ctx.putIo(self,level,x,ret)        
        return ret

    def backward(self,err,level,ctx):
        inp,out=ctx.getIo(self,level)
        g=0.01 + 0.99 * (out > 0)
        return err*g

    def printMe(self,full=False):
        self.printBrd("Activation(LeakyRelu)")
    
class Linear(Layer):
    def forward(self, x,level,ctx):
        ret=x
        ctx.putIo(self,level,x,ret)        
        return ret

    def backward(self,err,level,ctx):
        inp,out=ctx.getIo(self,level)    
        return err

    def printMe(self,full=False):
        self.printBrd("Activation(Linear)")
