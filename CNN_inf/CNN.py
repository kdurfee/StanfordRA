class CNN:
    def __init__(self,W=None,C=None,K=None,F=None,S=1,P=0,PF=1,PS=1,name='unnamed'):#first layer info and name
        self.layers = 1
        self.name = name
        self.W=[]
        self.C=[]
        self.K=[]
        self.F=[]
        self.S=[]
        self.P=[]
        self.PF=[]
        self.PS=[]
        self.W.append(W)
        self.C.append(C)
        self.K.append(K)
        self.F.append(F)
        self.S.append(S)
        self.P.append(P)
        self.PF.append(PF)
        self.PS.append(PS)

    def ComputeOutputSize(self,layer_index):
        out_act = self.W[layer_index] #start with the input activation size
        #if there is a stride and or padding
        #then we need to shrink
        out_act = ((out_act-self.F[layer_index]+(2*self.P[layer_index]))/self.S[layer_index]) + 1
        #if there is a pooling layer, then we need to shrink down accordingly
        out_act = ((out_act-self.PF[layer_index])/self.PS[layer_index]) + 1
        return out_act
        
        
    def AddLayer(self,K=None,F=None,S=1,P=0,PF=1,PS=1):
        #indices are 0 index
        prevLayer = self.layers-1
        newLayer = self.layers
        self.K.append(K)
        self.F.append(F)
        self.S.append(S)
        self.P.append(P)
        self.PF.append(PF)
        self.PS.append(PS)
        #Input activations are based on previous layer
        self.W.append(self.ComputeOutputSize(prevLayer))
        #Channels for layer is the K from previous layer
        self.C.append(self.K[prevLayer])
        self.layers = self.layers+1

    outAct=0
    def PrintLayerInfo(self):
        for x in range(0,self.layers):
            print "------------Layer {}----------".format(x)
            print "--Input Activations: {}x{}x{}".format(self.W[x],self.W[x],self.C[x])
            print "--Number of Filters {}".format(self.K[x])
            print "--Stride {}".format(self.S[x])
            print "--Pooling Size {} and stride {}".format(self.PF[x],self.PS[x])
            outAct=self.ComputeOutputSize(x)
            print "--Output Activations: {}x{}x{}".format(outAct,outAct,self.K[x])
            print " "
    
    
    

