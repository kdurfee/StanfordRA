import matplotlib.pyplot as plt
import numpy as np
import argparse
from enum import Enum
import CNN

class style(Enum):
    A=0 #Store activations within cores, pass weights systolically (and also activations .... dumb just for sanity)
    B=1 #divide activations spatially across cores by W, broadcast all filter weights
    C=2 #Block across K, replicating W across cores
    D=3 #Store filter weights within a core and pass activation systolically
    E=4 #Block across K between cores, broadcasting full activation
    F=5 #Block across W, replicating K across cores
    G=6 #Store activations blocked across W, broadcast filter weights per block
    H=7 #Store weights blocked across K, broadcast blocked activations

#Global Variables
#To start just look at a simple example of 8 cores 4x2 Cores each containing 4 PEs 2x2
Cores=1.0*8
CorePE=1.0*4
#PEBlock=1.0*4 
CoreBlocks=1.0*4 #split across columns in this case so 4
CoresPerBlock=1.0*2 #same as PEs per row, so 2
PE=Cores*CorePE
val_size = 2 #16 bit is 2 bytes


#this is just what we are holding for main computation
def ComputeLocalMem(Net,HWstyle):
    mem=[]
    for i in range(0,Net.layers):
        W=Net.W[i]
        C=Net.C[i]
        F=Net.F[i]
        K=Net.K[i]
        if HWstyle is style.A:
            if i==0:
                mem.append(((W*W)/(CorePE)) *C * val_size)
            else:
                mem.append(0)
        elif HWstyle is style.B:
            mem.append(((W*W)/(PE)) *C * val_size)
        elif HWstyle is style.C:
            mem.append ((W*W)/(Cores) * C * val_size)
        elif HWstyle is style.D:
            mem.append((K/(CorePE)) * F*F*C * val_size)
        elif HWstyle is style.E:
            mem.append((K/(PE)) * F*F*C * val_size)        
        elif HWstyle is style.F:
            mem.append((K/(Cores)) * F*F*C * val_size)
        elif HWstyle is style.G:
            mem.append((W*W)/(Cores/CoreBlocks) * C * val_size)
        elif HWstyle is style.H:
            mem.append((K/(Cores/CoreBlocks)) * F*F*C * val_size)
                           
    return mem

def ComputeInputBW(Net,HWstyle):
    mem=[]
    for i in range(0,Net.layers):
        W=Net.W[i]
        C=Net.C[i]
        F=Net.F[i]
        K=Net.K[i]
        if HWstyle is style.A:#incoming weights, or weights and activation after layer 0
            if i==0:
                mem.append(((F*F*C*K))* val_size)
            else:
                mem.append(((F*F*C*K))* val_size + (W*W*C)*val_size)
        elif HWstyle is style.B:#incoming weights broadcast
            mem.append((F*F*C*K)* val_size)
        elif HWstyle is style.C:#incoming weights blocked
            mem.append((F*F*C*K)/(Cores)* val_size)
        elif HWstyle is style.D:#incoming activations
            mem.append((W*W*C)* val_size)
        elif HWstyle is style.E:#incoming activatiions broadcast
            mem.append((W*W*C)* val_size)
        elif HWstyle is style.F:#incoming activatiion blocked
            mem.append((W*W*C)/(Cores)* val_size)
        elif HWstyle is style.G:
            mem.append((F*F*C*K)/(Cores/CoresPerBlock) * val_size)
        elif HWstyle is style.H:
            mem.append((W*W*C)/(Cores/CoresPerBlock) * val_size)
    return mem

#This is the BW between PEs for accumulations
def ComputePEBW(Net,HWstyle):
    mem=[]
    for i in range(0,Net.layers):
        W=Net.W[i]
        C=Net.C[i]
        F=Net.F[i]
        K=Net.K[i]
        if HWstyle is style.A:#Accumulation over W dimension within cores
            mem.append(((W*W)/(CorePE))*K * val_size)
        elif HWstyle is style.B:#accumulation over W within cores
            mem.append(((W*W)/(PE))*K * val_size)            
        elif HWstyle is style.C:#accumulation over W within cores
            mem.append(((W*W)/(CorePE))*K/Cores * val_size)
        elif HWstyle is style.D:#Accumulation across K within cores
            mem.append((W*W) * (K/(CorePE)) * val_size)
        elif HWstyle is style.E:#Sub Accumulation across K within cores
            mem.append(((W*W))*(K/PE)* val_size)
        elif HWstyle is style.F:#Accumulation over K within cores
            mem.append(((W*W)/(Cores)) * (K/CorePE) * val_size)
        elif HWstyle is style.G:#sub accumulation over blocked W within
            mem.append((W*W)/(PE/CoreBlocks) *K/(CoresPerBlock)*val_size)
        elif HWstyle is style.H:#sub accumulation over blocked K within
            mem.append((W*W)/(CoresPerBlock) * K/(PE/CoreBlocks)*val_size)
    return mem

#This is the BW between Cores for accumulations
def ComputeCoreBW(Net,HWstyle):
    mem=[]
    for i in range(0,Net.layers):
        W=Net.W[i]
        C=Net.C[i]
        F=Net.F[i]
        K=Net.K[i]
        if HWstyle is style.A:#none
            mem.append(0)
        elif HWstyle is style.B:#Broader accumulation over W across cores
            mem.append((((W*W)/Cores))*K* val_size)
        elif HWstyle is style.C:#accumulation over K across cores
            mem.append((W*W)*(K/Cores)*val_size)
        elif HWstyle is style.D:#None
            mem.append(0)
        elif HWstyle is style.E:#Broader accumulation across K across cores
            mem.append(((W*W))*(K/Cores)* val_size)
        elif HWstyle is style.F:#Accumulation across W across cores
            mem.append(((W*W)/(Cores)) * (K)* val_size)
        elif HWstyle is style.G:#over W and K
            mem.append((W*W)/(CoreBlocks) * K/(CoresPerBlock) * val_size)
        elif HWstyle is style.H:#over W and K
            mem.append((W*W)/(CoresPerBlock) * K/(CoreBlocks) * val_size)
    return mem

def ShowPlot(x,y,name):
    plt.figure()
    plt.title(name)
    y_pos = np.arange(len(x))    
    plt.bar(y_pos,y,align='center',alpha=.5)
    plt.xticks(y_pos,x)
    plt.draw()
    return

parser = argparse.ArgumentParser()
parser.add_argument("-s","--style", help= "Style of architecture to plot results for. if not specified, plots max for all styles",type=int)
parser.add_argument("-m","--max", help= "Plot the max of all layers per value",action="store_true")
args = parser.parse_args()

#create Alex Net CNN Model
Net = CNN.CNN(W=224,C=3,K=96,F=11,S=4,P=2,PF=3,PS=2,name='AlexNet')
Net.AddLayer(K=256,F=5,S=1,P=1,PF=3,PS=2)
Net.AddLayer(K=384,F=3,S=1,P=1)#1 is not max pooling
Net.AddLayer(K=384,F=3,S=1,P=1)#1 is not max pooling
Net.AddLayer(K=256,F=3,S=1,P=1,PF=3,PS=2)
Net.PrintLayerInfo()

layerNum = []
for x in range(0,Net.layers):
    layerNum.append(x)
    
mem=[]
inputBW=[]
peBW=[]
coreBW=[]
for s in style:
    mem.append(ComputeLocalMem(Net,s))
    inputBW.append(ComputeInputBW(Net,s))
    peBW.append(ComputePEBW(Net,s))
    coreBW.append(ComputeCoreBW(Net,s))

if args.style!=None:
    ShowPlot(layerNum,mem[args.style],'Local Mem (B) per layer for style' + str(args.style))
    ShowPlot(layerNum,inputBW[args.style],'Input BW required (B) per layer for style '+ str(args.style))
    ShowPlot(layerNum,peBW[args.style],'intra PE accum BW required (B) per layer for style '+ str(args.style))
    ShowPlot(layerNum,coreBW[args.style],'Intra Core accum BW required (B) per layer for style '+ str(args.style))


if args.max:
    maxMem=[]
    maxInputBW=[]
    maxPEBW=[]
    maxCoreBW=[]
    StyleID=[]
    for x in range(0,len(mem)):
        maxMem.append(max(mem[x]))
        maxInputBW.append(max(inputBW[x]))
        maxPEBW.append(max(peBW[x]))
        maxCoreBW.append(max(coreBW[x]))
        StyleID.append(x)
    ShowPlot(StyleID,maxMem,'maximum PE local memory (B) per style')
    ShowPlot(StyleID,maxInputBW,'maximum PE input BW (B) per style')
    ShowPlot(StyleID,maxPEBW,'maximum PE accum BW (B) per style')
    ShowPlot(StyleID,maxCoreBW,'maximum Core accum BW (B) per style')        

plt.show()



