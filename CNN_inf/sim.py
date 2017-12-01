import matplotlib.pyplot as plt
import numpy as np
import argparse
from enum import Enum
import CNN

class style(Enum):#TODO
    A=0 #divide activations spatially across all cores and PEs
    B=1 #divide activations spatially across cores and block across K inside cores (replicate HW inside cores)
    C=2 #divide activations spatially across cores and block across K across cores (replicate HW across cores)
    D=3 #divide K filters across all cores (replicate K in all PEs with in a core)
    E=4 #divide K filters across cores, blocking across K between cores
    F=5 #divide K filters across all cores, block across K within cores

#Global Variables
Cores=8
CorePE=16
PECol=4 
PERow=4
CoreCol=4 
CoreRow=2
PE=Cores*CorePE
val_size = 2 #16 bit is 2 bytes

def ComputeLocalMem(Net,HWstyle): #TODO this is just what we are holding, not counting any I/O buffer or anything like that
    mem=[]
    for i in range(0,Net.layers):
        W=Net.W[i]
        C=Net.C[i]
        F=Net.F[i]
        K=Net.K[i]
        if HWstyle is style.A:
            mem.append(((W*W)/(1.0*PE)) *C * val_size)
        elif HWstyle is style.B:
            mem.append( ((W*W)/(1.0* (Cores*CorePE/PECol))) * C * val_size)
        elif HWstyle is style.C:
            mem.append ((W*W)/(1.0*(Cores/CoreCol*CorePE)) * C * val_size)
        elif HWstyle is style.D:
            mem.append((K/(1.0*CorePE)) * F*F*C * val_size)
        elif HWstyle is style.E:
            mem.append((K/(1.0*(Cores/CoreCol*PECol))) * F*F*C * val_size)        
        elif HWstyle is style.F:
            mem.append((K/(1.0*(Cores*CorePE/PECol))) * F*F*C * val_size)
                           
    return mem

def ComputePEBW(Net,HWstyle):
    mem=[]
    for i in range(0,Net.layers):
        W=Net.W[i]
        C=Net.C[i]
        F=Net.F[i]
        K=Net.K[i]
        #BW is what is input TODO do we count accumulations / reductions here or seperately ?
        if HWstyle is style.A:#incoming weights
            mem.append(((F*F*C*K))* val_size)
        elif HWstyle is style.B:#incoming weights divided by PE block
            mem.append(((F*F*C*K)/PECol)* val_size)
        elif HWstyle is style.C:#incoming weights divided by Core Block
            mem.append(((F*F*C*K)/CoreCol)* val_size)
        elif HWstyle is style.D:#incoming activation 
            mem.append(((W*W*C))* val_size)
        elif HWstyle is style.E:#incoming activatiion
            mem.append(((W*W*C))* val_size)
        elif HWstyle is style.F:#incoming activatiion
            mem.append(((W*W*C))* val_size)
    return mem

def ComputeCoreBW(Net,HWstyle):
    mem=[]
    for i in range(0,Net.layers):
        W=Net.W[i]
        C=Net.C[i]
        F=Net.F[i]
        K=Net.K[i]
        #BW is what is input
        if HWstyle is style.A:#incoming weights (same for all cores)
            mem.append(((F*F*C*K))* val_size)
        elif HWstyle is style.B:#incoming weights
            mem.append(((F*F*C*K)/PECol)* val_size)
        elif HWstyle is style.C:#incoming weights divided by Core Block
            mem.append(((F*F*C*K)/CoreCol)* val_size)
        elif HWstyle is style.D:#incoming activation
            mem.append(((W*W*C))* val_size)
        elif HWstyle is style.E:#incoming activatiion
            mem.append(((W*W*C))* val_size)
        elif HWstyle is style.F:#incoming activatiion
            mem.append(((W*W*C))* val_size)
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
Net.AddLayer(K=384,F=3,S=1,P=1)#not max pooling
Net.AddLayer(K=384,F=3,S=1,P=1)#not max pooling
Net.AddLayer(K=256,F=3,S=1,P=1,PF=3,PS=2)
Net.PrintLayerInfo()

layerNum = []
for x in range(0,Net.layers):
    layerNum.append(x)
    
#Note, output of convolution, not looking at stride or padding is W * K
mem=[]
peBW=[]
coreBW=[]
for s in style:
    mem.append(ComputeLocalMem(Net,s))
    peBW.append(ComputePEBW(Net,s))
    coreBW.append(ComputeCoreBW(Net,s))

if args.style!=None:
    ShowPlot(layerNum,mem[args.style],'Local Mem (B) per layer for style' + str(args.style))
    ShowPlot(layerNum,peBW[args.style],'PE BW required (B) per layer for style '+ str(args.style))
    ShowPlot(layerNum,coreBW[args.style],'Core BW required (B) per layer for style '+ str(args.style))

maxMem=[]
maxPEBW=[]
maxCoreBW=[]
StyleID=[]
if args.max:
    for x in range(0,len(mem)):
        maxMem.append(max(mem[x]))
        maxPEBW.append(max(peBW[x]))
        maxCoreBW.append(max(coreBW[x]))
        StyleID.append(x)
    ShowPlot(StyleID,maxMem,'maximum local memory (B) per style')
    ShowPlot(StyleID,maxPEBW,'maximum PE BW (B) per style')
    ShowPlot(StyleID,maxCoreBW,'maximum Core BW (B) per style')             
        

plt.show()



