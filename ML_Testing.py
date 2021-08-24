#Kevin Kowalski
#kkowalski904@gmail.com

import numpy as np
import cvxopt as opt #quadratic program solver for kernel svm
from random import randrange, shuffle #random numbers for random forests, neural networks
import matplotlib.pyplot as plt #for plotting
import matplotlib.colors as color

'''Test of different classifiers used in machine learning for user selected data sets'''
'''Select points with left click and finalize points with a middle click or enter, then repeat for the 2nd class'''



'''Classifier and Hyperparameters'''

classifier='svm' #which classification algorithm to use with options below
#knn (k Nearest Neighbors)
#gnb (Gaussian Naive Bayes)
#svm (Linear Support Vector Machine)
#rbf (RBF Kernal Support Vector Machine)
#rnf (Random Forest)
#ann (artificial neural network)

k=3 #number of neighbors for knn
weighting=True #True for distance weighted knn, False for no knn weighting

c=1000 #weighting of svm hinge loss, larger c results in less regularization
nsteps=100000 #number of gradient descent steps for Adagrad svm optimization
a=1 #Adagrad step size adjustment

var=1 #rbf kernal variance
C='none' #kernal svm max alpha, 'none' for hard svm, number for soft svm

n_trees=100 #number of trees for random forest

n_layers=10 #number of hidden layers for neural network, keep >=1
npl=40 #number of neurons per hidden layer


'''Setting up Graph'''

plt.axis([0,10,0,10])



'''Creating 2 classes of points'''

data=[] #data labeled [x1,x2,y]

posdata=plt.ginput(-1,timeout=0) # y=1 class
px1=[]
px2=[]
for i in posdata:
	px1.append(i[0])
	px2.append(i[1])
	data.append([i[0],i[1],1])
plt.plot(px1,px2,'bo',markeredgecolor="black")

negdata=plt.ginput(-1,timeout=0) # y=-1 class
nx1=[]
nx2=[]
for i in negdata:
	nx1.append(i[0])
	nx2.append(i[1])
	data.append([i[0],i[1],-1])
plt.plot(nx1,nx2,'ro',markeredgecolor="black")
n=len(data)
pn=len(px1)
nn=len(nx1)


'''k Nearest Neighbors Classifier'''

if classifier == 'knn':
	def dist(x1,x2,dp):
		'''returns distance between chosen point (x1,x2) and a chosen data point dp'''
		return np.sqrt((x1-dp[0])**2 + (x2-dp[1])**2)
	
	def sdist(x1,x2,dp):
		'''returns signed distance between chosen point (x1,x2) and a chosen data point dp'''
		return (np.sqrt((x1-dp[0])**2 + (x2-dp[1])**2) * dp[2])
	
	#turn graph space into array of points
	x1=np.linspace(0,10,300)
	x2=np.linspace(0,10,300)
	
	def knnClass(x1,x2,data,k,weighting):
		'''finds the class based on the distance from the k nearest neighbors'''
		distList=[]
		for i in data:
			distList.append([dist(x1,x2,i),sdist(x1,x2,i)])
		ordered=sorted(distList)
		sum=0
		for i in range(0,k):
			if weighting == True:
				sum=sum+(1/(ordered[i][1])) #weighting by 1/distance so closer points matter more
			else:
				sum=sum+np.sign(ordered[i][1]) #takes most common class of the k nearest points
		if sum > 0: #average class is above 0, so predict y=1
			return 1
		elif sum < 0: #average class is below 0, so predict y=-1
			return -1
		else: #on decision boundary
			return 0
	
	z=[]
	for j in x2:
		zi=[]
		for i in x1:
			zi.append(knnClass(i,j,data,k,weighting))
		z.append(zi)
	
	colors = color.ListedColormap(["red", "black", "blue"])
	plt.pcolormesh(x1,x2,z, shading='nearest',cmap=colors)
	plt.show()
		
		
	
'''Gaussian Naive Bayes Classifier'''		
	
if classifier == 'gnb':
	def N(x,u,s):
		'''returns Gaussian with mean u and variance s'''
		denom=np.sqrt(2*np.pi*s)
		num=np.exp(-.5*((x-u)**2)*(1/s))
		return num/denom
		
	#turn graph space into array of points
	x1=np.linspace(0,10,300)
	x2=np.linspace(0,10,300)
	
	#average u and variance s for the x1 and x2 axes of the positive and negative data
	p1u=0
	p1s=0
	for i in px1:
		p1u=p1u+i/pn
	for i in px1:
		p1s=p1s+((i-p1u)**2)/pn
	
	p2u=0
	p2s=0
	for i in px2:
		p2u=p2u+i/pn
	for i in px2:
		p2s=p2s+((i-p2u)**2)/pn
	
	n1u=0
	n1s=0
	for i in nx1:
		n1u=n1u+i/nn
	for i in nx1:
		n1s=n1s+((i-n1u)**2)/nn
	
	n2u=0
	n2s=0
	for i in nx2:
		n2u=n2u+i/nn
	for i in nx2:
		n2s=n2s+((i-n2u)**2)/nn
	
	z=[]
	for j in x2:
		zi=[]
		for i in x1:
			Pp=N(i,p1u,p1s)*N(j,p2u,p2s)
			Pn=N(i,n1u,n1s)*N(j,n2u,n2s)
			if Pp>Pn:
				zi.append(1)
			elif Pp<Pn:
				zi.append(-1)
			else:
				zi.append(0)
		z.append(zi)
	

	
	colors = color.ListedColormap(["red", "black", "blue"])
	plt.pcolormesh(x1,x2,z, shading='nearest',cmap=colors)
	plt.show()



'''Linear Support Vector Machine'''

if classifier == 'svm':
	def cost(w1,w2,b,data,c):
		'''hinge loss cost function'''
		sum=0
		for i in data:
			sum=sum+max(0,1-i[2]*(w1*i[0]+w2*i[1]+b))
		return (w1**2 + w2**2 + c*sum)
	
	def grad(w1,w2,b,data):
		'''gradient of hinge loss for gradient descent optimization'''
		grad1=0 #w1 direction of gradient
		grad2=0 #w2 direction of gradient
		gradb=0 #b direction of gradient
		for i in data:
			hl= max(0,1-i[2]*(w1*i[0]+w2*i[1]+b))
			if  hl== 0:
				#if correctly classified, cost is only from the regularizer
				grad1=grad1+(w1/n)
				grad2=grad2+(w2/n)
			else:
				#if incorrectly classified, cost also comes from the classification error
				grad1=grad1+((w1-c*i[2]*i[0])/n)
				grad2=grad2+((w2-c*i[2]*i[1])/n)
				gradb=gradb-(c*i[2])
		return [grad1,grad2,gradb]
	
	#turn graph space into array of points
	x1=np.linspace(0,10,300)
	x2=np.linspace(0,10,300)
	
	#initialize svm with 0 weights
	w1=0
	w2=0
	b=0
	
	#save previous squared gradients over these for Adagrad optimization
	z1=0
	z2=0
	zb=0

	for i in range(0,nsteps):
		g=grad(w1,w2,b,data)
		
		#adjust z
		z1=z1+g[0]**2
		z2=z2+g[1]**2
		zb=zb+g[2]**2
		
		#adjust weights and bias
		w1=w1-a*((g[0])/(np.sqrt(z1+.0000001)))
		w2=w2-a*((g[1])/(np.sqrt(z2+.0000001)))
		b=b-a*((g[2])/(np.sqrt(zb+.0000001)))
	
	z=[]
	for j in x2:
		zi=[]
		for i in x1:
			if (w1*i+w2*j+b) > 0:
				zi.append(1)
			elif (w1*i+w2*j+b) < 0:
				zi.append(-1)
			elif(w1*i+w2*j+b) == 0:
				zi.append(0)
		z.append(zi)
	
	margin = np.sqrt(1/(w1**2+w2**2))
	angle = np.arctan(-w1/w2)
	sin=margin*np.sin(angle)
	cos=margin*np.cos(angle)
	
	colors = color.ListedColormap(["red", "black", "blue"])
	plt.pcolormesh(x1,x2,z, shading='nearest',cmap=colors)
	
	#decision boundary
	plt.axline((-b/w1, 0), (0, -b/w2),linewidth=4,color='black')
	
	#margin above
	plt.axline((-sin-b/w1, cos), (-sin, cos-b/w2),linewidth=1,linestyle='--',color='black')
	
	#margin below
	plt.axline((sin-b/w1, -cos), (sin, -cos-b/w2),linewidth=1,linestyle='--',color='black')
	
	plt.show()	



'''RBF Kernal Support Vector Machine'''

if classifier == 'rbf':	
	def kernel(p1,p2):
		'''finds rbf kernel element between the 2 points'''
		exponent=-1*(((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)/(var))
		return np.exp(exponent)

	km=[] #kernel matrix
	for i in data:
		ki=[]
		for j in data:
			ki.append(i[2]*j[2]*kernel(i,j))
		km.append(ki)
		
	y_vector=[]
	for i in data:
		y_vector.append(i[2])
	
	#set up the quadratic program solver
	#min 1/2 xTPx + qTx
	#subject to Gx <= h, Ax = b
	P=opt.matrix(np.outer(y_vector, y_vector)*km,tc='d')
	
	qb=[]
	for i in range(n):
		qb.append(-1)
	q=opt.matrix(qb,tc='d')
	
	if C=='none':
		Gb=[]
		for i in range(n):
			Gi=[]
			for j in range(n):
				if i==j:
					Gi.append(-1)
				else:
					Gi.append(0)
			Gb.append(Gi)
		G=opt.matrix(Gb,tc='d')

		hb=[]
		for i in range(n):
			hb.append(0)
		h=opt.matrix(hb,tc='d')

	if C!='none':
		G_max = np.identity(n) * -1
		G_min = np.identity(n)
		G = opt.matrix(np.vstack((G_max, G_min)))
		h_max = opt.matrix(np.zeros(n))
		h_min = opt.matrix(np.ones(n) *C)
		h = opt.matrix(np.vstack((h_max, h_min)))
	
	A=opt.matrix(y_vector, (1,n), tc='d')

	B=opt.matrix(0,tc='d')
	
	mini=opt.solvers.qp(P,q,G,h,A,B)
	alpha=np.ravel(mini['x'])
	
	#solve for b using the support vectors
	sv_index=[]
	for i in range(n):
		if alpha[i]>1e-10:
			sv_index.append(i)
			
	svn=len(sv_index)
	
	b_list=[]
	for j in sv_index:
		b_sum=0
		for i in range(n):
			b_sum=b_sum+alpha[i]*data[i][2]*kernel(data[i],data[j])
		b_list.append(data[j][2]-b_sum)	
		
	b=0
	for i in b_list:
		b=b+i/svn
		
	#turn graph space into array of points
	x1=np.linspace(0,10,200)
	x2=np.linspace(0,10,200)
	
	z=[]
	for j in x2:
		zi=[]
		for i in x1:
			hz=0
			for num in range(n):
				hz=hz+alpha[num]*data[num][2]*kernel([i,j],data[num])
			zi.append(np.sign(hz+b))
		z.append(zi)
	colors = color.ListedColormap(["red", "black", "blue"])
	plt.pcolormesh(x1,x2,z, shading='nearest',cmap=colors)

	plt.show()	



'''Random Forest'''

if classifier == 'rnf':
	def aveLabel(list):
		'''returns the average label of a list of labeled data points'''
		sum=0
		for i in list:
			sum=sum+i[2]/len(list)
		return sum
	
	def entropy(list):
		'''returns entropy of a list of labeled data points'''
		if len(list)==0:
			return 0
		sp=0
		sn=0
		for i in list:
			if i[2]==1:
				sp=sp+1
			else:
				sn=sn+1
		p1=sp/len(list) #ratio of +1 points to total points
		pn1=sn/len(list)
		if p1==1:
			return 0
		if pn1==1:
			return 0
		
		e1=-1*p1*np.log(p1) #entropy from +1 points
		en1=-1*pn1*np.log(pn1)
		return e1+en1
	
	def split(list,coord):
		'''returns the best x1 or x2 to split a list of labeled data to lower entropy
		and the list of points before and after that split'''
		co=0
		if coord==2:
			co=1
		sList=sorted(list, key=lambda x: x[co]) #sort by x1 or x2 value
		xs=0
		min_entropy=1
		L1_Final=[]
		L2_Final=[]
		for i in range(1,len(sList)):
			L1=[]
			L2=[]
			for j in sList:
				if j[co]>=sList[i][co]:
					L2.append(j)
				else:
					L1.append(j)
			ent=(len(L1)/len(list))*entropy(L1)+(len(L2)/len(list))*entropy(L2)
			if ent <= min_entropy:
				min_entropy=ent
				try:
					if sList[i][co] > sList[i-1][co]:
						xs=(sList[i][co]+sList[i-1][co])/2
					elif sList[i][co] > sList[i-2][co]:
						xs=(sList[i][co]+sList[i-2][co])/2
					elif sList[i][co] > sList[i-3][co]:
						xs=(sList[i][co]+sList[i-3][co])/2
					elif sList[i][co] > sList[i-4][co]:
						xs=(sList[i][co]+sList[i-4][co])/2
					else:
						xs=sList[i][co]
				except:
					xs=sList[i][co]
				L1_Final=L1
				L2_Final=L2
		return [xs,L1_Final,L2_Final]
	
	treeDataList=[] #List of datasets for each tree	
	for i in range(n_trees):
		treeData=[]
		for i in range(n):
			index=randrange(0,n)
			treeData.append(data[index])
		treeDataList.append(treeData)
		
	#make a tree for each dataset
	treeList=[]
	for i in treeDataList:
		treeEnd=False
		treeMax=100
		treeCount=0
		Branches=[i] #initial branch is the whole dataset
		ave=aveLabel(i)
		Splits=[[-1,11,-1,11,ave]] #bounds and label of each split [x1min,x1max,x2min,x2max,label]
		last_entropy=1
		while treeEnd == False:
			treeCount=treeCount+1
			if treeCount == treeMax:
				treeEnd=True
			newBranches=[]
			newSplits=[]
			for i in range(len(Branches)):
				if entropy(Branches[i]) !=0:
					sd=1+randrange(0,2) #split dimension
					sr=split(Branches[i],sd) #result of split function on the split dimension
					sc=sr[0] #right branch has sd coord >= sc
					
					Left=sr[1] #record information about left branch points and boundaries
					newBranches.append(Left)
					LeftSplit=Splits[i][:]
					LeftSplit[4]=aveLabel(Left)
					if sd==1:
						LeftSplit[1]=sc
					elif sd==2:
						LeftSplit[3]=sc
					newSplits.append(LeftSplit)
					
					Right=sr[2] #record information about right branch points and boundaries
					newBranches.append(Right)
					RightSplit=Splits[i][:]
					RightSplit[4]=aveLabel(Right)
					if sd==1:
						RightSplit[0]=sc
					elif sd==2:
						RightSplit[2]=sc
					newSplits.append(RightSplit)
					
				else: #branch already has only one label
					newBranches.append(Branches[i])
					newSplits.append(Splits[i])
				
			Branches=newBranches
			Splits=newSplits
			totalEntropy=0
			for i in Branches:
				totalEntropy=totalEntropy+entropy(i)
			if totalEntropy == 0:
				treeEnd=True
		treeList.append(Splits)
	
	def find_label(x1,x2,list):
		'''returns label of the branch that contains the point (x1,x2)'''
		for i in list:
			if x1 >= i[0]:
				if x1 < i[1]:
					if x2 >= i[2]:
						if x2 < i[3]:
							return i[4]
		
	def h_average(x1,x2,list):
		'''returns average class of the trees in the list for the point (x1,x2)'''
		sum=0
		for i in list:
			label=find_label(x1,x2,i)
			sum=sum+label/n_trees
		return sum
			
	#turn graph space into array of points
	x1=np.linspace(0,10,50)
	x2=np.linspace(0,10,50)
	z=[]
	for j in x2:
		zi=[]
		for i in x1:
			zi.append(h_average(i,j,treeList))
		z.append(zi)
	colors = color.ListedColormap(["red", "white", "blue"])
	plt.pcolormesh(x1,x2,z, shading='nearest',cmap='bwr_r')

	plt.show()	



'''Artificial Neural Network'''

if classifier == 'ann':
	def ReLU(xv):
		x=[]
		try:
			for i in xv:
				#x.append(i)
				x.append(max(0,i))
		except: #for row vector
			for i in xv[0]:
				x.append(max(0,i))
		return x
	
	def dReLU(xv):
		'''derivative of ReLU function'''
		try:
			x=[]
			for i in xv:
				if i>=0:
					x.append(1)
				else:
					x.append(0)
			return x
		except: #for row vector
			for i in xv[0]:
				if i>=0:
					x.append(1)
				else:
					x.append(0)
			return x
		
	def forward(Uv,xv,cv,lastlayer=False):
		'''returns output vector of layer with a given input xv, U matrix, and c vector'''
		prod=np.dot(xv,Uv)
		if lastlayer==True:
			yv=prod+cv
		else:
			yv=ReLU(prod+cv)
		return yv
		
	def backward(Uv,xv,cv,dLdOut,layer,lastlayer=False):
		'''returns dL/dIn and updates weights and bias for current layer'''
		if lastlayer==True:
			dLdIn=np.dot(dLdOut,np.transpose(Uv))
			dLdU=np.dot(np.transpose(xv),dLdOut)
			U[layer]=Uv-np.multiply(alpha,dLdU)
			c[layer]=cv-np.multiply(alpha,dLdOut)
			return dLdIn
		else:
			activ=dReLU(np.dot(xv,Uv)+cv)
			nonlin=np.multiply(activ,dLdOut)
			
			dLdIn=np.dot(nonlin,np.transpose(Uv))
			dLdU=np.dot(np.transpose(xv),nonlin)
			U[layer]=Uv-np.multiply(alpha,dLdU)
			c[layer]=cv-np.multiply(alpha,dLdOut)

		return dLdIn
	
	#initialize w,b,U's,c's randomly
	w=np.random.rand(npl,1)-.5
	b=np.random.rand(1,1)-.5
	U=[]
	c=[]
	for i in range(n_layers):
		if i==0:
			U.append(np.random.rand(2,npl)-.5)
		else:
			U.append(np.random.rand(npl,npl)-.5)
		c.append(np.random.rand(1,npl)-.5)
	U.append(w)
	c.append(b)

	#training
	n_epochs=500
	for j in range(n_epochs): #500 normally
		alpha=.01
		if j > .9*n_epochs:
			alpha=.000001
		elif j > .8*n_epochs:
			alpha=.00001
		elif j > .7*n_epochs:
			alpha=.0001
		elif j > .6*n_epochs:
			alpha=.001
		sdata=data[:]
		shuffle(sdata)
		for d in sdata:
			#rand_ind=randrange(0,n)
			#rp=data[rand_ind] #random point for sgd
			x0=[d[0]/10,d[1]/10]
			input=[]
			xc=x0[:]
			for i in range(n_layers+1): #forward propagation
				input.append([xc])
				if i==n_layers:
					yc=forward(U[i],xc,c[i],True)
				else:
					yc=forward(U[i],xc,c[i])
				xc=yc
			dOut=np.sign(yc[0][0])-d[2] #sgd
			for l in range(n_layers+1): #backward propagation
				index=-(l+1)
				if index==-1:
					dIn=backward(U[index],input[index],c[index],dOut,index,True)
					dOut=dIn
				else:
					dIn=backward(U[index],input[index],c[index],dOut,index)
					dOut=dIn

	#plotting

	#turn graph space into array of points
	x1=np.linspace(0,10,100)
	x2=np.linspace(0,10,100)
	z=[]
	for j in x2:
		zi=[]
		for i in x1:
			xt=[i/10,j/10]
			for k in range(n_layers+1):
				if k==(n_layers):
					yt=forward(U[k],xt,c[k],True)
				else:
					yt=forward(U[k],xt,c[k])
				xt=yt
			h=np.sign(yt[0][0])
			zi.append(h)
		z.append(zi)
	colors = color.ListedColormap(["red", "white", "blue"])
	plt.pcolormesh(x1,x2,z,shading='nearest',cmap=colors)

	plt.show()














