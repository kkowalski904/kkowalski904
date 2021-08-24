#Kevin Kowalski
#kkowalski904@gmail.com

'''Using Monte Carlo methods to approximate Pi'''

from random import uniform
import statistics

n = 1000000 #number of test points, larger n will increase accuracy but take O(n) longer to run

Points=[]
for i in range(n): #making n random points in the square with corners (-1,-1), (-1,1), (1,-1) and (1,1)
	x=uniform(-1,1)
	y=uniform(-1,1)
	Points.append([x,y])
	

InQuarterCircle=0
for i in Points: #finding what points fall within the circle of radius 1 and center (0,0)
	Dist_Squared=i[0]**2+i[1]**2
	if Dist_Squared <= 1:
		InQuarterCircle=InQuarterCircle+1
		
Pi=4*InQuarterCircle/n

print('Pi is approximately '+str(Pi))