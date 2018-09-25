from sympy import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

#----------------------------------------E03--------------------------------------
x1=symbols('x1')
x2=symbols('x2')
y=symbols('y')
y=(1+(x1+x2-5)**2)*(1+(3*x1-2*x2)**2)

d1=diff(y,x1)
d2=diff(y,x2)

d11=diff(d1,x1)
d12=diff(d1,x2)
d21=diff(d2,x1)
d22=diff(d2,x2)

print(d1,d2,d11,d12,d21,d22)

def d1(x1,x2):
    return (18*x1 - 12*x2)*((x1 + x2 - 5)**2 + 1) + ((3*x1 - 2*x2)**2 + 1)*(2*x1 + 2*x2 - 10)

def d2(x1,x2):
    return (-12*x1 + 8*x2)*((x1 + x2 - 5)**2 + 1) + ((3*x1 - 2*x2)**2 + 1)*(2*x1 + 2*x2 - 10)

def d11(x1,x2):
    return  2*(3*x1 - 2*x2)**2 + 2*(18*x1 - 12*x2)*(2*x1 + 2*x2 - 10) + 18*(x1 + x2 - 5)**2 + 20

def d12(x1,x2):
    return (-12*x1 + 8*x2)*(2*x1 + 2*x2 - 10) + 2*(3*x1 - 2*x2)**2 + (18*x1 - 12*x2)*(2*x1 + 2*x2 - 10) - 12*(x1 + x2 - 5)**2 - 10

def d21(x1,x2):
    return (-12*x1 + 8*x2)*(2*x1 + 2*x2 - 10) + 2*(3*x1 - 2*x2)**2 + (18*x1 - 12*x2)*(2*x1 + 2*x2 - 10) - 12*(x1 + x2 - 5)**2 - 10

def d22(x1,x2):
    return 2*(-12*x1 + 8*x2)*(2*x1 + 2*x2 - 10) + 2*(3*x1 - 2*x2)**2 + 8*(x1 + x2 - 5)**2 + 10




#----------------------------------------E03-i--------------------------------------
x0=np.array([10,10]).reshape(2,1)
g0=np.array([[d1(10,10)],[d2(10,10)]])
a=np.array([[d11(10,10),d12(10,10)],[d21(10,10),d22(10,10)]])

x1=x0 - np.mat(a).I*g0
print(x1)

#----------------------------------------E03-ii--------------------------------------
x0=np.array([[2],[2]])
g0=np.array([[d1(2,2)],[d2(2,2)]])
a=np.array([[d11(2,2),d12(2,2)],[d21(2,2),d22(2,2)]])

x1=x0-np.mat(a).I*g0
print(x1)


#----------------------------------------E03-iii--------------------------------------



fig = plt.figure(1)
ax = fig.gca(projection='3d')

X = np.arange(9.5, 11, 0.25)
Y = np.arange(9.5, 11, 0.25)

X, Y = np.meshgrid(X, Y)
R = np.sqrt((1+(X+Y-5)**2)*(1+(3*X-2*Y)**2))

Z = np.sin(R)

surf = ax.plot_surface(X, Y, Z)

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,  linewidth=0, antialiased=False)


ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))


fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()



#----------------------------------------E06--------------------------------------

#----------------------------------------E06-i------------------------------------

p=np.array([[1,4],[1,5],[2,4],[2,5],[3,1],[3,2],[4,1],[4,2]])
t=np.array([0,0,0,0,1,1,1,1])

def hardlim(n):
    if n>=0:
        return 1
    if n<0:
        return 0

w = np.random.random(size=(1,2))
b = np.random.random(size=(1,1))
e=np.zeros([10,1])

epochs=10
#sse=np.zeros([1,10])
sse=[]

for epoch in range(epochs):

    for i in range(0,len(p)):
        n=np.dot(w,p[i].transpose())
        a=hardlim(n)
        e[i]=t[i]-a
        w=w+e[i]*p[i]
        b=b+e[i]
    print(sum(e*e))
    se=sum(e*e)

    sse.append(float(se))



print(w)
print(b)
print(sse)

#plot sum squar error

plt.figure(2)
plt.plot(sse)
plt.xlabel('epochs',)
plt.ylabel('sse')
plt.show()


#----------------------------------------E06-ii---------------------------------

for i in range(0,len(p)):
    a=hardlim(np.dot(w,p[i].transpose())+b)
    if a==t[i]:
        print ('true')
    else:
        print('flase')

#----------------------------------------E06-iii---------------------------------

# all outputs are 'true' from ii.
# which means errors are 0, so they're correct weight and bis.
# then let's see how the input and DB look like
p1 = np.arange(1, 5, 0.1)
p2 = (-float(b) - float(w[:, 0]) * p1) / float(w[:, 1])

plt.figure(3)
plt.plot(p1, p2)
plt.scatter(p[:4,0],p[:4,1],marker='x',color='b')
plt.scatter(p[4:,0],p[4:,1],marker='o',color='r')
plt.xlabel('p1',)
plt.ylabel('p2')
plt.show()







