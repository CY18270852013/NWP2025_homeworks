#s5f
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


dt=0.004;dx=0.1;
x=10
#myu=u*dt/dx

m=np.arange(0,x+1,1)
n=np.arange(0,4000,1)
M,N=np.meshgrid(m,n)#km∆x-σn∆t

u=np.zeros((10000,x+1))

def NLDE(m,n,C,k,bc):
    u=np.zeros((n+1,m+1))
    x=np.arange(0,m+1,1)
    u[0,x]=np.sin(2*np.pi*x*dx)+C;

    if (k==1):#1#######
        if(bc==0):#cyclical u[:,-1]=u[:,10];u[:,11]=u[:,0];
            u[1,0]=0;u[1,10]=0;u[0,0]=0;u[0,10]=0;
            for j in range(1,m,1):
                u[1,j]=u[0,j]-dt/(dx*8)*((u[0,j+1]+u[0,j])*(u[0,j+1]+u[0,j])-(u[0,j]+u[0,j-1])*(u[0,j]+u[0,j-1]))

            for i in range(1,n-1,1):
                u[i+1,0]=0;u[i+1,10]=0;
                for j in range(1,m,1):
                    u[i+1,j]=u[i-1,j]-dt/(dx*4)*((u[i,j+1]+u[i,j])*(u[i,j+1]+u[i,j])-(u[i,j]+u[i,j-1])*(u[i,j]+u[i,j-1]))

            u[:,0]=C;u[:,10]=C;

        else:
            # u[1,0]=u[0,0]-dt/(dx*8)*((u[0,1]+u[0,0])*(u[0,1]+u[0,0])-(u[0,0]+u[0,10])*(u[0,0]+u[0,10]))
            # u[1,10]=u[0,10]-dt/(dx*8)*((u[0,0]+u[0,10])*(u[0,0]+u[0,10])-(u[0,10]+u[0,9])*(u[0,10]+u[0,9]))

            for j in range(-1,m,1):
                u[1,j]=u[0,j]-dt/(dx*8)*((u[0,j+1]+u[0,j])*(u[0,j+1]+u[0,j])-(u[0,j]+u[0,j-1])*(u[0,j]+u[0,j-1]))

            for i in range(1,n-1,1):

                # u[i+1,0]=u[i-1,0]-dt/(dx*4)*((u[i,1]+u[i,0])*(u[i,1]+u[i,0])-(u[i,0]+u[i,10])*(u[i,0]+u[i,10]))
                # u[i+1,10]=u[i-1,10]-dt/(dx*4)*((u[i,0]+u[i,10])*(u[i,0]+u[i,10])-(u[i,10]+u[i,9])*(u[i,10]+u[i,9]))

                for j in range(-1,m,1):
                    u[i+1,j]=u[i-1,j]-dt/(dx*4)*((u[i,j+1]+u[i,j])*(u[i,j+1]+u[i,j])-(u[i,j]+u[i,j-1])*(u[i,j]+u[i,j-1]))

    if(k==2):#2
        p=np.zeros((1,m+1))
        if(bc==0):
            u[:,0]=0;u[:,10]=0;
            for i in range(0,n-1,1):
                u[i+1,:]=u[i,:];
                while(True):
                    for j in range(1,m,1):
                        A=((u[i+1,j+1]+u[i,j+1])+(u[i+1,j]+u[i,j])+(u[i+1,j-1]+u[i,j-1]))*((u[i+1,j+1]+u[i,j+1])-(u[i+1,j-1]+u[i,j-1]))/4
                        p[0,j]=np.abs(u[i,j]-dt/(dx*6)*A-u[i+1,j]);u[i+1,j]=u[i,j]-dt/(dx*6)*A

                    if (np.max(p)<1e-5):
                        break

            u[:,0]=C;u[:,10]=C;
        else:
            for i in range(0,n-1,1):
                u[i+1,:]=u[i,:];
                while(True):
                    A=((u[i+1,1]+u[i,1])+(u[i+1,0]+u[i,0])+(u[i+1,10]+u[i,10]))*((u[i+1,1]+u[i,1])-(u[i+1,10]+u[i,10]))/4
                    p[0,0]=np.abs(u[i,0]-dt/(dx*6)*A-u[i+1,0]);u[i+1,0]=u[i,0]-dt/(dx*6)*A
                    A=((u[i+1,0]+u[i,0])+(u[i+1,10]+u[i,10])+(u[i+1,10-1]+u[i,10-1]))*((u[i+1,0]+u[i,0])-(u[i+1,9]+u[i,9]))/4
                    p[0,10]=np.abs(u[i,10]-dt/(dx*6)*A-u[i+1,10]);u[i+1,10]=u[i,10]-dt/(dx*6)*A
                    for j in range(1,m,1):
                        A=((u[i+1,j+1]+u[i,j+1])+(u[i+1,j]+u[i,j])+(u[i+1,j-1]+u[i,j-1]))*((u[i+1,j+1]+u[i,j+1])-(u[i+1,j-1]+u[i,j-1]))/4
                        p[0,j]=np.abs(u[i,j]-dt/(dx*6)*A-u[i+1,j]);u[i+1,j]=u[i,j]-dt/(dx*6)*A

                    if (np.max(p)<1e-5):
                        break

    if(k==3):#3????
        if(bc==0):#########
            u[0:1,0]=0;u[0:1,10]=0;      
            for j in range(1,m,1):
                u[1,j]=u[0,j]-dt/(dx*2)*u[0,j]*(u[0,j+1]-u[0,j-1])

            for i in range(1,n-1,1):
                u[i+1,0]=0;u[i+1,10]=0;
                for j in range(1,m,1):
                    u[i+1,j]=u[i-1,j]-dt/(dx*1)*u[i,j]*(u[i,j+1]-u[i,j-1]) 

            u[:,0]=C;u[:,10]=C;

        else:
            u[1,0]=u[0,0]-dt/(dx*2)*u[0,0]*(u[0,1]-u[0,10])
            u[1,10]=u[0,10]-dt/(dx*2)*u[0,10]*(u[0,0]-u[0,9])

            for j in range(1,m,1):
                u[1,j]=u[0,j]-dt/(dx*2)*u[0,j]*(u[0,j+1]-u[0,j-1])

            for i in range(1,n-1,1):
                u[i+1,0]=u[i-1,0]-dt/(dx*1)*u[i,0]*(u[i,0+1]-u[i,10])
                u[i+1,10]=u[i-1,10]-dt/(dx*1)*u[i,10]*(u[i,0]-u[i,9])

                for j in range(1,m,1):
                    u[i+1,j]=u[i-1,j]-dt/(dx*1)*u[i,j]*(u[i,j+1]-u[i,j-1])            

    return u

def Ke(m,n,C,k,bc):
    u=NLDE(m,n,C,k,bc)

    K=np.zeros(n+1000)
    for i in range(0,n,1):
        for j in range(0,m+1,1):
            K[i]=K[i]+1/2*u[i,j]*u[i,j];
    
    return K




#question1 2
fig=plt.figure(num=1)

K1=Ke(10,800,0,1,2)
K2=Ke(10,800,0,2,2)
ax1=fig.add_subplot(221)
plt.plot(np.arange(0,800),K1[0:800],K2[0:800])
plt.grid(linestyle = '--')
plt.xlim([0,800])
plt.legend(["1","2"],loc='upper left')
plt.title("K-n cyclical sin+0")

K1=Ke(10,3000,0.5,1,1)
K2=Ke(10,3000,0.5,2,1)
ax2=fig.add_subplot(222)
plt.plot(np.arange(0,3000),K1[0:3000],K2[0:3000])
plt.grid(linestyle = '--')
plt.xlim([0,3000])
plt.legend(["1","2"],loc='upper left')
plt.title("K-n cyclical sin+0.5")

K1=Ke(10,3000,0.75,1,1)
K2=Ke(10,3000,0.75,2,1)
ax3=fig.add_subplot(223)
plt.plot(np.arange(0,3000),K1[0:3000],K2[0:3000])
plt.grid(linestyle = '--')
plt.xlim([0,3000])
plt.legend(["1","2"],loc='upper left')
plt.title("K-n cyclical sin+0.75")

K1=Ke(10,3000,1.0,1,1)
K2=Ke(10,3000,1.0,2,1)
ax4=fig.add_subplot(224)
plt.plot(np.arange(0,3000),K1[0:3000],K2[0:3000])
plt.grid(linestyle = '--')
plt.xlim([0,3000])
plt.legend(["1","2"],loc='upper left')
plt.title("K-n cyclical sin+1.0")


fig=plt.figure(num=2)

K1=Ke(10,3000,1.5,1,1)
K2=Ke(10,3000,1.5,2,1)
ax1=fig.add_subplot(221)
plt.plot(np.arange(0,3000),K1[0:3000],K2[0:3000])
plt.grid(linestyle = '--')
plt.xlim([0,3000])
plt.legend(["1","2"],loc='upper left')
plt.title("K-n cyclical sin+1.5")

K1=Ke(10,3000,3.0,1,1)
K2=Ke(10,3000,3.0,2,1)
ax2=fig.add_subplot(222)
plt.plot(np.arange(0,3000),K1[0:3000],K2[0:3000])
plt.grid(linestyle = '--')
plt.xlim([0,3000])
plt.legend(["1","2"],loc='upper left')
plt.title("K-n cyclical sin+3.0")

K1=Ke(10,3000,4.5,1,1)
K2=Ke(10,3000,4.5,2,1)
ax3=fig.add_subplot(223)
plt.plot(np.arange(0,3000),K1[0:3000],K2[0:3000])
plt.grid(linestyle = '--')
plt.xlim([0,3000])
plt.legend(["1","2"],loc='upper left')
plt.title("K-n cyclical sin+4.5")

K1=Ke(10,3000,6.0,1,1)
K2=Ke(10,3000,6.0,2,1)
ax4=fig.add_subplot(224)
plt.plot(np.arange(0,3000),K1[0:3000],K2[0:3000])
plt.grid(linestyle = '--')
plt.xlim([0,3000])
plt.legend(["1","2"],loc='upper left')
plt.title("K-n cyclical sin+6.0")


#question3
K1=Ke(10,600,0,3,1)
K2=Ke(10,600,0.5,3,1)
K3=Ke(10,600,0.75,3,1)
K4=Ke(10,600,1.0,3,1)
K5=Ke(10,600,1.5,3,1)
K6=Ke(10,600,3.0,3,1)
K7=Ke(10,600,4.5,3,1)
K8=Ke(10,600,6.0,3,1)

fig=plt.figure(num=3)

ax1=fig.add_subplot(221)
plt.plot(np.arange(0,490),K1[0:490])
plt.grid(linestyle = '--')
plt.xlim([0,500])
plt.legend(["sin+0"],loc='upper left')
plt.title("K-n cyclical format 3")
ax2=fig.add_subplot(222)
plt.plot(np.arange(0,250),K1[0:250],K2[0:250])
plt.grid(linestyle = '--')
plt.xlim([0,250])
plt.legend(["sin+0","sin+0.5"],loc='upper left')
plt.title("K-n cyclical format 3")
ax3=fig.add_subplot(223)
plt.plot(np.arange(0,250),K1[0:250],K3[0:250])
plt.grid(linestyle = '--')
plt.xlim([0,250])
plt.legend(["sin+0","sin+0.75"],loc='upper left')
plt.title("K-n cyclical format 3")
ax4=fig.add_subplot(224)
plt.plot(np.arange(0,250),K1[0:250],K4[0:250])
plt.grid(linestyle = '--')
plt.xlim([0,250]);plt.ylim([0,25]);
plt.legend(["sin+0","sin+1.0"],loc='upper left')
plt.title("K-n cyclical format 3")

fig=plt.figure(num=4)
ax1=fig.add_subplot(221)
plt.plot(np.arange(0,250),K1[0:250],K5[0:250])
plt.grid(linestyle = '--')
plt.xlim([0,250]);plt.ylim([0,25]);
plt.legend(["sin+0","sin+1.5"],loc='upper left')
plt.title("K-n cyclical format 3")
ax2=fig.add_subplot(222)
plt.plot(np.arange(0,250),K1[0:250],K6[0:250])
plt.grid(linestyle = '--')
plt.xlim([0,250]);plt.ylim([0,60]);
plt.legend(["sin+0","sin+3.0"],loc='center left')
plt.title("K-n cyclical format 3")
ax3=fig.add_subplot(223)
plt.plot(np.arange(0,250),K1[0:250],K7[0:250])
plt.grid(linestyle = '--')
plt.xlim([0,250]);plt.ylim([0,120]);
plt.legend(["sin+0","sin+4.5"],loc='center left')
plt.title("K-n cyclical format 3")
ax4=fig.add_subplot(224)
plt.plot(np.arange(0,250),K1[0:250],K8[0:250])
plt.grid(linestyle = '--')
plt.xlim([0,250]);plt.ylim([0,250]);
plt.legend(["sin+0","sin+6.0"],loc='center left')
plt.title("K-n cyclical format 3")


fig=plt.figure(num=5)

K1=Ke(10,1000,0,1,0)
K2=Ke(10,1000,0,2,0)
ax1=fig.add_subplot(221)
plt.plot(np.arange(0,1000),K1[0:1000],K2[0:1000])
plt.grid(linestyle = '--')
plt.xlim([0,1000]);plt.ylim([0,4])
plt.legend(["1","2"],loc='lower left')
plt.title("K-n rigid sin+0")

K1=Ke(10,1000,0.5,1,0)
K2=Ke(10,1000,0.5,2,0)
ax2=fig.add_subplot(222)
plt.plot(np.arange(0,1000),K1[0:1000],K2[0:1000])
plt.grid(linestyle = '--')
plt.xlim([0,1000]);plt.ylim([1,7])
plt.legend(["1","2"],loc='upper left')
plt.title("K-n rigid sin+0.5")

K1=Ke(10,1000,0.75,1,0)
K2=Ke(10,1000,0.75,2,0)
ax3=fig.add_subplot(223);plt.ylim([0,8])
plt.plot(np.arange(0,1000),K1[0:1000],K2[0:1000])
plt.grid(linestyle = '--')
plt.xlim([0,1000])
plt.legend(["1","2"],loc='lower left')
plt.title("K-n rigid sin+0.75")

K1=Ke(10,1000,1.0,1,0)
K2=Ke(10,1000,1.0,2,0)
ax4=fig.add_subplot(224)
plt.plot(np.arange(0,1000),K1[0:1000],K2[0:1000])
plt.grid(linestyle = '--')
plt.xlim([0,1000]);plt.ylim([0,12])
plt.legend(["1","2"],loc='lower left')
plt.title("K-n rigid sin+1.0")

fig=plt.figure(num=6)

K1=Ke(10,1000,1.5,1,0)
K2=Ke(10,1000,1.5,2,0)
ax1=fig.add_subplot(221)
plt.plot(np.arange(0,1000),K1[0:1000],K2[0:1000])
plt.grid(linestyle = '--')
plt.xlim([0,1000])
plt.legend(["1","2"],loc='upper left')
plt.title("K-n rigid sin+1.5")

K1=Ke(10,1000,2.0,1,0)
K2=Ke(10,1000,2.0,2,0)
ax2=fig.add_subplot(222)
plt.plot(np.arange(0,1000),K1[0:1000],K2[0:1000])
plt.grid(linestyle = '--')
plt.xlim([0,1000])
plt.legend(["1","2"],loc='upper left')
plt.title("K-n rigid sin+2.0")

K1=Ke(10,1000,2.5,1,0)
K2=Ke(10,1000,2.5,2,0)
ax3=fig.add_subplot(223)
plt.plot(np.arange(0,1000),K1[0:1000],K2[0:1000])
plt.grid(linestyle = '--')
plt.xlim([0,1000])
plt.legend(["1","2"],loc='upper left')
plt.title("K-n rigid sin+2.5")

K1=Ke(10,1000,3.0,1,0)
K2=Ke(10,1000,3.0,2,0)
ax4=fig.add_subplot(224)
plt.plot(np.arange(0,1000),K1[0:1000],K2[0:1000])
plt.grid(linestyle = '--')
plt.xlim([0,1000])
plt.legend(["1","2"],loc='upper left')
plt.title("K-n rigid sin+3.0")



#question3
K1=Ke(10,600,0,3,0)
K2=Ke(10,600,0.5,3,0)
K3=Ke(10,600,0.75,3,0)
K4=Ke(10,600,1.0,3,0)
K5=Ke(10,600,1.5,3,0)
K6=Ke(10,600,2.0,3,0)
K7=Ke(10,600,2.5,3,0)
K8=Ke(10,600,3.0,3,0)
fig=plt.figure(num=7)
ax1=fig.add_subplot(221)
plt.plot(np.arange(0,490),K1[0:490])
plt.grid(linestyle = '--')
plt.xlim([0,500])#;plt.ylim([0,250]);
plt.legend(["sin+0"],loc='center left')
plt.title("K-n rigid format 3")

ax2=fig.add_subplot(222)
plt.plot(np.arange(0,250),K1[0:250],K2[0:250])
plt.grid(linestyle = '--')
plt.xlim([0,250])
plt.legend(["sin+0","sin+0.5"],loc='center left')
plt.title("K-n rigid format 3")

ax3=fig.add_subplot(223)
plt.plot(np.arange(0,250),K1[0:250],K3[0:250])
plt.grid(linestyle = '--')
plt.xlim([0,250])
plt.legend(["sin+0","sin+0.75"],loc='center left')
plt.title("K-n rigid format 3")

ax4=fig.add_subplot(224)#
plt.plot(np.arange(0,250),K1[0:250],K4[0:250])
plt.grid(linestyle = '--')
plt.xlim([0,250])
plt.legend(["sin+0","sin+1.0"],loc='center left')
plt.title("K-n rigid format 3")


fig=plt.figure(num=8)
ax1=fig.add_subplot(221)
plt.plot(np.arange(0,250),K1[0:250],K5[0:250])
plt.grid(linestyle = '--')
plt.xlim([0,250]);plt.ylim([0,200]);
plt.legend(["sin+0","sin+1.5"],loc='center left')
plt.title("K-n rigid format 3")

ax2=fig.add_subplot(222)
plt.plot(np.arange(0,250),K1[0:250],K6[0:250])
plt.grid(linestyle = '--')
plt.xlim([0,250]);plt.ylim([0,300]);
plt.legend(["sin+0","sin+2.0"],loc='center left')
plt.title("K-n rigid format 3")

ax3=fig.add_subplot(223)
plt.plot(np.arange(0,250),K1[0:250],K7[0:250])
plt.grid(linestyle = '--')
plt.xlim([0,250]);plt.ylim([0,400]);
plt.legend(["sin+0","sin+2.5"],loc='center left')
plt.title("K-n rigid format 3")

ax4=fig.add_subplot(224)
plt.plot(np.arange(0,250),K1[0:250],K8[0:250])
plt.grid(linestyle = '--')
plt.xlim([0,250]);plt.ylim([0,600]);
plt.legend(["sin+0","sin+3.0"],loc='center left')
plt.title("K-n rigid format 3")


plt.show()