from math import *
from matplotlib.pyplot import *
import numpy.matlib 
from numpy import *
import scipy.special
from scipy.integrate import quad # Pour l'intégral
from random import *
from scipy.stats import norm # Pour la fonction de répartition de la loi normale
from mpl_toolkits.mplot3d import Axes3D

close()


# Question 3
def price1(N, rN, hN, bN, s, f):
    qN = (rN - bN)/(hN - bN)
    S = 0
    for k in range (0,N+1):
        S += f(s * pow(1 + bN, k) * pow(1 + hN, N-k)) * scipy.special.binom(N, k) * pow(qN, N-k) * pow(1 - qN, k) 
        
    p = S / pow(1 + rN, N)
    return p

# Question 4
def f3(x):
    return max(x - 100, 0)
print("Question 4")
print(price1(10, 0.02, 0.05, -0.05, 100, f3))


# Question 5

# Fonction renvoyant la matrice générée pour calculer price2
def price2_Matrice(N, rN, hN, bN, s, f):
    qN = (rN - bN)/(hN - bN)
    V = zeros((N+1,N+1))
    for k in range(0, N+1):
        V[N][k] = f(s * pow(1 + bN, N-k) * pow(1 + hN, k)) 
    for i in [N-1-k for k in range(N)]:
        for j in range(0, i+1):
            V[i][j] = (V[i+1][j] * (1-qN) + V[i+1][j+1] * qN) * 1/(rN+1)
    return V

def price2(N, rN, hN, bN, s, f):
    V = price2_Matrice(N, rN, hN, bN, s, f)
    return V[0][0]
    
    
# Question 6

def f5(x):
    return max(x-95, 0)

print("\nQuestion 6")
print(price2(3, 0.02, 0.05, -0.05, 100, f5))    

# Dessin de l'arbre avec les valeurs de v_k pour chaque noeud de l'arbre
figure6 = figure(num="Question 6 - price2")

M = price2_Matrice(3, 0.02, 0.05, -0.05, 100, f5)
for i in range(len(M)-1):
    for j in range(0,i+1):
        plot([i,i+1],[M[i][j],M[i+1][j]],'b' , linewidth=1)
        plot([i,i+1],[M[i][j],M[i+1][j+1]],'b' , linewidth=1)

title("v_k(.)")
xlabel("i")
ylabel("v_k(.)")
grid()
show()
    
    
# Question 7
print("\nQuestion 7")
N = randint(5, 15)
print("price1 = ", price1(N, 0.03, 0.05, -0.05, 100, f3))    
print("price2 = ", price2(N, 0.03, 0.05, -0.05, 100, f3))    

    
# Question 12

# Require : n>0
def price3(n, s, r, sigma, T, f):
    X = [gauss(0,1) for i in range(n)] # Génération d'un échantillon aléatoire suivant une loi normale centrée réduite.
    S = 0
    for k in range(n):
        S += exp(-r*T) * f( s*exp( (r - pow(sigma,2)/2)/T + sigma*sqrt(T) * X[k] ))
    return S/n
    

# Question 13
def f13(x):
    return max(100 - x, 0)
    
X = []
Y = []

for k in range(1, 10+1):
    X.append(k)
    Y.append(price3(int(pow(10,5)*k), 100, 0.03, 0.1, 1, f13))

figure13 = figure(num="Question 13 - price3")
plot(X,Y)
title("price3 en fonction de k")
xlabel("k")
ylabel("price3")
grid()
show()




# Question 15

# Fonction put
def put(s, r, sigma, T, K):
    d = 1/(sigma*sqrt(T))
    d *= log(s/K) + (r + pow(sigma,2)/2)*T
    p = -s * norm.cdf(-d) + K*exp(-r*T)*norm.cdf(-d + sigma*sqrt(T))
    return p
    
# Question 16
print("\nQuestion 16")
print("put = ",put(100, 0.04, 0.1, 1, 100) )
    
    
# Question 17
X = []
p3 = []

for k in range(1, 10+1):
    X.append(k)
    p3.append(price3((10**5)*k, 100, 0.03, 0.1, 1, f13))
    
figure17 = figure(num="Question 17 - price3 et put")
plot(X, p3, label="price3")
title("price3 et put")

# Tracé de la valeur de p obtenue par la fonction put
axhline(put(100, 0.03, 0.1, 1, 100), color="r", label="put")

legend()
xlabel("k")
ylabel("price3")
grid()
show()
    
# On remarque que la valeur donnée pour put semble être la moyenne des valeurs données par price3
    

# Question 18

X_T = [1/12, 1/6, 1/4, 1/3, 1/2, 1]
X_k = [k for k in range(1,11)]
Y = zeros((len(X_T), len(X_k)))

figure18 = figure(num="Question 18 - Graphe 3D")
ax = Axes3D(figure18)
X_k, X_T = np.meshgrid(X_k, X_T) #Création de deux grilles pour le graphique 3D

for k in range(len(X_k)):
    for t in range(len(X_T[1])):
        Y[k][t] = put(20*X_k[k][t], 0.03, 0.1, X_T[k][t], 100)

ax.plot_surface(X_k, X_T, Y, cmap='hot_r', rstride=1, cstride=1)
title("put en fonction de k et T")
xlabel("k")
ylabel("T")

show()
 

# Question 19

P1 = []
k = [i for i in range(1,101)]

def f19(y):
    return max(100-y, 0)
    
s = 100
sigma = 0.2
r = 0.04
T = 1

for i in range(len(k)):
    N = 10*k[i]
    rN = r*T/N
    hN = (1+rN)*exp(sigma*sqrt(T/N)) - 1
    bN = (1+rN)*exp(-sigma*sqrt(T/N)) - 1
    P1.append(price1(N, rN, hN, bN, 100, f19))
   
figure19 = figure(num="Question 19")
plot(k, P1, label="price1")
title("put et price1 en fonction de k")
xlabel("k")
ylabel("price1")
grid()

# Tracé de la valeur p obtenue par la fonction p
axhline(put(s, r, sigma, T, 100), color="r", label="put")

legend()
show()

 
# Question 20

# Différences Finies Explicites
def DFE(K, r, sigma, T, L, M, N):
    A = zeros((M, M))
    for i in range(0, M):
        for j in range(0, M):
            if i == j:
                A[i][j] = N/T - sigma*sigma*(i+1)*(i+1) - r + r*(i+1)
            elif i-j == 1:
                A[i][j] = sigma*sigma*(i+1)*(i+1)/2 - r*(i+1)
            elif j-i == 1:
                A[i][j] = sigma*sigma*(i+1)*(i+1)/2
           
    P = zeros((M, 1))
    for i in range(0, M):
        P[i] = max(K-i*L/(M+1), 0)
    Pc = zeros((M, 1))
    X = zeros((M, 1))
    for i in range(1,N):
        Pc[0] = (sigma*sigma/2 - r) * K * exp(r*(i*T/N - T))
        X = np.matmul(A, P)
        P = T/N * (X + Pc)
    return(P)

# Différences Finies Implicites
def DFI(K, r, sigma, T, L, M, N):
    A = zeros((M, M))
    for i in range(0, M):
        for j in range(0, M):
            if i == j:
                A[i][j] = -N/T - sigma*sigma*(i+1)*(i+1) - r + r*(i+1)
            elif i-j == 1:
                A[i][j] = sigma*sigma*(i+1)*(i+1)/2 - r*(i+1)
            elif j-i == 1:
                A[i][j] = sigma*sigma*(i+1)*(i+1)/2
    A1 = np.linalg.inv(A)
    
    P = zeros((M, 1))
    for i in range(0, M):
        P[i] = max(K-i*L/(M+1), 0)
    Pc = zeros((M, 1))
    X = zeros((M, 1))
    for i in range(N-1, 0, -1):
        Pc[0] = (sigma*sigma/2 - r) * K * exp(r*(i*T/N - T))
        X = -N/T * P - Pc
        P = np.matmul(A1, X)
    return(P)
    
# Méthode de Crank-Nicholson
def MCN(K, r, sigma, T, L, M, N):
    A = zeros((M, M))
    B = zeros((M, M))
    for i in range(0, M):
        for j in range(0, M):
            if i == j:
                A[i][j] = -N/T - sigma*sigma*(i+1)*(i+1)/2 - r
            elif i-j == 1:
                A[i][j] = sigma*sigma*(i+1)*(i+1)/4 - r*(i+1)/4
            elif j-i == 1:
                A[i][j] = sigma*sigma*(i+1)*(i+1)/4 + r*(i+1)/4
    B = -A
    for i in range(0, M):
        for j in range(0, M):
            if i == j:
                B[i][j] = -N/T + sigma*sigma*(i+1)*(i+1)/2  
    A1 = np.linalg.inv(A)

    P = zeros((M, 1))
    for i in range(0, M):
        P[i] = max(K-i*L/(M+1), 0)
    Pc = zeros((M, 1))
    Pc1 = zeros((M, 1))
    X = zeros((M, 1))
    for i in range(N-1, 0, -1):
        Pc[0] = (r/4 - sigma*sigma/4) * K * exp(r*(i*T/N - T))
        Pc1[0] = (sigma*sigma/4 - r/4) * K * exp(r*((i+1)*T/N - T))
        X = np.matmul(B, P) + Pc1 - Pc 
        P = np.matmul(A1, X)
    return(P)    
    

X = []

for k in range(0, 1000):
     X.append(k)
P1 = DFE(100,0.04,0.1,1,400,1000,1000000)
P2 = DFI(100,0.04,0.1,1,400,1000,10)
P3 = MCN(100,0.04,0.1,1,400,1000,10)


figure20 = figure(num="Question 20")
subplot(221)
plot(X,P1)
title("Différences Finies Explicites")
xlabel("i")
ylabel("Pn")
grid()

subplot(222)
plot(X,P2)
title("Différences Finies Implicites")
xlabel("i")
ylabel("Pn")
grid()

subplot(223)
plot(X,P3)
title("Méthode de Crank-Nicholson")
xlabel("i")
ylabel("Pn")
grid()
show() 
    
    
 
 
 
    
    
    
    
    
    
    
    
    
    
    
