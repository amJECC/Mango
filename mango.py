import numpy as np
import matplotlib.pyplot as plt
import functools
import random
import collections
import math
from matplotlib.animation import FuncAnimation

#----------Variables Globales-----------
print("Hola")
n_poblacion = 100
n_generacion = 1000
tasaMutacion = 0.05
tasaCruza = 1.0

dmin = 0
dmax = 1
n=3
m=3
#---------------------------------------

def graficar2d(pt):
    
    plt.plot(pt[:n_poblacion,0],pt[:n_poblacion,1],"+")
    plt.pause(0.001)
    plt.draw()
    plt.cla()
    
def graficar(pt,ax):
    
    xg = pt[:n_poblacion,0]
    yg = pt[:n_poblacion,1]
    zg = pt[:n_poblacion,2]
    
    # Plotea los puntos en el espacio 3D
    ax.scatter(xg, yg, zg, c='black', marker='+')

    # Configura las etiquetas de los ejes
    ax.set_xlabel('Eje X')
    ax.set_ylabel('Eje Y')
    ax.set_zlabel('Eje Z')
    plt.pause(0.001)
    plt.draw()
    plt.cla()
    
def dominancia(a,b):
    condicion_1 = a <= b
    condicion_2 = a < b
    if np.all(condicion_1) and np.any(condicion_2):
        return True
    return False

def matriz_dominancia(datos):    
    
    p_t = datos.shape[0]
    n = datos.shape[1]
    
    x = np.zeros([p_t,p_t,n])
    y = np.zeros([p_t,p_t,n])
    
    for i in range(p_t):
        x[i,:,:] = datos[i]
        y[:,i,:] = datos[i]
    
    condicion_1 = x <= y
    condicion_2 = x < y
    
    return np.logical_and(np.all(condicion_1,axis=2),np.any(condicion_2,axis=2))  
       
def frentes_pareto(datos):
    
    conjunto_d = []
    cuenta = []
    
    matriz = matriz_dominancia(datos)
    pop_size = datos.shape[0]
    
    for i in range(pop_size):
        dominante_actual = set()
        cuenta.append(0)
        for j in range(pop_size):
            if matriz[i,j]:
                dominante_actual.add(j)
            elif matriz[j,i]:
                cuenta[-1] += 1
                
        conjunto_d.append(dominante_actual)

    cuenta = np.array(cuenta)
    frentes = []
    while True:
        frente_actual = np.where(cuenta==0)[0]
        if len(frente_actual) == 0:
            
            break
        
        frentes.append(frente_actual)

        for individual in frente_actual:
            cuenta[individual] = -1 
            dominado_actual_c = conjunto_d[individual]
            for dominado_aux in dominado_actual_c:
                cuenta[dominado_aux] -= 1
            
    return frentes

def crowding(datos,fronts):
    
    columnas = datos.shape[1] #columnas=f1(x->), f2(x->)
    filas = datos.shape[0] #filas=x-> = [var1, var2, var3, hasta n]
    
    crowding_distance = np.zeros_like(datos)
    for n in range(columnas):
        f_min = np.min(datos[:,n])
        f_max = np.max(datos[:,n])
        v = f_max - f_min
        crowding_distance[:,n] = (datos[:,n] - f_min) / v
    
    datos = crowding_distance
    crowding_medida = np.zeros(filas)

    for front in fronts:
        for n in range(columnas):
            
            frente_ordenado = sorted(front,key = lambda x : datos[x,n])
            #0 pos actual, -1 sig pos
            crowding_medida[frente_ordenado[0]] = np.inf
            crowding_medida[frente_ordenado[-1]] = np.inf #se ponen en infinito ambas boundary solutions
            if len(frente_ordenado) > 2:
                for i in range(1,len(frente_ordenado)-1):
                    crowding_medida[frente_ordenado[i]] += datos[frente_ordenado[i+1],n] - datos[frente_ordenado[i-1],n]
    return  crowding_medida

def rank(fronts):  #primer ordenamiento antes de aplicar crowding
    rank_indice = {}
    for i,front in enumerate(fronts):
        for x in front:   
            rank_indice[x] = i
            
    return rank_indice
        
def ordenamiento_no_dominado(rank_indice,crowding):
    
    num_filas = len(crowding)
    indicies = list(range(num_filas))

    def rank_no_dominado(a,b):
        
        if rank_indice[a] > rank_indice[b]:  
            return -1 #-1 si bdom a y b es el menos aglomerado
        elif rank_indice[a] < rank_indice[b]:
            return 1 #si a domina b y a es menos aglomerado
        else:
            if crowding[a] < crowding[b]:   
                return -1
            elif crowding[a] > crowding[b]:
                return 1
            else:
                return 0 #si son iguales en todo sentido
    #indices del ordenamiento no dominado, del mejor al "peor"
    no_dominantes = sorted(indicies,key = functools.cmp_to_key(rank_no_dominado),reverse=True) 
    
    return no_dominantes

def seleccion(individuos,q_t): #seleccion aleatoria
    elegidos = []
    for _ in range(q_t):
        index = np.random.randint(0,individuos) 
        padre = index
        elegidos.append(padre)
    
    return elegidos

def cruza(individuos, tasa):
    descendencia = individuos.copy()
    for i in range(int(tasa/2)):
        x = np.random.randint(0, individuos.shape[0])
        y = np.random.randint(0, individuos.shape[0])
        while x == y:
            x = np.random.randint(0, individuos.shape[0])
            y = np.random.randint(0, individuos.shape[0])
        punto_cruza = np.random.randint(1, individuos.shape[1])
        descendencia[2*i, 0:punto_cruza] = individuos[x, 0:punto_cruza] #hijo 1
        descendencia[2*i, punto_cruza:] = individuos[y, punto_cruza:]
        descendencia[2*i+1, 0:punto_cruza] = individuos[y, 0:punto_cruza] #hijo_2
        descendencia[2*i+1, punto_cruza:] = individuos[x, punto_cruza:] 
    return descendencia

def mutacion(individuo,min_val,max_val,tasa_m):
    mutacion_p = (max_val - min_val) * tasa_m
    descendencia = individuo.copy()
    descendencia += np.random.normal(0,mutacion_p,size = descendencia.shape)
    descendencia = np.clip(descendencia,min_val,max_val)
    return descendencia

def NSGA_II(p_t,valores,g):
    print(f"NSGA-ll en la generacion {g+1}")
    fronts = frentes_pareto(valores)
    posicion_ranking = rank(fronts)
    
    crowding_d = crowding(valores,fronts)
    
    indices_nd = ordenamiento_no_dominado(posicion_ranking,crowding_d)
    
    sobrevivientes = p_t[indices_nd[:n_poblacion]] #tomamos a la porcion de mejores sobrevivientes
    seleccionados = seleccion(individuos=n_poblacion,q_t=n_poblacion)
    cruza_t = cruza(seleccionados, tasa=tasaCruza)
    pt_next = np.array([mutacion(sobrevivientes[i],dmin,dmax,tasaMutacion) for i in cruza_t]) #con los mecanismos obtenenos la porcion q_t
    
    poblacion_sig = np.concatenate([sobrevivientes,pt_next])  # se forma r_t con p_t y q_t
    return poblacion_sig

def sumatoriaFon(x,n,f): 
    s=0
    for i in range(n):
        if(f == "f1"):
            s += pow((x[:,i] - (1- np.sqrt(n) ) ),2)
        elif(f == "f2"):
            s += pow((x[:,i] + (1- np.sqrt(n))),2)
    return s 

def fonseca(x,n):

    f1 = 1 - np.exp(-sumatoriaFon(x,n,"f1"))
    f2 = 1 - np.exp(-sumatoriaFon(x,n,"f2"))
    
    return np.stack([f1,f2],axis=1)

def seno(f):
    for i,g in enumerate(f):
        f[i] = math.sin(10*math.pi*f[i])     
    return f

def g(x,n):
    g =0
    for i in range(n):
        g += 1 + (9/29) * (x[:,i])
    return g
    
def h1(f,g):
    h = 1 - np.sqrt(f/g)
    return h
 
def h2(f,g):
    h = 1 - pow((f/g),2)
    return h

def h3(f,g):
    h = 1 - np.sqrt(f/g) - ((f/g) * seno(f))
    return h
   
def f(x):
    f = x[:,1]
    return f

def ZDT1(x,n):
    f1 = f(x)
    f2 = g(x,n)*h1(f(x),g(x,n))
    return np.stack([f1,f2],axis=1)

def ZDT2(x,n):
    f1 = f(x)
    f2 = g(x,n)*h2(f(x),g(x,n))
    return np.stack([f1,f2],axis=1)

def ZDT3(x,n):
    f1 = f(x)
    f2 = g(x,n)*h3(f(x),g(x,n))
    return np.stack([f1,f2],axis=1)
 
def omega(gx,xi):

    return np.pi / (4 * (1 + gx)) * (1 + 2 * gx * xi)

def DTLZ1(x):

    funciones = []

    gx = 100 * (np.sum(np.square(x[:, m:]-0.5) - np.cos(20*np.pi*(x[:, m:]-0.5)), axis=1))
 
    for f in range(m): #M numero de funciones
        if(f == m-1):
            xi = (1 - x[:,0]) #( 1 - x1 )
        else:
            xi = x[:,0] #x1
            
            for i in range((m-1)-f-1): #(... Xm-1)
                xi  = xi * x[:,(i+1)] 
            
            if(f == 0):
                xi = xi
            else:
                xi = xi * (1-x[:,((m-1)-f)]) # (1 - Xm-1)
            
        fi = 0.5 * (xi * (1+gx)) #fm(x)

        funciones.append(fi)

    return np.stack(funciones,axis=1)
   
def DTLZ2(x):

    funciones = []
    
    gx = np.sum(np.square(x[:, m:]-0.5), axis=1)
    
    for f in range(m): #M numero de funciones
        if(f == m-1):
            xi = (np.sin(x[:,0] * (np.pi)/2)) #( sen(x1) pi/2 )
        else:
            xi = np.cos(x[:,0] * (np.pi)/2) #x1
            
            for i in range((m-1)-f-1): #(... Xm-1)
                
                xi  = xi * (np.cos(x[:,i+1] * (np.pi)/2)) 
            
            if(f == 0):
                xi = xi * (np.cos(x[:,((m-2)-f)] * (np.pi)/2)) 
            else:
                xi = xi * (np.sin(x[:,((m-1)-f)] * (np.pi)/2))  # (1 - Xm-1)
            
        fi = (1 + gx) * xi #fm(x)

        funciones.append(fi)
    
    
    return np.stack(funciones, axis=1)
    
def DTLZ3(x):

    funciones = []
    
    gx = 100 * (np.sum(np.square(x[:, m:]-0.5) - np.cos(20*np.pi*(x[:, m:]-0.5)), axis=1))
    
    for f in range(m): #M numero de funciones
        if(f == m-1):
            xi = (np.sin(x[:,0] * (np.pi)/2)) #( sen(x1) pi/2 )
        else:
            xi = np.cos(x[:,0] * (np.pi)/2) #x1
            
            for i in range((m-1)-f-1): #(... Xm-1)
                
                xi  = xi * (np.cos(x[:,i+1] * (np.pi)/2)) 
            
            if(f == 0):
                xi = xi * (np.cos(x[:,((m-2)-f)] * (np.pi)/2)) 
            else:
                xi = xi * (np.sin(x[:,((m-1)-f)] * (np.pi)/2))  # (1 - Xm-1)
            
        fi = (1 + gx) * xi #fm(x)

        funciones.append(fi)
    
    
    return np.stack(funciones, axis=1)
    
def DTLZ4(x):

    funciones = []
    
    a = 10
    
    gx = np.sum(np.square(x[:, m:]-0.5), axis=1)
    
    for f in range(m): #M numero de funciones
        if(f == m-1):
            xi = (np.sin(np.power(x[:,0],a) * (np.pi)/2)) #( sen(x1) pi/2 )
        else:
            xi = np.cos(np.power(x[:,0],a) * (np.pi)/2) #x1
            
            for i in range((m-1)-f-1): #(... Xm-1)
                
                xi  = xi * (np.cos(np.power(x[:,i+1],a) * (np.pi)/2)) 
            
            if(f == 0):
                xi = xi * (np.cos(np.power(x[:,((m-2)-f)],a) * (np.pi)/2)) 
            else:
                xi = xi * (np.sin(np.power(x[:,((m-1)-f)],a) * (np.pi)/2))  # (1 - Xm-1)
            
        fi = (1 + gx) * xi #fm(x)

        funciones.append(fi)
    
    
    return np.stack(funciones, axis=1)
    
def DTLZ5(x):

    funciones = []

    gx = np.sum(np.square(x[:, m:]-0.5), axis=1)
    
    for f in range(m): #M numero de funciones
        if(f == m-1):
            xi = (np.sin(omega(gx,x[:,0]) * (np.pi)/2)) #( sen(x1) pi/2 )
        else:
            xi = np.cos(omega(gx,x[:,0]) * (np.pi)/2) #x1
            
            for i in range((m-1)-f-1): #(... Xm-1)
                
                xi  = xi * (np.cos(omega(gx,x[:,i+1]) * (np.pi)/2)) 
            
            if(f == 0):
                xi = xi * (np.cos(omega(gx,x[:,((m-2)-f)]) * (np.pi)/2)) 
            else:
                xi = xi * (np.sin(omega(gx,x[:,((m-1)-f)]) * (np.pi)/2))  # (1 - Xm-1)
            
        fi = (1 + gx) * xi #fm(x)

        funciones.append(fi)
    
    
    return np.stack(funciones, axis=1)
    
def DTLZ6(x):
   
    funciones = []

    gx = np.sum(np.power(x[:, m:],0.1), axis=1)
    
    for f in range(m): #M numero de funciones
        if(f == m-1):
            xi = (np.sin(omega(gx,x[:,0]) * (np.pi)/2)) #( sen(x1) pi/2 )
        else:
            xi = np.cos(omega(gx,x[:,0]) * (np.pi)/2) #x1
            
            for i in range((m-1)-f-1): #(... Xm-1)
                
                xi  = xi * (np.cos(omega(gx,x[:,i+1]) * (np.pi)/2)) 
            
            if(f == 0):
                xi = xi * (np.cos(omega(gx,x[:,((m-2)-f)]) * (np.pi)/2)) 
            else:
                xi = xi * (np.sin(omega(gx,x[:,((m-1)-f)]) * (np.pi)/2))  # (1 - Xm-1)
            
        fi = (1 + gx) * xi #fm(x)

        funciones.append(fi)
    
    
    return np.stack(funciones, axis=1)
    
def DTLZ7(x):
    print("DTLZ7")

def kursawe(x):

    sq_x = x**2
    objetivo_1 = -10*np.exp(-0.2*np.sqrt(np.sum(sq_x[:, :2], axis=1))) - 10*np.exp(-0.2*np.sqrt(np.sum(sq_x[:, 1:], axis=1)))
    objetivo_2 = np.sum(np.power(np.abs(x), 0.8) + 5*np.sin(x**3), axis=1)
    
    
    return np.stack([objetivo_1, objetivo_2], axis=1)

def graf(P):
    # Extrae las coordenadas X, Y y Z de tus datos
    x = P[:n_poblacion,0]  # Primera columna
    y = P[:n_poblacion,1] # Segunda columna
    z = P[:n_poblacion,2]  # Tercera columna

    # Crea una figura 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotea los puntos en el espacio 3D
    ax.scatter(x, y, z, c='b', marker='o')

    # Configura las etiquetas de los ejes
    ax.set_xlabel('Eje X')
    ax.set_ylabel('Eje Y')
    ax.set_zlabel('Eje Z')

    # Muestra el gráfico
    plt.show()  

def main():
    print("Hola, iniciando . . . ")
        
    #Creamos la poblacion inicial p0
    x = np.random.uniform(dmin,dmax,(2*n_poblacion,n))
    plt.ion()
    ax = plt.axes(projection = "3d")
    ax.view_init(elev=30, azim=45)
    
    for generation in range(n_generacion):

        pt = DTLZ2(x)

        x = NSGA_II(x,pt,generation)
        
        graficar(pt,ax)
        #graficar2d(pt)
        
    #graf(pt)
    print("Evaluacion terminada")   
        

    plt.ioff()
    
    #plt.plot(pt[:n_poblacion,0],pt[:n_poblacion,1],".",color="r")
    #plt.show()

    ax = plt.axes(projection = "3d")
    ax.view_init(elev=30, azim=45)
    xg = pt[:n_poblacion,0]
    yg = pt[:n_poblacion,1]
    zg = pt[:n_poblacion,2]
    
    # Plotea los puntos en el espacio 3D
    ax.scatter(xg, yg, zg, c='black', marker='+')

    # Configura las etiquetas de los ejes
    ax.set_xlabel('Eje X')
    ax.set_ylabel('Eje Y')
    ax.set_zlabel('Eje Z')
    
    plt.show()

main()


#Comentario
