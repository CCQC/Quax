import numpy as np
import time
# Initialize Arrays  
dim = 2 # dimension of arrays ... e.g number of basis functions  
MO1 = np.zeros((dim,dim,dim,dim)) # For our first dumb O[N^8] method  
MO2 = np.zeros((dim,dim,dim,dim)) # For our smarter O[N^5] method

INT = np.random.randint(9,size=(dim,dim,dim,dim)) # Our toy "two electron integrals"  
C = np.random.randint(9,size=(dim,dim)) # Toy "wavefunction coefficients"

# Begin first method. It scales as N^8, as you could  
# have guessed with there being 8 loops over dimension 'dim' (N)

t0 = time.time()  
for i in range(0,dim):  
    for j in range(0,dim):  
        for k in range(0,dim):  
            for l in range(0,dim):  
                for m in range(0,dim):  
                    for n in range(0,dim):  
                        for o in range(0,dim):  
                            for p in range(0,dim):  
                                MO1[i,j,k,l] += C[i,m]*C[j,n]*C[k,o]*C[l,p]*INT[m,n,o,p]  
t1 = time.time()

# Begin second method, scaling as N^5. We end up having four 5-loops, each  
# over dimension 'dim' (N).

t2 = time.time()  
temp = np.zeros((dim,dim,dim,dim))  
temp2 = np.zeros((dim,dim,dim,dim))  
temp3= np.zeros((dim,dim,dim,dim))  
for i in range(0,dim):  
    for m in range(0,dim):  
        temp[i,:,:,:] += C[i,m]*INT[m,:,:,:]  
    for j in range(0,dim):  
        for n in range(0,dim):  
            temp2[i,j,:,:] += C[j,n]*temp[i,n,:,:]  
        for k in range(0,dim):  
            for o in range(0,dim):  
                temp3[i,j,k,:] += C[k,o]*temp2[i,j,o,:]  
            for l in range(0,dim):  
                for p in range(0,dim):  
                    MO2[i,j,k,l] += C[l,p]*temp3[i,j,k,p]  
t3 = time.time()

# Set up random index to check correctness.  
i = np.random.randint(dim)  
j = np.random.randint(dim)  
k = np.random.randint(dim)  
l = np.random.randint(dim)

print(MO1[i,j,k,l]  )
print(MO2[i,j,k,l]  )
print(np.allclose(MO1,MO2))
print("TIME1: ", t1-t0  )
print("TIME2: ", t3-t2 )

#MO3 = np.einsum('pqrs, pP, qQ, rR, sS -> PQRS', INT, C, C, C, C, optimize='optimal')  
MO3 = np.einsum('pqrs, Pp, Qq, Rr, Ss -> PQRS', INT, C, C, C, C, optimize='optimal')  
print(np.allclose(MO1,MO3))

print(MO1)
print(MO3)
