
def shell_iterator(am): # From the libint manual
    for i in range(am+1):
        for j in range(i+1):
            print(am-i, i-j, j)
            yield am-i,i-j,j
    return

for i in shell_iterator(3):
    k = 0
    #print(i)
#print(shell_iterator(3))
