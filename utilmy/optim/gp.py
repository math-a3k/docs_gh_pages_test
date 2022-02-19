
ni = 2
no = 1

#=================================
# Printing control panel
#=================================
print_after = 100
print_best = 0
#=================================
# Grid Parameters
#=================================
nf = ['+','-','*','/']
nc,nr = 18,1  ## columns x rows (Controls the grid shape and size)
l = nc  ## levels-back (Suggested: l=nc if nr=1)
o2i = 0 # Output connect to inputs?
#=================================
# Algorithm Configuration
#=================================
## Default (n=10, pa=0.3)
n = 20  ## Nest size (Population) (Suggested: 10~15)
pa = 0.3  ## Parasitic Probability (Suggested: 0.3)
kmax = 100000  ## Max iterations
rand_inject = 25
#=================================
# Packages
#=================================
import signal
import sys
from random import randint,randrange,uniform,choices,sample,choice
import math
import numpy as np
from subprocess import check_output
import re
from itertools import islice
from copy import deepcopy
from operator import itemgetter
from timeit import default_timer as timer
from datetime import datetime
from pathlib import Path
from collections import OrderedDict
sys.setrecursionlimit(24000)

#=================================
# Function Definitions
#=================================
def sigint_handler(signal, frame):  # Press Ctrl+C to interrupt
    print ("\n\nInterrupted !!")
    print ("xxxxxxxxxxxxxxxxxxxxxxxxx")
    try:
        total,ER,r_cost,ratioA,n_level,ckt,n_ratioA,vectors,n_gates = cost(best_egg[0])
        print('\n#{}'.format(k),('cost:{} [{}] [{}] ({})'.format('%.3f'%(total),'%.2f'%(ER),'%.2f'%(n_ratioA)),\
                               'distance:{}'.format(vectors),\
                               'A:{}/{}'.format('%.3f'%(ratioA),'%.3f'%(tgt_ratioA)),\
                               'level:{}/{}'.format(n_level,tgt_level),\
                               'gates:{}/{} [{}/{}]'.format(n_gates,tgt_gates,'%.3f'%(r_cost),'%.3f'%(tgt_cost))),\
                                datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        printls(ckt)
    except:
        print ("No best_egg so far")
    sys.exit(0)
signal.signal(signal.SIGINT, sigint_handler)



def is_valid(G):
    def allUnique(x):
        seen = list()
        return not any(i in seen or seen.append(i) for i in x)

    if allUnique(G[Lg-no:])==True:
        NU = [False]*M

        for i in range(Lg-no,Lg):
            NU[G[i]] = True

        # Find active nodes
        NG = [None]*nn
        for k in range(Ln): # Iterate over all nodes
            for i in range(ni,M):
                if NU[i] == True:
                    index = nn*(i-ni)
                    for j in range (0, nn):
                        NG[j] = G[index+j] # Get node genes
                    for j in range (1, a+1):
                        NU[NG[j]] = True
        
        NP = []
        for j in range (ni, M):
            if NU[j] == True:
                NP.append(j)

        dic_nodes = {}  # Dictionary Initialize 
        for j in range(0,len(NP)):
            node = []
            g = nn * (NP[j]-ni) # Function gene
            node.append(G[g])

            for i in range(1,nn):
                node.append(G[g+i]) # Connection gene

            dic_nodes[NP[j]] = node # Dictionary

        ls = []
        for node in dic_nodes.values():
            ls.append(node[1])
            ls.append(node[2])
                            
        if o2i != 0:
            for output in G[Lg-no:]: # Since o/p allowed to connect to i/p, check o/p for connections to i/p
                ls.append(output)
        
        if set(list(range(ni))) <= set(ls):
            return True
        else:
            return False
    else:
        return False


def random_solution():
    while True:
        G = []
        for j in range(nc):
            for k in range(nr):
                # function bit
                bit0 = randint(0,len(nf)-1)
                # connection bit 1
                if j>=l:
                    bit1 = randint(ni+(j-l)*nr, ni+j*nr-1)
                else:
                    bit1 = randint(0, ni+j*nr-1)
                # connection bit 2
                if j>=l:
                    bit2 = randint(ni+(j-l)*nr, ni+j*nr-1)
                else:
                    bit2 = randint(0, ni+j*nr-1)

                # appending to chromosome
                G.append(bit0)
                G.append(bit1)
                G.append(bit2)

        for k in range(no):
            # output bit
            G.append(randint(ni, ni+Ln-1))

        if is_valid(G) == True: # check if all i/o have been used
            return G


def NodesToProcess(G):
    NU = [False]*M

    for i in range(Lg-no,Lg):
        NU[G[i]] = True

    # Find active nodes
    NG = [None]*nn
    for k in range(Ln): # Iterate over all nodes
        for i in range(ni,M):
            if NU[i] == True:
                index = nn*(i-ni)
                for j in range (0, nn):
                    NG[j] = G[index+j] # Get node genes
                for j in range (1, a+1):
                    NU[NG[j]] = True
    
    NP = []
    for j in range (ni, M):
        if NU[j] == True:
            NP.append(j)

    dic_nodes = {}  # Dictionary Initialize 
    for j in range(0,len(NP)):
        node = []
        g = nn * (NP[j]-ni) # Function gene
        node.append(G[g])

        for i in range(1,nn):
            node.append(G[g+i]) # Connection gene

        dic_nodes[NP[j]] = node # Dictionary

    return dic_nodes


def decode(G):
    nodes = NodesToProcess(G)
    eqn = []

    for i in range (ni):  #Inputs
        eqn.append("INPUT(n{})\n".format(i))

    for item in G[Lg-no:]:
        eqn.append("OUTPUT(n{})\n".format(item))  #Outputs
    
    for key, value in nodes.items(): 
        func = nf[value[0]]
        eqn.append("x{} = ({})\n".format(key,value[1],func,value[2]))

    return eqn


def random_walker(G,h):
    golden_G = deepcopy(G)

    while True:
        for k in range(h):
            # generate a random index in range length(chromosome) or Lg
            i_bit = randrange(Lg)

            if i_bit >= (Lg-no):  # its an output bit
                while True:
                    to_replace = randint(ni, ni+Ln-1)
                    if not(G[i_bit] == to_replace):
                        G[i_bit] = to_replace
                        break

            elif i_bit % 3 == 0 or i_bit == '0': # its a function bit
                while True:
                    to_replace = randint(0,len(nf)-1)
                    if not(G[i_bit] == to_replace):
                        G[i_bit] = to_replace
                        break

            else: # else its a connection bit
                for index in range(nc):
                    if i_bit in range(nr*nn*index, (nr*nn*(index+1))):
                        j = index

                if j>=l:
                    while True:
                        to_replace = randint(ni+(j-l)*nr, ni+j*nr-1)
                        if not(G[i_bit] == to_replace):
                            G[i_bit] = to_replace
                            break
                else:
                    while True:
                        to_replace = randint(0, ni+j*nr-1)
                        if not(G[i_bit] == to_replace):
                            G[i_bit] = to_replace
                            break

        if is_valid(G) == True: # check if all i/o have been used
            return G
        else:
            G = deepcopy(golden_G)


def cost(G):
    def normalize(val,Rmin,Rmax,Tmin,Tmax):
        return (((val-Rmin)/(Rmax-Rmin)*(Tmax-Tmin))+Tmin)

    def denormalize(val,Rmin,Rmax,Tmin,Tmax):
        return (((val-Tmin)/(Tmax-Tmin)*(Rmax-Rmin))+Rmin)

    ckt = decode(G)
    
    # Process for ER
    difference_CNF(ckt) # Make CNF b/w target and candidate ckt (G/s)
    
    os.chdir(dir_cachet)
    lines = str(check_output('./cachet CNF', shell=True)).split('\n') # Call Cachet
    for line in lines: # Process output from Cachet
        if "Number of solutions" in line:
            vectors = int(re.search(r'\d+', line.split('Number of solutions')[1]).group())
            break

    ER = vectors/(2**ni)
    n_ER = normalize(vectors,0,2**ni,0,alpha)
    
    # Process for ratio A
    os.chdir(dir_hope)

    f = open("ckt",'w')
    f.write("# ckt\n")
    for line in ckt:
        f.write(line)
    f.close()

    ## Call HOPE
    check_output('./hope -N -t patt -l log ckt', shell=True)

    ## Read the log file
    f = open("log",'r')
    log = f.readlines()
    f.close()

    for line in log:

        #if "Number of combinational gates" in line:
            #n_gates = int(line.split(':')[1].strip())

        if "Level of the circuit" in line:
            n_level = int(line.split(':')[1].strip())

        elif "Number of collapsed faults" in line:
            n_collapsed = int(line.split(':')[1].strip())

        elif "Number of detected faults" in line:
            n_detected = int(line.split(':')[1].strip())

    r_cost, n_gates = implementation_cost(ckt)

    ## ratio A
    ratioA = n_detected/n_collapsed
    n_ratioA = normalize(ratioA,0,1.0,0,1-alpha)

    ## final cost
    total = n_ER + n_ratioA
        
    os.chdir(dir_home)
    
    return(total,ER,r_cost,ratioA,n_level,ckt,n_ratioA,vectors,n_gates)


def search():
    def levyFlight(u):
        return (math.pow(u,-1.0/3.0)) # Golden ratio = 1.62

    def randF():
        return (uniform(0.0001,0.9999))

    #############################################################
    var_levy = []
    for i in range(1000):    
        var_levy.append(round(levyFlight(randF())))
    var_choice = choice
    #############################################################

    def offsprings(nest,k_way):
        ## Best egg as father for breeding
        father = nest[0][0]
        children = []
        
        while(True):
            ## Tournament Selection for mother
            tournament = choices(nest[1:],k=k_way) # Eggs for tournament
            mother = min(tournament, key=itemgetter(1))[0] # winner            

            ## Single-point Crossover
            i_bit = randint(0,Lg-1)

            if randF() < randF():
                # Child1
                child1 = father[:i_bit]+mother[i_bit:]
                # Child2
                child2 = mother[:i_bit]+father[i_bit:]
            else:
                # Child1
                child1 = mother[:i_bit]+father[i_bit:]
                # Child2
                child2 = father[:i_bit]+mother[i_bit:]
                    
            if is_valid(child1)==True:
                child1 = random_walker(child1,var_choice(var_levy)) # Mutation
                children = children + [child1]
            
            if is_valid(child2)==True:
                child2 = random_walker(child2,var_choice(var_levy)) # Mutation
                children = children + [child2]

            if len(children) >= n_replace:
                return(children[:n_replace])

    
    # Initialize the nest
    nest = []
    for i in range(n-1):
        egg = random_solution()
        nest.append((egg,cost(egg)[:6]))
    nest.append((seed,cost(seed)[:6])) # Adding seed as nth member
    nest.sort(key = itemgetter(1)) #Sorting -> same as: nest.sort(key = lambda x: x[1][0])

    ## Dump reference
    f = open(f_dump,'w')
    f.write('RatioA*Area*Level*Gates*Delay*Power*PDP*Circuit\n')
    f.write(str(round(nest[0][1][3], 3))+'*'+str(round(nest[0][1][2], 3))+'*'+str(round(nest[0][1][4], 3))\
            +'*'+'*'+'*'+'*'+'*'+(''.join(map(str, nest[0][1][-1])).replace('\n','\\n'))+'\n')
    f.close()
    
    global best_egg
    global k
    global dic_front
    ls_trace = []
    dic_front = {}
    dic_front[(nest[0][1][2],nest[0][1][3],nest[0][1][4])] = nest[0][1][-1]
    counter = 0
    # Main Loop
    for k in range(kmax+1):
        # Lay cuckoo eggs
        for i in range(n_cuckoo_eggs):
            index = randint(0,n-1)
            egg = deepcopy(nest[index]) # Pick an egg at random from the nest
            cuckoo = random_walker(egg[0],var_choice(var_levy)) # Lay a cuckoo egg from that egg
            cost_cuckoo = cost(cuckoo)[:6]
            if (cost_cuckoo[0] <= egg[1][0]): # Check if the cuckoo egg is better
                nest[index] = (cuckoo,cost_cuckoo)

        nest.sort(key = itemgetter(1)) # Sorting


        # Frontiers
        ls_eqv = [egg for egg in nest if egg[1][1]==0] # Consider only if ER==0
        if len(ls_eqv) > 0: # if any ER>0
            for egg in ls_eqv:

                if (egg[1][2],egg[1][3],egg[1][4]) not in dic_front.keys():
                    dic_front[(egg[1][2],egg[1][3],egg[1][4])] = egg[1][-1]


        # Store ratioA for trace
        ls_trace.append(nest[0][1][3])
        
                
        if counter<rand_inject:
            # Genetically modify eggs with probability pa
            if n_replace>0:
                children = offsprings(nest,k_way)
                for i in range(n_replace):
                    nest[(n-1)-(i)] = (children[i],cost(children[i])[:6])
            counter+=1
        else:
            counter = 0
            for i in range(n_replace):
                egg = random_solution()
                nest[(n-1)-(i)] = (egg,cost(egg)[:6])


        # Iterational printing
        if (k%print_after == 0):
            
            with open(f_trace,'a') as f:
                for trace in ls_trace:
                    f.write(str(round(trace, 3))+'\n')
            ls_trace = [] # empty after dumping
            
            nest.sort(key = itemgetter(1)) # Rank nests and find current best
            best_egg = deepcopy(nest[0])
            total,ER,r_cost,ratioA,n_level,ckt,n_ratioA,vectors,n_gates = cost(best_egg[0])
            print('\n#{}'.format(k),('cost:{} [{}] ({})'.format('%.3f'%(total),'%.2f'%(ER),'%.2f'%(n_ratioA)),\
                                   'distance:{}'.format(vectors),\
                                   'A:{}/{}'.format('%.3f'%(ratioA),'%.3f'%(tgt_ratioA)),\
                                   'level:{}/{}'.format(n_level,tgt_level),\
                                   'gates:{}/{} [{}/{}]'.format(n_gates,tgt_gates,'%.3f'%(r_cost),'%.3f'%(tgt_cost))),\
                                    datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

            if (print_best==1):
                printls(ckt)
                print('\n')
                
                
#=================================
# Remaining Grid Parameters
#=================================
Ln = nc * nr  # Total nodes
M = Ln + ni  # Max no. of addresses in the Graph
a = 2  # Arity
nn = a + 1  # Integers per node/ Size of each node
Lg = (nc * nr*(a + 1)) + no  # Integers in a candidate (Size of a candidate)
nu = 0  # No of active/used nodes

#=================================
# Init Status Printing
#=================================
print("I/O count: {}/{}".format(ni,no))
print("Grid shape: {}x{}".format(nc,nr))
print("Cells: {}".format(nf))
if o2i==1:
    print("o2i: Yes")
    o2i = 0
elif o2i==0:
    print("o2i: No")
    o2i = ni+Ln-(nr*l)
print("Nest size: {}".format(n))
print("Parasitic Probability: {}".format(pa))
print("kmax: {}".format(kmax))
print("rand_inject: {}".format(rand_inject))
try:
    print("Step size: {}".format(step))
    s = round(Lg*step)
except:
    print("Step size: Lévy flight")
print("print_after: {}".format(print_after))
print("print_best: {}\n\n".format(print_best))


#=================================
# Another Miscellaneous
#=================================
n_cuckoo_eggs = round(pa*n)
n_replace = round(pa*n)
k_way = round(pa*n)




geno = random_solution()
pheno = decode(geno)
print(pheno)


#=================================
# Execution
#=================================



#search()




print('\n\n\n\nDONE')
## That's it folks





    

    

