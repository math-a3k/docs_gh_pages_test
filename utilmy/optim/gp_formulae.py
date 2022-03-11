
def run1():
    #=================================
    # Printing control panel
    #=================================
    print_after = 100
    print_best = 1
    #=================================
    # Graph Parameters
    #=================================
    ni = 4
    no = 1
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
    from random import randint,randrange,uniform,choices,sample,choice,shuffle
    import math
    import numpy as np
    from copy import deepcopy
    from operator import itemgetter
    from timeit import default_timer as timer
    import scipy.stats
    from icecream import ic
    import warnings
    warnings.filterwarnings("ignore")
    #=================================
    # Function Definitions
    #=================================
    def sigint_handler(signal, frame):  # Press Ctrl+C to interrupt
        print ("\n\nInterrupted !!")
        print ("xxxxxxxxxxxxxxxxxxxxxxxxx")
        try:
            cost = round(best_egg[1][0], 3)
            print(f'\n#{k}', f'{cost}')
            if (print_best==1):
                print(best_egg[1][1])
                print('\n')
        except:
            print ("No best_egg so far")
        sys.exit(0)
    signal.signal(signal.SIGINT, sigint_handler)

    def log(*s): print(*s, flush=True)

    def get_correlm(eqn):
        """  compare 2 lists lnew, ltrue and output correlation.
        Goal is to find rank_score such Max(correl(lnew(rank_score), ltrue ))
        
        """
        ##### True list
        ltrue = [ str(i)  for i in range(0, 100) ]   

        #### Create noisy list 
        ltrue_rank = {i:x for i,x in enumerate(ltrue)}
        list_overlap =  ltrue[:70]  #### Common elements

        nsample=5
        correls = []
        for i in range(nsample):
            ll1  = rank_generate_fake(ltrue_rank, list_overlap, nsize=100, ncorrect=40)
            ll2  = rank_generate_fake(ltrue_rank, list_overlap, nsize=100, ncorrect=50)

            #### Merge them using rank_score
            lnew = rank_merge_v5(ll1, ll2, kk= 1, eqn = eqn)
            lnew = lnew[:100]
            # log(lnew) 

            ### Eval with True Rank
            correls.append(scipy.stats.spearmanr(ltrue,  lnew).correlation)

        correlm = np.mean(correls)
        return -correlm  ### minimize correlation val

    #### Example of rank_scores0
    def rank_score(eqn:str, rank1:list, rank2:list, adjust=1.0, kk=1.0)-> list:
        """     ### take 2 np.array and calculate one list of float (ie NEW scores for position)
    
        list of items:  a,b,c,d, ...
        item      a,b,c,d,e
        rank1 :   1,2,3,4 ,,n     (  a: 1,  b:2, ..)
        rank2 :   5,7,2,1 ,,n     (  a: 5,  b:6, ..)
        
        scores_new :   a: -7.999,  b:-2.2323   
        (item has new scores)
        
        """

        x0 = adjust
        x1 = kk
        x2 = rank1
        x3 = rank2

        scores_new =  eval(eqn)
        return scores_new

    def rank_merge_v5(ll1:list, ll2:list, eqn:str, kk= 1):
        """ Re-rank elements of list1 using ranking of list2
            20k dataframe : 6 sec ,  4sec if dict is pre-build
            Fastest possible in python
        """
        if len(ll2) < 1: return ll1
        n1, n2 = len(ll1), len(ll2)

        if not isinstance(ll2, dict) :
            ll2 = {x:i for i,x in enumerate( ll2 )  }  ### Most costly op, 50% time.

        adjust, mrank = (1.0 * n1) / n2, n2
        rank2 = np.array([ll2.get(sid, mrank) for sid in ll1])
        rank1 = np.arange(n1)
        rank3 = rank_score(eqn, rank1, rank2, adjust=1.0, kk=1.0) ### Score   

        #### re-rank  based on NEW Scores.
        v = [ll1[i] for i in np.argsort(rank3)]
        return v  #### for later preprocess

    def rank_generate_fake(dict_full, list_overlap, nsize=100, ncorrect=20):
        """  Returns a list of random rankings of size nsize where ncorrect
            elements have correct ranks
            Keyword arguments:
            dict_full    : a dictionary of 1000 objects and their ranks
            list_overlap : list items common to all lists
            nsize        : the total number of elements to be ranked
            ncorrect     : the number of correctly ranked objects
        """
        # first randomly sample nsize - len(list_overlap) elements from dict_full
        # of those, ncorrect of them must be correctly ranked
        random_vals = []
        while len(random_vals) <= nsize - len(list_overlap):
        rand = sample(list(dict_full), 1)
        if (rand not in random_vals and rand not in list_overlap):
            random_vals.append(rand[0])

        # next create list as aggregate of random_vals and list_overlap
        list2 = random_vals + list_overlap
        
        # shuffle nsize - ncorrect elements from list2 
        copy = list2[0:nsize - ncorrect]
        shuffle(copy)
        list2[0:nsize - ncorrect] = copy

        # ensure there are ncorrect elements in correct places
        if ncorrect == 0: 
        return list2
        rands = sample(list(dict_full)[0:nsize + 1], ncorrect + 1)
        for r in rands:
        list2[r] = list(dict_full)[r]
        return list2

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
        #eqn = []
        eqn = {}

        for i in range (ni):  #Inputs
            #eqn.append("INPUT(n{})\n".format(i))
            eqn[i] = f'x{i}'

    ##    for item in G[Lg-no:]:
    ##        eqn.append("OUTPUT(n{})\n".format(item))  #Outputs


        for key in nodes.keys():
            eqn[key] = None

        # Initialize flag for repeated traversing of the netlist
        flag = False
        while True:
            # Check if a node is still at level "None"
            if None in eqn.values():
                flag = True # Raise flag if any node is found "None"
            if flag == True:
                for key, value in nodes.items():
                    func = nf[value[0]]
                    val1 = eqn[value[1]]
                    val2 = eqn[value[2]]
                    
                    if val1!=None and val2!=None:
                        eqn[key] = f'({val1}{func}{val2})'
                flag = False
            else: # If all nodes have been expressed, break this loop
                break

        keys = list(eqn.keys())
        for key in keys:
            if key not in G[Lg-no:]:
                eqn.pop(key)
            

    ##    for key, value in nodes.items(): 
    ##        func = nf[value[0]]
    ##        eqn.append("x{} = (x{}{}x{})\n".format(key,value[1],func,value[2]))

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

    def get_cost(G):
        def normalize(val,Rmin,Rmax,Tmin,Tmax):
            return (((val-Rmin)/(Rmax-Rmin)*(Tmax-Tmin))+Tmin)

        def denormalize(val,Rmin,Rmax,Tmin,Tmax):
            return (((val-Tmin)/(Tmax-Tmin)*(Rmax-Rmin))+Rmin)

        eqn = decode(G)

        try:
            equation = eqn[G[Lg-no:][0]]
            correlm = get_correlm(equation)
        except:
            correlm = 1.0

        return(correlm, equation)

    def search():
        def levyFlight(u):
            return (math.pow(u,-1.0/3.0)) # Golden ratio = 1.62

        def randF():
            return (uniform(0.0001,0.9999))

        var_levy = []
        for i in range(1000):    
            var_levy.append(round(levyFlight(randF())))
        var_choice = choice

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
        for i in range(n):
            egg = random_solution()
            nest.append((egg,get_cost(egg)))

        nest.sort(key = itemgetter(1)) #Sorting -> same as: nest.sort(key = lambda x: x[1][0])
        
        global best_egg
        global k
        global dic_front
        ls_trace = []
        counter = 0
        # Main Loop
        for k in range(kmax+1):
            # Lay cuckoo eggs
            for i in range(n_cuckoo_eggs):
                index = randint(0,n-1)
                egg = deepcopy(nest[index]) # Pick an egg at random from the nest
                cuckoo = random_walker(egg[0],var_choice(var_levy)) # Lay a cuckoo egg from that egg
                cost_cuckoo = get_cost(cuckoo)
                if (cost_cuckoo[0] <= egg[1][0]): # Check if the cuckoo egg is better
                    nest[index] = (cuckoo,cost_cuckoo)

            nest.sort(key = itemgetter(1)) # Sorting

            # Store ratioA for trace
            ls_trace.append(nest[0][1][0])
                    
            if counter<rand_inject:
                # Genetically modify eggs with probability pa
                if n_replace>0:
                    children = offsprings(nest,k_way)
                    for i in range(n_replace):
                        nest[(n-1)-(i)] = (children[i],get_cost(children[i]))
                counter+=1
            else:
                counter = 0
                for i in range(n_replace):
                    egg = random_solution()
                    nest[(n-1)-(i)] = (egg,get_cost(egg))

            # Iterational printing
            if (k%print_after == 0):
                
                with open(f_trace,'a') as f:
                    for trace in ls_trace:
                        f.write(str(round(trace, 3))+'\n')
                ls_trace = [] # empty after dumping
                
                nest.sort(key = itemgetter(1)) # Rank nests and find current best
                best_egg = deepcopy(nest[0])
                cost = round(best_egg[1][0], 3)
                print(f'\n#{k}', f'{cost}')

                if (print_best==1):
                    print(best_egg[1][1])
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
    print("print_best: {}".format(print_best))
    print({'adjust':'x0', 'kk':'x1', 'rank1':'x2', 'rank2':'x3'})
    print('')


    #=================================
    # Another Miscellaneous
    #=================================
    n_cuckoo_eggs = round(pa*n)
    n_replace = round(pa*n)
    k_way = round(pa*n)
    f_trace = 'trace'

    #=================================
    # Execution
    #=================================
    ##geno = random_solution()
    ##pheno = decode(geno)
    ##print(pheno)
    ##
    ##cost = get_cost(geno)
    ##print(cost)


    search()




    print('\n\n\n\nDONE')
    ## That's it folks





        

    

