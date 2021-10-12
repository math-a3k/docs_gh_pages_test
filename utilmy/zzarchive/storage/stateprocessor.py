# -*- coding: utf-8 -*-
"""
Utilities to calculate States for Stock Selection 
"""


#####################################################################################
def sort(x, col, asc): return   util.sortcol(x, col, asc)
def perf(close, t0, t1):  return  100*( close2[:,t1] / close2[:,t0] -1)
def and2(tuple1):  return np.logical_and.reduce(tuple1)

def ff(x, symfull=symfull) : return util.find(x, symfull)  # 607  


def gap(close, t0, t1, lag):   
 ret= pf.getret_fromquotes(close[:,t0:t1], lag)
 rmin= 100*np.amin(ret, axis=1)
 return rmin

def process_stock(stkstr, show1=1) :
 stklist= stkstr.split(",") 
 for k,x in  enumerate(stklist) : stklist[k]= x.strip().replace('\n','')
 v= list(set(stklist))
 v.sort(); aux=""
 for x in v :  aux= aux+x+"," 
 if aux[0] == ',' : aux= aux[1:]
  
 if show1 : 
  print  aux + "\n", "   "
  print 'http://finviz.com/screener.ashx?v=211&t='+aux
 return v

def printn(ss, symfull=symfull, s1=s1) :
 ss= util.sortcol(ss, 71, asc=True) ;
 aux2=[];  aux=""; aux3=""
 for k in range(0, len(ss)) :
   kx= int(ss[k,0]); 
   aux= aux + symfull[kx] +","
   aux3=  aux3 + "'" +  symfull[kx] + "',"
   aux2.append(  symfull[kx] )
   print  kx, symfull[kx], round(s1[kx,70],2), round(s1[kx,71],2), round(s1[kx,45],1)

 #print "["+aux3[:-1]+"]", "\n"
 print '\n----------------------\n'
 print aux, "\n"
 print 'http://finviz.com/screener.ashx?v=211&t='+aux
 return aux2


def show(ll, s1=s1): 
 if type(ll)==str: ll= [ll]
 for x in ll :
  k=util.find(x, symfull)
  if k==-1 : print 'error' 
  else :
   print k, x, s1[k,45],  s1[k,42]  
   print "Trend:", s1[k,54] ,  s1[k,58] ,   s1[k,60] , s1[k,62], s1[k,64], s1[k,57] , s1[k,65]    #Trend t+5,10,15 
   print "Ret 5 days:",  s1[k,1], s1[k,24], s1[k,25], s1[k,26], s1[k,2]  #Return
   print "Daily 5 days:", s1[k,1], s1[k,27], s1[k,28], s1[k,29], s1[k,30]  #Return

   print "DistFromMinMax: ",  s1[k,68], s1[k,69]  
 
   print "Ret 20 days:",   s1[k,3],s1[k,4], s1[k,5],   s1[k,6] , s1[k,7],   s1[k,8]   #Return

   print "Trend Max: 200d",  s1[k,70], s1[k,71]  
   print "Trend Min: 200d",  s1[k,72], s1[k,73]  
   print "LowerBand, Price, TopBand 120 days", s1[k,91], s1[k,45], s1[k,90] 
   print "LowerBand, Price, TopBand 200 days", s1[k,94], s1[k,45], s1[k,93] 
   print "LowerBand, Price, TopBand 300 days", s1[k,97], s1[k,45], s1[k,96] 
   print "MA20,MA50, RMI ", s1[k,30], s1[k,31], s1[k,32] 
   
   print "Trend Max: 100d",  s1[k,74], s1[k,75]  
   print "Trend Min: 100d",  s1[k,76], s1[k,77], "\n-----"   

#################################################################################




#####################################################################################
######################  Decision Tree  For Stock Selection ##########################
def get_treeselect(stk, s1=s1, xnewdata=None, newsample=5, show1=1, nbtree=5, depthtree=10):
 ll1= process_stock(stk, show1)
 select1= [  util.find(x, symfull) for x in ll1]  
 Xtrain=  s1[:,1:]  
 Ytrain=  np.array([ 1 if  i in select1  else 0 for i in range(0, np.shape(s1)[0])] )

 if not xnewdata is None :
   for vv in xnewdata :
     Xtrain= np.row_stack((Xtrain, vv[:,1:]))
     ynew= np.ones(np.shape(vv)[0])
     Ytrain= np.concatenate((Ytrain, ynew))

 print np.max(Xtrain), np.min(Xtrain)

 clfrfmin, cmin, c0= util.sk_tree(Xtrain, Ytrain, nbtree, depthtree, 0)
 errmin=20; diversity= 0; divermax= newsample #max(10, np.sum(Ytrain) /3)
 for k in range(0,500):
   clfrf, c1, c0= util.sk_tree(Xtrain, Ytrain, nbtree,depthtree, 0)
   Ystock= clfrf.predict(Xtrain)

   if c1[0,1] >= diversity and c1[0,1] < divermax :  #choose more stock
     diversity= c1[0,1]
     if c1[1,0]  <= errmin :   #choose same stocks
       errmin= c1[1,0]; clfrfmin= clfrf; cmin= c1
       #print(cmin)
 print(cmin)
 return clfrfmin


def store_patternstate(tree, sym1, theme, symfull=symfull) :
 lstate=[]
 for x in sym1 :
   kid= util.find(x, symfull)
   lstate.append(s1[kid,:])

 lstate= np.array(lstate, dtype=np.float32)
 name1= 'stat_stk_pattern_'+theme+'_'+ str(dateref[-1])
 
 aux= (tree, dateref[-1],sym1, lstate) 
 util.save_obj(aux, name1)
 print name1


def load_patternstate(name1):
  tree, date, sym1, lstate= util.load_obj(name1)
  return tree, date, sym1, lstate
  

def get_stocklist(clf, s11, initial, show1=1):
 ll0= process_stock(initial, show1=0)
 Ystock= clf.predict(s11[:,1:])
 aux=''; laux=[]
 for k,x in enumerate(Ystock):
  if x != 0.0 : 
    aux= aux  + str(symfull[k]) +','
    laux.append(str(symfull[k]))    
  
 if show1: print'Full_List: ';  print aux
 aux2= list(set(laux).difference(ll0))
 aux2.sort()
 print '\nNew ones:';   print ",".join(list(aux2))
 aux3= ",".join(list(aux2))
 return aux3

#####################################################################################
#####################################################################################






############################################################################
#---------------------             --------------------







############################################################################













############################################################################
#---------------------             --------------------






























