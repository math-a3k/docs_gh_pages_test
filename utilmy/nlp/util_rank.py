# pylint: disable=C0103, R0914, R0201
HELP= """
Main module for ranking comparison , including rbo

https://github.com/changyaochen/rbo/blob/master/rbo/rbo.py
https://towardsdatascience.com/rbo-v-s-kendall-tau-to-compare-ranked-lists-of-items-8776c5182899
https://changyaochen.github.io/Comparing-two-ranked-lists/

"""
from typing import List, Optional, Union
import numpy as np
from tqdm import tqdm
import math, glob, os, sys, time

from utilmy.utilmy import log, log2


def test_all():
  test()


def test():
  S = [1, 2, 3]; T = [1, 3, 2]
  log( rbo.RankingSimilarity(S, T).rbo() )

  # Example usage
  rbo([1,2,3], [3,2,1]) # Output: 0.8550000000000001 
  
  

#################################################################################################
#################################################################################################
def rank_topk_check(dirin=None, dirout=None, nmax=3000, tag='fasttext'):      
    """ Compare ranking in "," format
        Compare id_list with ref_list  ranking and show some correlation numbers.
        id_list  : Vector based ranking  "," separated
        ref_list : reference ranking (ie co-count).

        id, id_list,  ref_list

    """
    from scipy.stats import kendalltau
    from utilmy.nlp import util_rank as rank1
    from utilmy import pd_read_file, pd_to_file

    flist = sorted(glob.glob( dirin  ))
    log(len(flist), str(flist)[:100])
    kk = 0
    for fi in flist :  
        df             = pd_read_file(fi)  
        df = df.iloc[:nmax,:]
        if 'ref_list' not in df.columns :
            df['ref_list'] = df['id'].apply( lambda x : db_cocount_name.get( int(x),"")  )                

        log(df)        
        df0   = [] 
        for i,x in df.iterrows():
            # if i > 10 : break
            xid  = x['id']                
            real = x['ref_list']

            if real != "" : 
               # log(i, xid ) 
               x2 = real.split(",")
               x1 = x['id_list'].split(",")
               ni = min( len(x1), len(x2) )
               x2 = x2[:ni]
               x1 = x1[:ni]

               ninter   = len( set(x1) & set(x2) )   ### Intersection elements
               corr     = kendalltau( x1, x2   )                    
               rbo      = rank1.rank_biased_overlap(x1, x2, p=0.8) 
               corrken2 = rank1.rank_topk_kendall( np.array(x1), np.array(x2), topk=50, p=0.8)

               df0.append([ i,tag,  xid, ninter,  corr.correlation, rbo, corrken2  ])
            if i % 10000 == 0 : log( df0[-1] if len(df0) >0 else '' )


        kk  = kk + 1
        df0 = pd.DataFrame(df0, columns = [ 'i', 'tag', 'id', 'n_intersection', 'corr_kendall', 'rank_rbo', 'corr_kendall_topk' ])
        pd_to_file( df0, dirout + f"/rank_check_{kk}.csv" )
        df0 = []


  

def rank_biased_overlap(list1, list2, p=0.9):
   """ rank based comparison between 2 lists.
   The parameter p is a tunable parameter in the range (0, 1) that can be used to determine
   the contribution of the top d ranks to the final value of the RBO similarity measure. 
   
   """
   # tail recursive helper function
   def helper(ret, i, d):
       l1 = set(list1[:i]) if i < len(list1) else set(list1)
       l2 = set(list2[:i]) if i < len(list2) else set(list2)
       a_d = len(l1.intersection(l2))/i
       term = math.pow(p, i) * a_d
       if d == i:
           return ret + term
       return helper(ret + term, i + 1, d)
   k = max(len(list1), len(list2))
   x_k = len(set(list1).intersection(set(list2)))
   summation = helper(0, 1, k)
   return ((float(x_k)/k) * math.pow(p, k)) + ((1-p)/p * summation)


def rbo_find_p():
  p = 0.9
  d = 10

  def sum_series(p, d):
     # tail recursive helper function
     def helper(ret, p, d, i):
         term = math.pow(p, i)/i
         if d == i:
             return ret + term
         return helper(ret + term, p, d, i+1)
     return helper(0, p, d, 1)

  wrbo1_d = 1 - math.pow(p, d-1) + (((1-p)/p) * d *(np.log(1/(1-p)) - sum_series(p, d-1)))
  print(wrbo1_d) # Output: 0.855585446747351

  
  
#################################################################################################################   
def rank_topk_kendall(a:list, b:list, topk=5,p=0): #zero is equal 1 is max distance, compare with 1-scipy.stats.kendalltau(a,b)/2+1/2
    """
    kendall_top_k(np.array,np.array,k,p)
    #zero is equal 1 is max distance, compare with 1-scipy.stats.kendalltau(a,b)/2+1/2
    This function generalise kendall-tau as defined in [1] Fagin, Ronald, Ravi Kumar, and D. Sivakumar. "Comparing top k lists." SIAM Journal on Discrete Mathematics 17.1 (2003): 134-160.
    It returns a distance: 0 for identical (in the sense of top-k) lists and 1 if completely different.
    Example:
        Simply call it with two same-length arrays of ratings (or also rankings), length of the top elements k (default is the maximum length possible), and p (default is 0, see [1]) as parameters:
            $ a = np.array([1,2,3,4,5])
            $ b = np.array([5,4,3,2,1])
            $ kendall_top_k(a,b,k=4)
     Author: Alessandro Checco  https://github.com/AlessandroChecco       
            
    """
    k = topk
    import numpy as np
    import scipy.stats as stats
    import scipy.special as special

    if k is None:
        k = len(a)
    if len(a) != len(b) :
        raise NameError('The two arrays need to have same lengths')
    k = min(k,a.size)
    a_top_k = np.argpartition(a,-k)[-k:]
    b_top_k = np.argpartition(b,-k)[-k:]
    common_items = np.intersect1d(a_top_k,b_top_k)
    only_in_a = np.setdiff1d(a_top_k, common_items)
    only_in_b = np.setdiff1d(b_top_k, common_items)
    kendall = (1 - (stats.kendalltau(a[common_items], b[common_items])[0]/2+0.5)) * (common_items.size**2) #case 1
    if np.isnan(kendall): # degenerate case with only one item (not defined by Kendall)
        kendall = 0
    for i in common_items: #case 2
        for j in only_in_a:
            if a[i] < a[j]:
                kendall += 1
        for j in only_in_b:
            if b[i] < b[j]:
                kendall += 1
    kendall += 2*p * special.binom(k-common_items.size,2)     #case 4
    kendall /= ((only_in_a.size + only_in_b.size + common_items.size)**2 ) #normalization
    return kendall
    
  
  
  
#############################################################################
class RankingSimilarity:
    """
    This class will include some similarity measures between two different
    ranked lists.
    """

    def __init__(
            self,
            S: Union[List, np.ndarray],
            T: Union[List, np.ndarray],
            verbose=False):
        """
        Initialize the object with the required lists.
        Examples of lists:
        S = ["a", "b", "c", "d", "e"]
        T = ["b", "a", 1, "d"]
        Both lists reflect the ranking of the items of interest, for example,
        list S tells us that item "a" is ranked first, "b" is ranked second,
        etc.
        Args:
            S, T (list or numpy array): lists with alphanumeric elements. They
                could be of different lengths. Both of the them should be
                ranked, i.e., each element"s position reflects its respective
                ranking in the list. Also we will require that there is no
                duplicate element in each list.
            verbose (bool). If True, print out intermediate results.
                Default to False.
        """

        assert type(S) in [list, np.ndarray]
        assert type(T) in [list, np.ndarray]

        assert len(S) == len(set(S))
        assert len(T) == len(set(T))

        self.S, self.T = S, T
        self.N_S, self.N_T = len(S), len(T)
        self.verbose = verbose
        self.p = 0.5  # just a place holder


    def assert_p(self, p: float) -> None:
        """Make sure p is between (0, 1), if so, assign it to self.p.
        Args:
            p (float): The value p.
        """
        assert 0.0 < p < 1.0, "p must be between (0, 1)"
        self.p = p


    def _bound_range(self, value: float):
        """Bounds the value to [0.0, 1.0].
        """

        try:
            assert (0 <= value <= 1 or np.isclose(1, value))
            return value

        except AssertionError:
            print("Value out of [0, 1] bound, will bound it.")
            larger_than_zero = max(0.0, value)
            less_than_one = min(1.0, larger_than_zero)
            return less_than_one


    def rbo(
            self,
            k: Optional[float] = None,
            p: float = 1.0,
            ext: bool = False):
        """
        This the weighted non-conjoint measures, namely, rank-biased overlap.
        Unlike Kendall tau which is correlation based, this is intersection
        based.
        The implementation if from Eq. (4) or Eq. (7) (for p != 1) from the
        RBO paper: http://www.williamwebber.com/research/papers/wmz10_tois.pdf
        If p=1, it returns to the un-bounded set-intersection overlap,
        according to Fagin et al.
        https://researcher.watson.ibm.com/researcher/files/us-fagin/topk.pdf
        The fig. 5 in that RBO paper can be used as test case.
        Note there the choice of p is of great importance, since it
        essentically control the "top-weightness". Simply put, to an extreme,
        a small p value will only consider first few items, whereas a larger p
        value will consider more itmes. See Eq. (21) for quantitative measure.
        Args:
            k (int), default None: The depth of evaluation.
            p (float), default 1.0: Weight of each agreement at depth d:
                p**(d-1). When set to 1.0, there is no weight, the rbo returns
                to average overlap.
            ext (Boolean) default False: If True, we will extrapolate the rbo,
                as in Eq. (23)
        Returns:
            The rbo at depth k (or extrapolated beyond)
        """

        if not self.N_S and not self.N_T:
            return 1  # both lists are empty

        if not self.N_S or not self.N_T:
            return 0  # one list empty, one non-empty

        if k is None:
            k = float("inf")
        k = min(self.N_S, self.N_T, k)

        # initialize the agreement and average overlap arrays
        A, AO = [0] * k, [0] * k
        if p == 1.0:
            weights = [1.0 for _ in range(k)]
        else:
            self.assert_p(p)
            weights = [1.0 * (1 - p) * p**d for d in range(k)]

        # using dict for O(1) look up
        S_running, T_running = {self.S[0]: True}, {self.T[0]: True}
        A[0] = 1 if self.S[0] == self.T[0] else 0
        AO[0] = weights[0] if self.S[0] == self.T[0] else 0

        for d in tqdm(range(1, k), disable=~self.verbose):

            tmp = 0
            # if the new item from S is in T already
            if self.S[d] in T_running:
                tmp += 1
            # if the new item from T is in S already
            if self.T[d] in S_running:
                tmp += 1
            # if the new items are the same, which also means the previous
            # two cases did not happen
            if self.S[d] == self.T[d]:
                tmp += 1

            # update the agreement array
            A[d] = 1.0 * ((A[d - 1] * d) + tmp) / (d + 1)

            # update the average overlap array
            if p == 1.0:
                AO[d] = ((AO[d - 1] * d) + A[d]) / (d + 1)
            else:  # weighted average
                AO[d] = AO[d - 1] + weights[d] * A[d]

            # add the new item to the running set (dict)
            S_running[self.S[d]] = True
            T_running[self.T[d]] = True

        if ext and p < 1:
            return self._bound_range(AO[-1] + A[-1] * p**k)

        return self._bound_range(AO[-1])


    def rbo_ext(self, p=0.98):
        """
        This is the ultimate implementation of the rbo, namely, the
        extrapolated version. The corresponding formula is Eq. (32) in the rbo
        paper.
        """

        self.assert_p(p)

        if not self.N_S and not self.N_T:
            return 1  # both lists are empty

        if not self.N_S or not self.N_T:
            return 0  # one list empty, one non-empty

        # since we are dealing with un-even lists, we need to figure out the
        # long (L) and short (S) list first. The name S might be confusing
        # but in this function, S refers to short list, L refers to long list
        if len(self.S) > len(self.T):
            L, S = self.S, self.T
        else:
            S, L = self.S, self.T

        s, l = len(S), len(L)  # noqa

        # initialize the overlap and rbo arrays
        # the agreement can be simply calculated from the overlap
        X, A, rbo = [0] * l, [0] * l, [0] * l

        # first item
        S_running, L_running = {S[0]}, {L[0]}  # for O(1) look up
        X[0] = 1 if S[0] == L[0] else 0
        A[0] = X[0]
        rbo[0] = 1.0 * (1 - p) * A[0]

        # start the calculation
        disjoint = 0
        ext_term = A[0] * p

        for d in tqdm(range(1, l), disable=~self.verbose):
            if d < s:  # still overlapping in length

                S_running.add(S[d])
                L_running.add(L[d])

                # again I will revoke the DP-like step
                overlap_incr = 0  # overlap increment at step d

                # if the new items are the same
                if S[d] == L[d]:
                    overlap_incr += 1
                else:
                    # if the new item from S is in L already
                    if S[d] in L_running:
                        overlap_incr += 1
                    # if the new item from L is in S already
                    if L[d] in S_running:
                        overlap_incr += 1

                X[d] = X[d - 1] + overlap_incr
                # Eq. (28) that handles the tie. len() is O(1)
                A[d] = 2.0 * X[d] / (len(S_running) + len(L_running))
                rbo[d] = rbo[d - 1] + 1.0 * (1 - p) * (p**d) * A[d]

                ext_term = 1.0 * A[d] * p**(d + 1)  # the extrapolate term

            else:  # the short list has fallen off the cliff
                L_running.add(L[d])  # we still have the long list

                # now there is one case
                overlap_incr = 1 if L[d] in S_running else 0

                X[d] = X[d - 1] + overlap_incr
                A[d] = 1.0 * X[d] / (d + 1)
                rbo[d] = rbo[d - 1] + 1.0 * (1 - p) * (p**d) * A[d]

                X_s = X[s - 1]  # this the last common overlap
                # second term in first parenthesis of Eq. (32)
                disjoint += 1.0 * (1 - p) * (p**d) * \
                    (X_s * (d + 1 - s) / (d + 1) / s)
                ext_term = 1.0 * ((X[d] - X_s) / (d + 1) + X[s - 1] / s) * \
                    p**(d + 1)  # last term in Eq. (32)

        return self._bound_range(rbo[-1] + disjoint + ext_term)


    def top_weightness(
            self,
            p: Optional[float] = None,
            d: Optional[int] = None):
        """
        This function will evaluate the degree of the top-weightness of the
        rbo. It is the implementation of Eq. (21) of the rbo paper.
        As a sanity check (per the rbo paper),
        top_weightness(p=0.9, d=10) should be 86%
        top_weightness(p=0.98, d=50) should be 86% too
        Args:
            p (float), default None: A value between zero and one.
            d (int), default None: Evaluation depth of the list.
        Returns:
            A float between [0, 1], that indicates the top-weightness.
        """

        # sanity check
        self.assert_p(p)

        if d is None:
            d = min(self.N_S, self.N_T)
        else:
            d = min(self.N_S, self.N_T, int(d))

        if d == 0:
            top_w = 1
        elif d == 1:
            top_w = 1 - 1 + 1.0 * (1 - p) / p * (np.log(1.0 / (1 - p)))
        else:
            sum_1 = 0
            for i in range(1, d):
                sum_1 += 1.0 * p**(i) / i
            top_w = 1 - p**(i) + 1.0 * (1 - p) / p * (i + 1) * \
                (np.log(1.0 / (1 - p)) - sum_1)  # here i == d-1

        if self.verbose:
            print("The first {} ranks have {:6.3%} of the weight of "
                  "the evaluation.".format(d, top_w))

        return self._bound_range(top_w)
