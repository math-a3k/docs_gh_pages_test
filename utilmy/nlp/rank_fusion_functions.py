import math
import sys

def isr(result_list,params):
    """function isr
    Args:
        result_list:   
        params:   
    Returns:
        
    """
    score = 0
    size = len(result_list)
    for doc_score, rank, engine in result_list:
        score += 1.0/math.pow(rank,2)
    return size * score
 
def log_isr(result_list,params):
    """function log_isr
    Args:
        result_list:   
        params:   
    Returns:
        
    """
    score = 0
    size = len(result_list)
    for doc_score, rank, engine in result_list:
        score += 1.0/math.pow(rank,2)
    return math.log(size) * score
    
def logn_isr(result_list,params):
    """function logn_isr
    Args:
        result_list:   
        params:   
    Returns:
        
    """
    score = 0
    size = len(result_list)
    for doc_score, rank, engine in result_list:
        score += 1.0/math.pow(rank,2)
    return math.log(size+0.01) * score

def expn_isr(result_list,params):
    """function expn_isr
    Args:
        result_list:   
        params:   
    Returns:
        
    """
    if len(params) > 0:
        k = float(params[0])
    else:
        k = 60.0
    if len(params) > 1:
        h = float(params[0])
    else:
        h = 60.0
    score = 0
    size = len(result_list)
    for doc_score, rank, engine in result_list:
        score += 1.0/math.pow(rank,2)
    return (h**(1/h))**(size-1) * score


def logn_rrf(result_list,params):
    """function logn_rrf
    Args:
        result_list:   
        params:   
    Returns:
        
    """
    if len(params) > 0:
        k = float(params[0])
    else:
        k = 60.0
    score = 0
    size = len(result_list)
    for doc_score, rank, engine in result_list:
        score += 1.0/(rank+k)
    return math.log(size+0.01) * score

def expn_rrf(result_list,params):
    """function expn_rrf
    Args:
        result_list:   
        params:   
    Returns:
        
    """
    if len(params) > 0:
        k = float(params[0])
    else:
        k = 60.0
    if len(params) > 1:
        h = float(params[0])
    else:
        h = 60.0
    score = 0
    size = len(result_list)
    for doc_score, rank, engine in result_list:
        score += 1.0/(rank+k)
    return (h**(1/h))**(size-1) * score

    
def rrf(result_list,params):
    """function rrf
    Args:
        result_list:   
        params:   
    Returns:
        
    """
    if len(params) > 0:
        k = float(params[0])
    else:
        k = 60.0
    score = 0
    for doc_score, rank, engine in result_list:
        score += 1.0/(rank+k)
    return score
    
def rr(result_list,params):
    """function rr
    Args:
        result_list:   
        params:   
    Returns:
        
    """
    score = 0
    for doc_score, rank, engine in result_list:
        score += 1.0/(rank)
    return score

###############################################################  

def votes(result_list,params):
    """function votes
    Args:
        result_list:   
        params:   
    Returns:
        
    """
    return len(result_list)
    
###############################################################  

def mnz(result_list,params):
    """function mnz
    Args:
        result_list:   
        params:   
    Returns:
        
    """
    score = 0
    for doc_score, rank, engine in result_list:
        score += doc_score
    return score * len(result_list)
    
def sum(result_list,params):
    """function sum
    Args:
        result_list:   
        params:   
    Returns:
        
    """
    score = 0
    for doc_score, rank, engine in result_list:
        score += doc_score
    return score

def max(result_list,params):
    """function max
    Args:
        result_list:   
        params:   
    Returns:
        
    """
    score = -sys.maxsize - 1
    for doc_score, rank, engine in result_list:
        if doc_score > score:
            score = doc_score
    return score

def min(result_list,params):
    """function min
    Args:
        result_list:   
        params:   
    Returns:
        
    """
    score = sys.maxsize
    for doc_score, rank, engine in result_list:
        if doc_score < score:
            score = doc_score
    return score

###############################################################  

def condor(doc_id_scores):
    """function condor
    Args:
        doc_id_scores:   
    Returns:
        
    """
    ranks = []
    for doc_id in doc_id_scores:
        ranks.append((doc_id, 1))   
    sorted_ranks = sorted(ranks, cmp=compareCondor, reverse=True)
    
    c_ranks = []
    
    rank = 1
    for (doc_id, score) in sorted_ranks:
        c_ranks.append((doc_id, 1.0/rank))
        rank += 1   
    
    return c_ranks

def compareCondor(item1, item2):
    """function compareCondor
    Args:
        item1:   
        item2:   
    Returns:
        
    """
    (doc_id1, score1) = item1
    (doc_id2, score2) = item2
    
    votes1 = 0
    votes2 = 0
    
    untested_engines = set()
    
    for (rank_score1, rank_rank1, rank_name1) in doc_id_scores[doc_id1]:
        if rank_name1 in untested_engines:
            untested_engines.remove(rank_name1)
        found_in_2 = False
        for (rank_score2, rank_rank2, rank_name2) in doc_id_scores[doc_id2]:
            if rank_name1 == rank_name2:
                found_in_2 = True
                if rank_rank1 < rank_rank2:
                    votes1 += 1
                elif rank_rank1 > rank_rank2:
                    votes2 += 1
            else:
                untested_engines.add(rank_name2)
        if not found_in_2:
            votes1 += 1

    votes2 += len(untested_engines)
    return votes1-votes2
