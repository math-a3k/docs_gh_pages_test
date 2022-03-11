# -*- coding: utf-8 -*-
"""
Customize Risk Functions 
ss: State
tr: Trigger
t
risk0: previous Risk Level
"""
import portfolio as pf, numpy as np








def mapping_calc_risk_elvis_v03(ss, tr, t, riskout) :
     ''' ss: state,  tr:trigger level,  risk0: previous risk value '''
     # Pattern Regime:   Drop --> Crash --> OVerSold --> Recovery ---> Bull ---> Drop 
     risk0= riskout[t-1, 1]      
     return mapping_calc_risk_v01(ss, tr, t, risk0) 



#Risk Indicator 2, Smootht the noise / false signal
'''
Random Forest can be used to generate those rules, by using human mapping

'''
def mapping_calc_risk_v02(ss, tr, t, risk0) :
     ''' ss: state,  tr:trigger level,  risk0: previous risk value '''
     # Pattern Regime:   Drop --> Crash --> OVerSold --> Recovery ---> Bull ---> Drop

     # Super Bear Protection
     if (ss[t, 0] < -13.0 and ss[t, 2] < -18.0) or (ss[t, 0] < -9.0 and ss[t, 2] < -26.0) or ( ss[t, 0] < -5.0 and ss[t, 2] < -35.0   ) or (risk0==6 and ss[t, 0] < -8.0 and ss[t, 2] < -16.0) :
         risk = 6.0

     #Crash Protection
     elif ((ss[t, 0] <= 0.001 and ss[t-1, 0] >= ss[t, 0]-0.01 ) or  ss[t, 0] <= 0.2  ) and  0.333*( ss[t-2, 1] + ss[t-1, 1] + ss[t, 1]) < 3.5 :
       
        #Recovery from botoom, over-sold position
        if (risk0 == 2.0 or risk0 == 0.0 ) and ss[t, 0] <= -8.0 and ss[t, 2] < -16.0  :
            risk= 0.0   
          
        elif risk0 == 2.0  and ss[t, 0] == 0.0  and ss[t, 1] > 0.0 and ss[t-1, 1] < 0.01 and ss[t-2, 1] < 0.01  :
            risk= 0.0

        #Drawdown reduces
        elif (risk0 == 0.0 or risk0==6 ) and ss[t, 2]  >  0.333 *( ss[t-1, 2] + ss[t-2, 2] + ss[t-3, 2]) - 1.5   :
            risk= 0.0   

        elif ss[t, 2] > -1.0  :      #Drawdown is too small, do NOT move to Bear Protection
            risk = risk0

        # Crash Protection
        else :
            risk = 2.0

     #Drawdown from Bull position
     elif (risk0==4.0 or risk0==2.0 or risk0 == 0.0 or risk0== 6.0) and ss[t, 2] < -5.9  :
            risk = 2.0 
     else:
            risk = 4.0  # Bull Case
     
     return risk




#Risk Indicator 1
def mapping_calc_risk_v01(ss, tr, t, risk0) :
     ''' ss: state,  tr:trigger level,  risk0: previous risk value '''
     # Pattern Regime:   Drop --> Crash --> OVerSold --> Recovery ---> Bull ---> Drop 
     if (ss[t, 0] < -13.0 and ss[t, 2] < -18.0) or (ss[t, 0] < -9.0 and ss[t, 2] < -25.0) :  
         risk = 6.0  # Super Bear Protection

     elif (ss[t, 0] <= 0.001 and ss[t-1, 0] >= ss[t, 0]-0.01 ) or  ss[t, 0] <= -1.0  :  #Crash Protection
       
        if ss[t, 0] <= -10.0 and ss[t, 2] < -15.0 and (risk0 == 2.0 or risk0 == 0.0 ) :  
            risk= 0.0   #Recovery from botoom, over-sold position
          
        elif ss[t, 0] == 0.0  and ss[t, 1] > 0.0 and ss[t-1, 1] < 0.01 and ss[t-2, 1] < 0.01  and risk0 == 2.0 :
            risk= 0.0   #Recovery from bottom
           
        else :    risk = 2.0  # Crash Protection

     elif (risk0==4.0 or risk0==2.0) and ss[t, 2] < -7.0  :  #Drawdown from Bull position
            risk = 2.0 
     else:
            risk = 4.0  # Bull Case
     return risk





def mapping_risk_ww_v01(risk, wwmat, ww2):
      """function mapping_risk_ww_v01
      Args:
          risk:   
          wwmat:   
          ww2:   
      Returns:
          
      """
      if   risk== 6.0 :    return ww2                               # Super Bear
      elif risk== 0.0 :    vv= wwmat[:,0];   ww2 = vv / np.sum(vv)  
      elif risk== 2.0 :    vv= wwmat[:,1];   ww2 = vv / np.sum(vv)
      elif risk== 4.0 :    vv= wwmat[:,2];   ww2 = vv / np.sum(vv)  
      elif risk== 8.0 :    ww2 = np.array([ 0.0, 0.1, 0.0, 0.2, 0.3, 0.0, 0.0, 0.0, 0.4 ])
      return ww2  

# Initial part
def mapping_calc_risk_v00(self, ss, tr, t, risk0) :
     """function mapping_calc_risk_v00
     Args:
         self:   
         ss:   
         tr:   
         t:   
         risk0:   
     Returns:
         
     """
     # Pattern Regime:   Drop --> Crash --> OVerSold --> Recovery ---> Bull ---> Drop 
     if  ss[t,0] < tr[0] and ss[t,2] < tr[1] :
         risk= 6.0    # Super Bear Protection
      
     # elif  ss[t,5] > 95.0      :  # 35  10
     #    risk= 8.0    # Mean reversion period
     #    self.tstart= t
 
     # elif  (risk0==8.0   and t-self.tstart < 10  )    :  # 45   10
     #    risk= 8.0    # Mean reversion period

     elif (ss[t,2] < tr[2] and risk0==2.0) or (tr[3] < ss[t,1] < tr[4] and risk0==0.0  )  :
         risk= 0.0    # Bottom oversold

     elif ((ss[t, 0] <= tr[5]) or (risk0 == 4.0 and ss[t, 2] < tr[6]) or (ss[t, 2] < tr[7])):
         risk = 2.0  # Crash Protection

     else :
         risk= 4.0    # Bull Case 
     return risk         




   
def getweight(ww,size=(9,3), norm=1):
 """function getweight
 Args:
     ww:   
     size:   
     3:   
 Returns:
     
 """
 ww2= np.reshape(ww, size)
 if norm== 1 : ww2= ww2/ np.sum(ww2, axis=0)
 return ww2


def fun_obj(vv, ext) :
  """function fun_obj
  Args:
      vv:   
      ext:   
  Returns:
      
  """
  volta=    pf.folio_volta(ext[0], vv[0], int(vv[1]), ext[1])
  vol= pf.volhisto_fromprice(volta,-1, len(volta))
  return -volta[-1] / vol






############################################################################
#---------------------             --------------------







############################################################################











############################################################################
#---------------------             --------------------




############################################################################





















############################################################################
#---------------------             --------------------






























