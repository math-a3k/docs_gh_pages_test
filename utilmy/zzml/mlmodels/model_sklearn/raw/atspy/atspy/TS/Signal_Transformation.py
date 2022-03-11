# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np

from . import Utils as tsutil


def testTransform_one_seed(tr1 , seed_value):
    """function testTransform_one_seed
    Args:
        tr1:   
        seed_value:   
    Returns:
        
    """
    df = pd.DataFrame();
    np.random.seed(seed_value)
    df['A'] = np.random.normal(0, 1.0, 10);
    # df['A'] = range(1, 6000);
    sig = df['A'];

    tr1.mOriginalSignal = "selfTestSignal";
    tr1.fit(sig)
    sig1 = tr1.apply(sig);
    sig2 = tr1.invert(sig1)
    # print(sig)
    # print(sig1)
    # print(sig2)
    n = np.linalg.norm(sig2 - sig)
    lEps = 1e-7
    if(n > lEps):
        print("'" + tr1.get_name("Test") + "'" , " : ", n)
        print(sig.values)
        print(sig1.values)
        print(sig2.values)    

    assert(n <= lEps)    


def testTransform(tr1):
    """function testTransform
    Args:
        tr1:   
    Returns:
        
    """
    for seed_value in range(0,10,100):
        testTransform_one_seed(tr1, seed_value)

class cAbstractSignalTransform:
    def __init__(self):
        """ cSignalTransform_RelativeDifferencing:__init__
        Args:
        Returns:
           
        """
        """ cSignalTransform_None:__init__
        Args:
        Returns:
           
        """
        """ cSignalTransform_Logit:__init__
        Args:
        Returns:
           
        """
        """ cSignalTransform_Fisher:__init__
        Args:
        Returns:
           
        """
        """ cSignalTransform_Differencing:__init__
        Args:
        Returns:
           
        """
        """ cSignalTransform_Anscombe:__init__
        Args:
        Returns:
           
        """
        """ cSignalTransform_Accumulate:__init__
        Args:
        Returns:
           
        """
        """ cAbstractSignalTransform:__init__
        Args:
        Returns:
           
        """
        self.mOriginalSignal = None;
        self.mComplexity = None;
        self.mScaling = None;
        self.mDebug = False;
        pass

    def is_applicable(self, sig):
        """ cSignalTransform_Quantize:is_applicable
        Args:
            sig:     
        Returns:
           
        """
        """ cSignalTransform_Logit:is_applicable
        Args:
            sig:     
        Returns:
           
        """
        """ cAbstractSignalTransform:is_applicable
        Args:
            sig:     
        Returns:
           
        """
        return True;

    def fit_scaling_params(self, sig):
        """ cAbstractSignalTransform:fit_scaling_params
        Args:
            sig:     
        Returns:
           
        """
        if(self.mScaling is not None):
            # self.mMeanValue = np.mean(sig);
            # self.mStdValue = np.std(sig);
            self.mMinValue = np.min(sig);
            self.mMaxValue = np.max(sig);
            self.mDelta = self.mMaxValue - self.mMinValue;
            eps = 1.0e-10
            if(self.mDelta < eps):
                self.mDelta = eps;
        else:
            return sig;

    def scale_value(self, x):
        """ cAbstractSignalTransform:scale_value
        Args:
            x:     
        Returns:
           
        """
        return (x - self.mMinValue) / self.mDelta;

    def scale_signal(self, sig):
        """ cAbstractSignalTransform:scale_signal
        Args:
            sig:     
        Returns:
           
        """
        if(self.mScaling is not None):
            # print("SCALE_START", sig.values[1:5]);
            sig1 = sig.apply(self.scale_value);
            # print("SCALE_END", sig1.values[1:5]);
            return sig1;
        else:
            return sig;

    def rescale_value(self, x):
        """ cAbstractSignalTransform:rescale_value
        Args:
            x:     
        Returns:
           
        """
        return self.mMinValue + x * self.mDelta;
        

    def rescale_signal(self, sig1):
        """ cAbstractSignalTransform:rescale_signal
        Args:
            sig1:     
        Returns:
           
        """
        if(self.mScaling is not None):
            # print("RESCALE_START", sig1.values[1:5]);
            sig = sig1.apply(self.rescale_value);
            # print("RESCALE_END", sig.values[1:5]);
            return sig;
        else:
            return sig1;

    def fit(self , sig):
        """ cAbstractSignalTransform:fit
        Args:
            sig:     
        Returns:
           
        """
        # print("FIT_START", self.mOriginalSignal, sig.values[1:5]);
        self.fit_scaling_params(sig);
        sig1 = self.scale_signal(sig);
        self.specific_fit(sig1);
        # print("FIT_END", self.mOriginalSignal, sig1.values[1:5]);
        pass

    def apply(self, sig):
        """ cAbstractSignalTransform:apply
        Args:
            sig:     
        Returns:
           
        """
        # print("APPLY_START", self.mOriginalSignal, sig.values[1:5]);
        sig1 = self.scale_signal(sig);
        sig2 = self.specific_apply(sig1);
        # print("APPLY_END", self.mOriginalSignal, sig2.values[1:5]);
        if(self.mDebug):
            self.check_not_nan(sig2 , "transform_apply");
        return sig2;

    def invert(self, sig1):
        """ cAbstractSignalTransform:invert
        Args:
            sig1:     
        Returns:
           
        """
        # print("INVERT_START", self.mOriginalSignal, sig1.values[1:5]);
        sig2 = self.specific_invert(sig1);
        rescaled_sig = self.rescale_signal(sig2);
        # print("INVERT_END", self.mOriginalSignal, rescaled_sig.values[1:5]);
        return rescaled_sig;


    def transformDataset(self, df, isig):
        """ cAbstractSignalTransform:transformDataset
        Args:
            df:     
            isig:     
        Returns:
           
        """
        df[self.get_name(isig)] = self.apply(df[isig])
        return df;

    def test(self):
        """ cAbstractSignalTransform:test
        Args:
        Returns:
           
        """
        # import copy;
        # tr1 = copy.deepcopy(self);
        # testTransform(tr1);
        pass

    def dump_apply_invert(self, df_before_apply, df_after_apply):
        """ cAbstractSignalTransform:dump_apply_invert
        Args:
            df_before_apply:     
            df_after_apply:     
        Returns:
           
        """
        df = pd.DataFrame();
        df['before_apply'] = df_before_apply;
        df['after_apply'] = df_after_apply;
        print("dump_apply_invert_head", df.head());
        print("dump_apply_invert_tail", df.tail());
        
    def check_not_nan(self, sig , name):
        """ cAbstractSignalTransform:check_not_nan
        Args:
            sig:     
            name:     
        Returns:
           
        """
        if(np.isnan(sig).any()):
            print("TRANSFORMATION_RESULT_WITH_NAN_IN_SIGNAL" , sig);
            raise tsutil.Internal_PyAF_Error("Invalid transformation for column '" + name + "'");
        pass


class cSignalTransform_None(cAbstractSignalTransform):

    def __init__(self):
        cAbstractSignalTransform.__init__(self);
        self.mFormula = "NoTransf";
        self.mComplexity = 0;
        pass

    def get_name(self, iSig):
        """ cSignalTransform_RelativeDifferencing:get_name
        Args:
            iSig:     
        Returns:
           
        """
        """ cSignalTransform_Quantize:get_name
        Args:
            iSig:     
        Returns:
           
        """
        """ cSignalTransform_None:get_name
        Args:
            iSig:     
        Returns:
           
        """
        """ cSignalTransform_Logit:get_name
        Args:
            iSig:     
        Returns:
           
        """
        """ cSignalTransform_Differencing:get_name
        Args:
            iSig:     
        Returns:
           
        """
        """ cSignalTransform_BoxCox:get_name
        Args:
            iSig:     
        Returns:
           
        """
        """ cSignalTransform_Accumulate:get_name
        Args:
            iSig:     
        Returns:
           
        """
        return "_" + str(iSig);
    
    def specific_fit(self , sig):
        """ cSignalTransform_Quantize:specific_fit
        Args:
            sig:     
        Returns:
           
        """
        """ cSignalTransform_None:specific_fit
        Args:
            sig:     
        Returns:
           
        """
        """ cSignalTransform_Fisher:specific_fit
        Args:
            sig:     
        Returns:
           
        """
        """ cSignalTransform_Anscombe:specific_fit
        Args:
            sig:     
        Returns:
           
        """
        """ cSignalTransform_Accumulate:specific_fit
        Args:
            sig:     
        Returns:
           
        """
        pass
    
    def specific_apply(self, df):
        """ cSignalTransform_RelativeDifferencing:specific_apply
        Args:
            df:     
        Returns:
           
        """
        """ cSignalTransform_Quantize:specific_apply
        Args:
            df:     
        Returns:
           
        """
        """ cSignalTransform_None:specific_apply
        Args:
            df:     
        Returns:
           
        """
        """ cSignalTransform_Logit:specific_apply
        Args:
            df:     
        Returns:
           
        """
        """ cSignalTransform_Differencing:specific_apply
        Args:
            df:     
        Returns:
           
        """
        """ cSignalTransform_BoxCox:specific_apply
        Args:
            df:     
        Returns:
           
        """
        return df;
    
    def specific_invert(self, df):
        """ cSignalTransform_RelativeDifferencing:specific_invert
        Args:
            df:     
        Returns:
           
        """
        """ cSignalTransform_Quantize:specific_invert
        Args:
            df:     
        Returns:
           
        """
        """ cSignalTransform_None:specific_invert
        Args:
            df:     
        Returns:
           
        """
        """ cSignalTransform_Logit:specific_invert
        Args:
            df:     
        Returns:
           
        """
        """ cSignalTransform_Differencing:specific_invert
        Args:
            df:     
        Returns:
           
        """
        """ cSignalTransform_BoxCox:specific_invert
        Args:
            df:     
        Returns:
           
        """
        """ cSignalTransform_Accumulate:specific_invert
        Args:
            df:     
        Returns:
           
        """
        return df;

        

class cSignalTransform_Accumulate(cAbstractSignalTransform):

    def __init__(self):
        cAbstractSignalTransform.__init__(self);
        self.mFormula = "Integration";
        self.mComplexity = 1;
        pass

    def get_name(self, iSig):
        return "CumSum_" + str(iSig);
    
    def specific_fit(self , sig):
        pass
    
    def specific_apply(self, sig):
        """ cSignalTransform_Fisher:specific_apply
        Args:
            sig:     
        Returns:
           
        """
        """ cSignalTransform_Anscombe:specific_apply
        Args:
            sig:     
        Returns:
           
        """
        """ cSignalTransform_Accumulate:specific_apply
        Args:
            sig:     
        Returns:
           
        """
        return sig.cumsum(axis = 0)
    
    def specific_invert(self, df):
        df_orig = df - df.shift(1);
        df_orig.iloc[0] = df.iloc[0];
        return df_orig;


class cSignalTransform_Quantize(cAbstractSignalTransform):

    def __init__(self, iQuantiles):
        """ cSignalTransform_Quantize:__init__
        Args:
            iQuantiles:     
        Returns:
           
        """
        cAbstractSignalTransform.__init__(self);
        self.mQuantiles = iQuantiles;
        self.mFormula = "Quantization";
        self.mComplexity = 2;
        pass

    def get_name(self, iSig):
        return "Quantized_" + str(self.mQuantiles) + "_" + str(iSig);

    def is_applicable(self, sig):
        N = sig.shape[0];
        if(N < (5 * self.mQuantiles)) :
            return False;
        return True;
    
    def specific_fit(self , sig):
        Q = self.mQuantiles;
        q = pd.Series(range(0,Q)).apply(lambda x : sig.quantile(x/Q))
        self.mCurve = q.to_dict()
        (self.mMin, self.mMax) = (min(self.mCurve.keys()), max(self.mCurve.keys()))
        pass

    def signal2quant(self, x):
        """ cSignalTransform_Quantize:signal2quant
        Args:
            x:     
        Returns:
           
        """
        curve = self.mCurve;
        return min(curve.keys(), key=lambda y:abs(float(curve[y])-x))
    
    def specific_apply(self, df):
        lSignal_Q = df.apply(self.signal2quant);
        return lSignal_Q;

    def quant2signal(self, x):
         """ cSignalTransform_Quantize:quant2signal
         Args:
             x:     
         Returns:
            
         """
         curve = self.mCurve;
         key = int(x);
         if(key >= self.mMax):
             key = self.mMax;
         if(key <= self.mMin):
             key = self.mMin;            
         val = curve[key]
         return val;

    def specific_invert(self, df):
        lSignal = df.apply(self.quant2signal);
        return lSignal;


class cSignalTransform_BoxCox(cAbstractSignalTransform):

    def __init__(self, iLambda):
        """ cSignalTransform_BoxCox:__init__
        Args:
            iLambda:     
        Returns:
           
        """
        cAbstractSignalTransform.__init__(self);
        self.mFormula = "BoxCox";
        self.mLambda = iLambda;
        self.mComplexity = 2;
        self.mScaling = True;
        pass

    def get_name(self, iSig):
        return "Box_Cox_" + str(self.mLambda) + "_" + str(iSig);

    def specific_fit(self, sig):
        """ cSignalTransform_RelativeDifferencing:specific_fit
        Args:
            sig:     
        Returns:
           
        """
        """ cSignalTransform_Logit:specific_fit
        Args:
            sig:     
        Returns:
           
        """
        """ cSignalTransform_Differencing:specific_fit
        Args:
            sig:     
        Returns:
           
        """
        """ cSignalTransform_BoxCox:specific_fit
        Args:
            sig:     
        Returns:
           
        """
        self.mFormula = "BoxCox(Lambda=" + str(self.mLambda) + ")";
        pass
    

    def specific_apply(self, df):
        lEps = 1e-20
        log_df = df.apply(lambda x : np.log(max(x , lEps)));
        if(abs(self.mLambda) <= 0.01):
            return log_df;
        return (np.exp(log_df * self.mLambda) - 1) / self.mLambda;

    def invert_value(self, y):
        """ cSignalTransform_BoxCox:invert_value
        Args:
            y:     
        Returns:
           
        """
        x = y;
        lEps = 1e-20
        x1 = np.log(max(self.mLambda * x + 1, lEps)) / self.mLambda;
        return np.exp(x1.clip(-20, 20)) ;        
    
    def specific_invert(self, df):
        if(abs(self.mLambda) <= 0.001):
            df_orig = np.exp(df.clip(-20, 20));
            return df_orig;
        df_pos = df.apply(self.invert_value);
        return df_pos;



class cSignalTransform_Differencing(cAbstractSignalTransform):

    def __init__(self):
        cAbstractSignalTransform.__init__(self);
        self.mFirstValue = None;
        self.mFormula = "Difference";
        self.mComplexity = 1;
        pass

    def get_name(self, iSig):
        return "Diff_" + str(iSig);

    def specific_fit(self, sig):
        # print(sig.head());
        self.mFirstValue = sig.iloc[0];
        pass
    

    def specific_apply(self, df):
        df_shifted = df.shift(1)
        df_shifted.iloc[0] = self.mFirstValue;
        return (df - df_shifted);
    
    def specific_invert(self, df):
        df_orig = df.cumsum();
        df_orig = df_orig + self.mFirstValue;
        return df_orig;


class cSignalTransform_RelativeDifferencing(cAbstractSignalTransform):

    def __init__(self):
        cAbstractSignalTransform.__init__(self);
        self.mFirstValue = None;
        self.mFormula = "RelativeDifference";
        self.mComplexity = 1;
        self.mScaling = True;
        pass

    def get_name(self, iSig):
        return "RelDiff_" + str(iSig);

    def specific_fit(self, sig):
        self.mFirstValue = sig.iloc[0];
        pass

    def specific_apply(self, df):
        lEps = 1e-8
        # print("RelDiff_apply_DEBUG_START" , self.mFirstValue, df.values[0:10]);
        df1 = df.apply(lambda x : x if (abs(x) > lEps) else lEps)
        df_shifted = df1.shift(1)
        # df_shifted[df_shifted <= lEps] = lEps
        rate = (df1 - df_shifted) / df_shifted
        rate.iloc[0] = 0.0;
        # print(df1)
        # print(df_shifted)
        rate = rate.clip(-1.0e+8 , +1.0e+8)
        # print("RelDiff_apply_DEBUG_END" , rate[0:10]);
        return rate;
    
    def specific_invert(self, df):
        # print("RelDiff_invert_DEBUG_START" , self.mFirstValue, df.values[0:10]);
        rate = df + 1;
        rate = rate.clip(-1.0e+8 , +1.0e+8)
        rate_cum = rate.cumprod();
        df_orig = rate_cum.clip(-1.0e+8 , +1.0e+8)
        df_orig = self.mFirstValue * df_orig;
        # print("rate" , rate)
        # print("rate_cum", rate_cum)
        # print("RelDiff_invert_DEBUG_START" , df_orig[0:10])
        return df_orig;


class cSignalTransform_Logit(cAbstractSignalTransform):

    def __init__(self):
        cAbstractSignalTransform.__init__(self);
        self.mFormula = "Logit";
        self.mComplexity = 1;
        self.mScaling = True;
        pass

    def get_name(self, iSig):
        return "Logit_" + str(iSig);


    def is_applicable(self, sig):
        if(self.mScaling is not None):
            return True;
        # this has to be a proportion ( 0 <= p <= 1.0 )
        lMinValue = np.min(sig);
        lMaxValue = np.max(sig);
        if((lMinValue >= 0.0) and (lMaxValue <= 1.0)):
            return True;
        return False;

    def specific_fit(self, sig):
        pass

    def logit(self, x):
        """ cSignalTransform_Logit:logit
        Args:
            x:     
        Returns:
           
        """
        eps = 1.0e-8;
        x1 = x;
        if(x < eps):
            x1 = eps;
        if(x > (1.0 - eps)):
            x1 = 1.0 - eps;
        y = np.log(x1) - np.log(1 - x1);
        return y;

    def inv_logit(self, y):
        """ cSignalTransform_Logit:inv_logit
        Args:
            y:     
        Returns:
           
        """
        x = np.exp(y);
        p = x / (1 + x);
        return p;

    def specific_apply(self, df):
        # logit
        df1 = df.apply(self.logit);
        return df1;
    
    def specific_invert(self, df):
        # logit
        df1 = df.apply(self.inv_logit);
        return df1;


        

class cSignalTransform_Anscombe(cAbstractSignalTransform):
    '''
    More suitable for poissonnian signals (counts)
    See https://en.wikipedia.org/wiki/Anscombe_transform
    '''

    def __init__(self):
        cAbstractSignalTransform.__init__(self);
        self.mComplexity = 1;
        self.mFormula = "Anscombe";
        self.mConstant = 3.0/ 8.0;
        self.mScaling = True;
        pass

    def get_name(self, iSig):
        return "Anscombe_" + str(iSig);
    
    def specific_fit(self , sig):
        pass
    
    def specific_apply(self, sig):
        y = sig.apply(lambda x : 2 * np.sqrt(x + self.mConstant));
        return y;
    
    def specific_invert(self, sig):
        """ cSignalTransform_Fisher:specific_invert
        Args:
            sig:     
        Returns:
           
        """
        """ cSignalTransform_Anscombe:specific_invert
        Args:
            sig:     
        Returns:
           
        """
        x = sig.apply(lambda x : ((x/2 * x/2) - self.mConstant))
        return x;


class cSignalTransform_Fisher(cAbstractSignalTransform):
    '''
    https://en.wikipedia.org/wiki/Fisher_transformation
    '''

    def __init__(self):
        cAbstractSignalTransform.__init__(self);
        self.mFormula = "Fisher";
        self.mComplexity = 1;
        self.mScaling = True;
        pass

    def get_name(self, iSig):
        return "Fisher_" + str(iSig);
    
    def specific_fit(self , sig):
        pass
    
    def specific_apply(self, sig):
        eps = 1.0e-8;
        y = sig.apply(lambda x : np.arctanh(np.clip(x , -1 + eps , 1.0 - eps)));
        return y;
    
    def specific_invert(self, sig):
        x = sig.apply(np.tanh);
        return x;



def create_tranformation(iName , arg):
    """function create_tranformation
    Args:
        iName:   
        arg:   
    Returns:
        
    """
    if(iName == 'None'):
        return cSignalTransform_None();

    if(iName == 'Difference'):
        return cSignalTransform_Differencing()
    
    if(iName == 'RelativeDifference'):
        return cSignalTransform_RelativeDifferencing()
            
    if(iName == 'Integration'):
        return cSignalTransform_Accumulate()
        
    if(iName == 'BoxCox'):
        return cSignalTransform_BoxCox(arg)
    
    if(iName == 'Quantization'):
        return cSignalTransform_Quantize(arg)
        
    if(iName == 'Logit'):
        return cSignalTransform_Logit()
        
    if(iName == 'Fisher'):
        return cSignalTransform_Fisher()
        
    if(iName == 'Anscombe'):
        return cSignalTransform_Anscombe()

    # assert(0)
    return None
