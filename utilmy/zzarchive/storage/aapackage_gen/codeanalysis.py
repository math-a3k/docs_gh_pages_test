# -*- coding: utf-8 -*-
# Describe classes, methods and functions in a module.  Works with user-defined modules, all Python library modules, including built-in modules.

#-------------------------------------------------------------------------------
#  getmodule_doc("pyfolio", file1)

#  getmodule_doc("jedi", file1)

''' Transport Python 2 to Python 3
import lib2to3

!2to3 D:\_devs\Python01\project\aapackage\codeanalysis.py


D:\_devs\Python01\project\zjavajar

'''

#---------------------------------------------------------------------------------
import inspect;import os, sys

INDENT=0
file1= r"D:\_devs\Python01\aapackage\printdoc.txt"


#Print with indentation
def wi(*args):
   aux=''
   if INDENT: aux= str(' '*INDENT)
   for arg in args: 
       dx= str(arg).replace("'", "");
       dx= dx.replace("[","");   dx= dx.replace("]","")
       aux= aux + dx  + "\n"
       
   printinfile(aux, file1)


def printinfile(vv, file1):
# print vv
 with open(file1, "a") as text_file:
    text_file.write(vv)   
    

def wi2(*args):
   if INDENT: print(' '*INDENT,)
   for arg in args: print(arg,)
   print

def indent():     global INDENT; INDENT += 4
def dedent():      global INDENT; INDENT -= 4


#----Describe a builtin function
def describe_builtin(obj):
   wi('+Built-in Function: %s' % obj.__name__)
# Built-in functions cannot be inspected by inspect.getargspec.  parse the __doc__ attribute of the function.
   docstr = obj.__doc__ ;   args = ''
   if docstr:
      items = docstr.split('\n')
      if items:
         func_descr = items[0]
         s = func_descr.replace(obj.__name__,'')
         idx1 = s.find('(')
         idx2 = s.find(')',idx1)
         if idx1 != -1 and idx2 != -1 and (idx2>idx1+1):
            args = s[idx1+1:idx2]
            wi('\t-Method Arguments:', args)

   if args=='':  wi('\t-Method Arguments: None')
   print


# Describe the function  passed as argument. method the 2n argument will be passed as True 
def describe_func(obj, method=False):
   try:
       arginfo = inspect.getargspec(obj)
   except TypeError:
      print 
      return
   args = arginfo[0];   argsvar = arginfo[1]

   if args:
       if method: wi('  +  '+obj.__name__ +'('+ str(args) +')' )
       else:    wi('  +Func: '+obj.__name__ +'('+ str(args) +')' ) 
#       if args[0] == 'self':  wi('Instance method' );    args.pop(0)
       if arginfo[3]:
           dl = len(arginfo[3]);   al = len(args)
           defargs = args[al-dl:al]
           ax= str(zip(defargs, arginfo[3]))
           wi('\t  	  Default_Args:'+ax)

   if arginfo[1]:  wi('\t   Positional_Args: ' + str( arginfo[1]))
   if arginfo[2]:  wi('\t   Keyword_Args: ' + str(arginfo[2]))
   print



#Describe class object passed as argument,including its methods 
def describe_klass(obj):
   wi('\n   +Class: %s' % obj.__name__)
   indent();   count = 0
   for name in obj.__dict__:
       try:
         item = getattr(obj, name)
         if inspect.ismethod(item): count += 1;describe_func(item, True)
       except TypeError:
           print('error')
           return
   if count==0:  wi('(No members)')
   dedent()
   print 



#Describe the module object passed as argument classes and functions 
def describe(module):
   wi('\n \n[Module: %s]-------------------------------------------------' % module.__name__)
   indent();   count = 0   
   for name in dir(module):
       obj = getattr(module, name)
       if inspect.isclass(obj): 
           count += 1; 
           try :  describe_klass(obj)
           except :
               print('error')
               return
       elif (inspect.ismethod(obj) or inspect.isfunction(obj)):
          count +=1 ; describe_func(obj)
       elif inspect.isbuiltin(obj):
          count += 1; describe_builtin(obj)

   if count==0: wi('(No members)')
   dedent()







#----Print in 1 Line Documentation of the function----------------------------------

def describe_builtin2(obj, name1):
   wi(name1+'.'+obj.__name__)
# Built-in functions cannot be inspected by inspect.getargspec.  parse the __doc__ attribute of the function.
   docstr = obj.__doc__ ;   args = ''
   if docstr:
      items = docstr.split('\n')
      if items:
         func_descr = items[0]
         s = func_descr.replace(obj.__name__,'')
         idx1 = s.find('(');  idx2 = s.find(')',idx1)
         if idx1 != -1 and idx2 != -1 and (idx2>idx1+1):
            args = s[idx1+1:idx2]
            wi('(', args)

#   if args=='':  wi('()')
   print



def describe_func2(obj, method=False, name1=''):
   try:  arginfo = inspect.getargspec(obj)
   except :
      print 
      return
   args = arginfo[0];   argsvar = arginfo[1]
   if args:
       if method: wi( name1+'.'+obj.__name__ +'('+ str(args) +')' )
       else:    wi( name1+'.'+obj.__name__ +'('+ str(args) +')' ) 



def describe_klass2(obj, name1=''): 
   for name in obj.__dict__:
       try:
         item = getattr(obj, name)
         if inspect.ismethod(item): count += 0; describe_func2(item, True, name1+'.'+obj.__name__)
       except TypeError:
           print;  return
   print 


def describe2(module):
   wi('\n \n ')   
   for name in dir(module):
       obj = getattr(module, name)
       if inspect.isclass(obj): 
           try :  describe_klass2(obj, module.__name__)
           except :
               print('error');  return
       elif (inspect.ismethod(obj) or inspect.isfunction(obj)):  describe_func2(obj, False,  module.__name__)
          
       elif inspect.isbuiltin(obj):  describe_builtin2(obj, module.__name__)




#-------------------Parse the module ------------------------------------------------
def getmodule_doc(module1, file1='moduledoc.txt'):
 import importlib;  import pkgutil;  global INDENT    
 package= importlib.import_module(module1);

#Get list of sub-module
 vv= []; INDENT =0
 for importer, modname, ispkg in  pkgutil.walk_packages(path=package.__path__, 
                                                      prefix=package.__name__+'.',
                                                      onerror=lambda x: None):
     vv.append(modname)                                                   
     wi(str(modname))
 
 
 # 1 Line Doc for each function
 for submodule1 in vv:
    try:
      mod2 = importlib.import_module(submodule1);     INDENT=0
      describe2(mod2)
    except : print(sys.exc_info())

 wi('\n \n \n \n -----------------------------------------------------------------------------')

 #Tree Base Documentation
 for submodule1 in vv:
    try:
      mod2 = importlib.import_module(submodule1);     INDENT=0
      describe(mod2)
    except :  print(sys.exc_info())
 print('Document generated in'+  file1)



#-------------------------------------------------------------------------------
#  getmodule_doc("pyfolio", file1)

#  getmodule_doc("jedi", file1)
































'''


#----------------------Example of using JEDI---------------------------------
import jedi

source = ''
 import datetime
datetime.da''

script = jedi.Script(source, 3, len('datetime.da'), 'example.py')

completions = script.completions()

print(completions[0].complete)

print(completions[0].name)




http://stackoverflow.com/questions/100298/how-can-i-analyze-python-code-to-identify-problematic-areas


https://blog.landscape.io/prospector-python-static-analysis-for-humans.html


#Find Clone Code
http://clonedigger.sourceforge.net/



https://pypi.python.org/pypi/bandit/0.17.3



pip install prospector




#To get the docs on all the functions at once, interactively. Or you can use:
help(jedi)



#To simply list the names of all the functions and variables defined in the module.
dir(jedi)


['Interpreter',
 'NotFoundError',
 'Script',
 '__builtins__',
 '__doc__',
 '__file__',
 '__name__',
 '__package__',
 '__path__',
 '__version__',
 '_compatibility',
 'api',
 'cache',
 'common',
 'debug',
 'defined_names',
 'evaluate',
 'names',
 'parser',
 'preload_module',
 'set_debug_function',
 'settings']
 
 
 






#------http://stackoverflow.com/questions/8718885/import-module-from-string-variable
list and process it with pydoc
pprint:


import pydoc
!pydoc sys










if __name__ == "__main__":
   import sys
   
   if len(sys.argv)<2:
      sys.exit('Usage: %s <module>' % sys.argv[0])

   module = sys.argv[1].replace('.py','')
   mod = __import__(module)
   describe(mod)







import pkgutil

# this is the package we are inspecting -- for example 'email' from stdlib
import jedi

package = jedi
for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
    print " %s ,Is_package: %s" % (modname, ispkg)
    
    
    

import jedi
package = jedi

                                                       

# Describe  Class
describe(jedi.api)


# Describe  sub class
describe(jedi.api)


import numpy
describe(numpy)



#Need to write the recursion


def get_all_subclasses(cls):
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses
    
    
get_all_subclasses(jedi.api)    
                                                          
                                                          
def itersubclasses(cls, _seen=None):
    """
    itersubclasses(cls)

    Generator over all subclasses of a given class, in depth first order.

    >>> list(itersubclasses(int)) == [bool]
    True
    >>> class A(object): pass
    >>> class B(A): pass
    >>> class C(A): pass
    >>> class D(B,C): pass
    >>> class E(D): pass
    >>> 
    >>> for cls in itersubclasses(A):
    ...     print(cls.__name__)
    B
    D
    E
    C
    >>> # get ALL (new-style) classes currently defined
    >>> [cls.__name__ for cls in itersubclasses(object)] #doctest: +ELLIPSIS
    ['type', ...'tuple', ...]
    """
    
    if not isinstance(cls, type):
        raise TypeError('itersubclasses must be called with '
                        'new-style classes, not %.100r' % cls)
    if _seen is None: _seen = set()
    try:
        subs = cls.__subclasses__()
    except TypeError: # fails only when cls is type
        subs = cls.__subclasses__(cls)
    for sub in subs:
        if sub not in _seen:
            _seen.add(sub)
            yield sub
            for sub in itersubclasses(sub, _seen):
                yield sub


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=1)



Evaluator.eval_statement doesn’t do much, because there’s no assignment.
Evaluator.eval_element cares for resolving the dotted path
Evaluator.find_types searches for global definitions of datetime, which it finds in the definition of an import, by scanning the syntax tree.
Using the import logic, the datetime module is found.
Now find_types is called again by eval_element to find date inside the datetime module.





Logo  
jedi.api

 jedi
  jedi.Interpreter
  jedi.Interpreter._get_completion_parts
  jedi.Interpreter.completions
  
  jedi.Script
  jedi.Script.call_signatures
  jedi.Script.complete
  jedi.Script.completions
  jedi.Script.get_in_function_call
  jedi.Script.goto_assignments
  jedi.Script.goto_definitions
  jedi.Script.usages
  
  jedi.__doc__
  jedi.__version__
  jedi._compatibility
  
  jedi.api
  jedi.api.Interpreter
  jedi.api.NotFoundError
  jedi.api.Script
  jedi.api.classes
  jedi.api.defined_names
  jedi.api.helpers
  jedi.api.interpreter
  jedi.api.keywords
  jedi.api.preload_module
  jedi.api.set_debug_function
  jedi.api.usages
  jedi.api_classes
  jedi.api_classes.CallDef
  jedi.api_classes.Definition
  
  jedi.cache
  jedi.common
  jedi.debug
  jedi.defined_names
  
  jedi.evaluate
  jedi.evaluate.Evaluator
  jedi.evaluate.cache
  jedi.evaluate.compiled
  jedi.evaluate.docstrings
  jedi.evaluate.dynamic
  jedi.evaluate.filter_private_variable
  jedi.evaluate.finder
  jedi.evaluate.follow_call
  jedi.evaluate.get_names_for_scope
  jedi.evaluate.get_scopes_for_name
  jedi.evaluate.helpers
  jedi.evaluate.imports
  jedi.evaluate.iterable
  jedi.evaluate.precedence
  jedi.evaluate.recursion
  jedi.evaluate.representation
  jedi.evaluate.stdlib
  jedi.evaluate.sys_path
  
  
  jedi.fast_parser
  jedi.fast_parser.FastParser
  
  
  jedi.functions
  jedi.functions.completions
  jedi.helpers
  jedi.interpret
  jedi.modules
  
  
  jedi.parser
  jedi.parser.Parser
  jedi.parser.fast
  jedi.parser.representation
  jedi.parser.tokenize
  jedi.parser.user_context
  
  
  jedi.parsing
  jedi.parsing.Parser
  jedi.parsing_representation
  jedi.parsing_representation.Array
  jedi.parsing_representation.Base
  jedi.parsing_representation.Call
  jedi.parsing_representation.Class
  jedi.parsing_representation.Flow
  jedi.parsing_representation.ForFlow
  jedi.parsing_representation.Function
  jedi.parsing_representation.Import
  jedi.parsing_representation.Name
  jedi.parsing_representation.Statement
  jedi.parsing_representation.SubModule
  
  
  jedi.preload_module
  jedi.refactoring
  jedi.set_debug_function
  jedi.settings
  jedi.utils
  jedi.utils.setup_readline
  jedi.utils.version_info
  
  
  
  
  '''
  
  
  