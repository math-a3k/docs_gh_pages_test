# -*- coding: utf-8 -*-
# Describe classes, methods and functions in a module.  Works with user-defined modules, all Python library modules, including built-in modules.

#  getmodule_doc("jedi")


#---------------------------------------------------------------------------------
import inspect;import os, sys; global file1, dirdoc1;   INDENT=0
file1= r"D:\_devs\Python01\printdoc.txt"
dirdoc1= r'D:\_devs\Python01'


def wi(*args):  #Print with indentation
   aux=''
   if INDENT: aux= str(' '*INDENT)
   for arg in args: 
       dx= str(arg).replace("'", "");
       dx= dx.replace("[","");   dx= dx.replace("]","")
       aux= aux + dx  + "\n"
       
   printinfile(aux, file1)


def printinfile(vv, file2):  # print vv
 with open(file2, "a") as text_file:    text_file.write(vv)   
    

def wi2(*args):
   if INDENT: print ' '*INDENT,
   for arg in args: print arg,
   print

def indent():     global INDENT; INDENT += 4
def dedent():      global INDENT; INDENT -= 4


#----Describe a builtin function-----------------------------------------------------
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
   try: arginfo = inspect.getargspec(obj)
   except TypeError:
      print ;      return
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
           print 'error'
           return
   if count==0:  wi('(No members)')
   dedent();   print 



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
               print 'error';   return
       elif (inspect.ismethod(obj) or inspect.isfunction(obj)):   count +=1 ; describe_func(obj)
       elif inspect.isbuiltin(obj):   count += 1; describe_builtin(obj)

   if count==0: wi('(No members)')
   dedent()




#----Print in 1 Line + Documentation of the function----------------------------------

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

#No doc priniting
def describe_func2(obj, method=False, name1=''):
   try:  arginfo = inspect.getargspec(obj)
   except :
      print ;      return
   args = arginfo[0];   argsvar = arginfo[1]
   if args: wi( name1+'.'+obj.__name__ +'('+ str(args) +') ' )



#Doc Printing
def describe_func3(obj, method=False, name1=''):
   try:  arginfo = inspect.getargspec(obj)
   except :
      print ;      return
   args = arginfo[0];   argsvar = arginfo[1]
   if args:
       aux= name1+'.'+obj.__name__ +'('+ str(args) +')  \n' + str(inspect.getdoc(obj))
       aux= aux.replace('\n', '\n       ') 
       aux= aux.rstrip()
       aux= aux + ' \n'
       wi( aux)


def describe_klass2(obj, name1=''): 
   for name in obj.__dict__:
       try:
         item = getattr(obj, name)
         if inspect.ismethod(item):  describe_func2(item, True, name1+'.'+obj.__name__)
       except :
           print;  return
   print 


def describe2(module, type1=0):
   wi('\n \n ')   
   for name in dir(module):
       obj = getattr(module, name)
       if inspect.isclass(obj): 
           try :  describe_klass2(obj, module.__name__)
           except :  print 'error';  return
       elif (inspect.ismethod(obj) or inspect.isfunction(obj)): 
         if type1==0:         describe_func2(obj, False,  module.__name__)
         elif type1==1:       describe_func3(obj, False,  module.__name__) 
       elif inspect.isbuiltin(obj):  describe_builtin2(obj, module.__name__)




'''  get info 
inspect.getdoc. 
remove space
'test string \n'.rstrip('\n')
'''


####################################################################################
#-------------------Parse the module recursively------------------------------------------------
def getmodule_doc(module1, file2=''):
 import importlib;  import pkgutil;  global INDENT, file1    
 package= importlib.import_module(module1);
 
 if file2== '': file1= dirdoc1 + '\\'+ module1 + '_doc.txt'
 else: file1= file2

#Get list of sub-module
 vv= []; INDENT =0
 for importer, modname, ispkg in  pkgutil.walk_packages(path=package.__path__, 
                                                      prefix=package.__name__+'.',
                                                      onerror=lambda x: None):
     vv.append(modname)                                                   
     wi(str(modname))
 
 for submodule1 in vv:    # 1 Line Doc for each function
    try:
      mod2 = importlib.import_module(submodule1);     INDENT=0
      describe2(mod2)
    except : print sys.exc_info()

 for submodule1 in vv:   # Function + Doc
    try:
      mod2 = importlib.import_module(submodule1,1);     INDENT=0
      describe2(mod2, 1)
    except : print sys.exc_info()
        
 wi('\n \n \n \n -----------------------------------------------------------------------------')
 for submodule1 in vv:   #Tree Base Documentation
    try:
      mod2 = importlib.import_module(submodule1);     INDENT=0
      describe(mod2)
    except :  print sys.exc_info()

 print('Document generated in '+  file1)



#-------------------------------------------------------------------------------
#  getmodule_doc("pyfolio", file1)

# getmodule_doc("jedi")
































'''


#----------------------Example of using JEDI---------------------------------
import jedi

source = '
 import datetime
datetime.da'

script = jedi.Script(source, 3, len('datetime.da'), 'example.py')

completions = script.completions()

print(completions[0].complete)

print(completions[0].name)




from jedi import Script
source = 
import os
os.path.join
script = Script(source, 3, len('os.path.join'), 'example.py')
print(script.goto_definitions()[0].full_name)



http://jedi.jedidjah.ch/en/latest/docs/plugin-api.html


A Script is the base for completions, goto or whatever you want to do with Jedi.
jedi.api.Script(source=None, line=None, column=None, path=None, encoding='utf-8', source_path=None, source_encoding=None)


You can either use the source parameter or path to read a file.

Parameters:	
source (str) – The source code of the current file, separated by newlines.
line (int) – The line to perform actions on (starting with 1).
column (int) – The column of the cursor (starting with 0).
path (str or None) – The path of the file in the file system, or '' if it hasn’t been saved yet.
encoding (str) – The encoding of source, if it is not a unicode object (default 'utf-8').
source_encoding – The encoding of source, if it is not a unicode object (default 'utf-8').


http://jedi.jedidjah.ch/en/latest/docs/plugin-api-classes.html#jedi.api.classes.Definition





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
  
  

jedi.__main__.abspath(path)
jedi.__main__.dirname(p)
jedi.__main__._isdir
()
jedi.__main__.join(path)

 
 
jedi._compatibility.Python3Method.__init__(self, fun_apply)
jedi._compatibility.Python3Method.__get__(self, obj, objtype)
jedi._compatibility.exec_function(source, global_map)
jedi._compatibility.find_module_pre_py33(string, path)
jedi._compatibility.find_module_pre_py33(string, path)
jedi._compatibility.find_module_py33(string, path)
jedi._compatibility.literal_eval(string)
jedi._compatibility.no_unicode_pprint(dct)
jedi._compatibility.reraise(exception, traceback)
jedi._compatibility.u(string)
jedi._compatibility.use_metaclass(meta)
jedi._compatibility.utf8_repr(fun_apply)

 
 
jedi.api.Evaluator.wrapper(obj)
jedi.api.Evaluator.goto(self, name)
jedi.api.Evaluator.execute_evaluated(self, obj)
jedi.api.Evaluator.find_types(self, scope, name_str, position, search_global, is_goto)
jedi.api.Evaluator.goto_definition(self, name)
jedi.api.Evaluator._eval_atom(self, atom)
jedi.api.Evaluator.wrapper(obj)
jedi.api.Evaluator.wrap(self, element)
jedi.api.Evaluator.__init__(self, grammar)
jedi.api.Evaluator.eval_trailer(self, types, trailer)
jedi.api.FakeName.get_definition(self)
jedi.api.FakeName.is_definition(self)
jedi.api.FakeName.__init__(self, name_str, parent, start_pos, is_definition)
jedi.api.Interpreter.__init__(self, source, namespaces)
jedi.api.Interpreter._simple_complete(self, path, dot, like)
jedi.api.Parser._add_syntax_error(self, message, position)
jedi.api.Parser.remove_last_newline(self)
jedi.api.Parser.error_recovery(self, grammar, stack, typ, value, start_pos, prefix, add_token_callback)
jedi.api.Parser.__repr__(self)
jedi.api.Parser.__init__(self, grammar, source, module_path, tokenizer)
jedi.api.Parser.convert_node(self, grammar, type, children)
jedi.api.Parser.convert_leaf(self, grammar, type, value, prefix, start_pos)
jedi.api.Parser._stack_removal(self, grammar, stack, start_index, value, start_pos)
jedi.api.Parser._tokenize(self, tokenizer)
jedi.api.Script._simple_complete(self, path, dot, like)
jedi.api.Script.completions(self)
jedi.api.Script.wrapper(obj)
jedi.api.Script.usages(self, additional_module_paths)
jedi.api.Script.goto_assignments(self)
jedi.api.Script._goto(self, add_import_name)
jedi.api.Script._prepare_goto(self, goto_path, is_completion)
jedi.api.Script.goto_definitions(self)
jedi.api.Script._analysis(self)
jedi.api.Script.__repr__(self)
jedi.api.Script.__init__(self, source, line, column, path, encoding, source_path, source_encoding)
jedi.api.Script.call_signatures(self)
jedi.api.Script._parsed_callback(self, parser)
jedi.api.UserContext._backwards_line_generator(self, start_pos)
jedi.api.UserContext.get_path_under_cursor(self)
jedi.api.UserContext.get_line(self, line_nr)
jedi.api.UserContext._calc_path_until_cursor(self, start_pos)
jedi.api.UserContext._get_backwards_tokenizer(self, start_pos, line_gen)
jedi.api.UserContext.get_position_line(self)
jedi.api.UserContext.get_path_after_cursor(self)
jedi.api.UserContext.get_operator_under_cursor(self)
jedi.api.UserContext.call_signature(self)
jedi.api.UserContext.get_context(self, yield_positions)
jedi.api.UserContext.wrapper(self)
jedi.api.UserContext.__init__(self, source, position)
jedi.api.UserContextParser.wrapper(self)
jedi.api.UserContextParser.module(self)
jedi.api.UserContextParser.__init__(self, grammar, source, path, position, user_context, parser_done_callback, use_fast_parser)
jedi.api.UserContextParser.wrapper(self)
jedi.api.UserContextParser.wrapper(self)
jedi.api.UserContextParser.wrapper(self)
jedi.api.defined_names(source, path, encoding)
jedi.api.filter_definition_names(names, origin, position)
jedi.api.get_module_names(module, all_scopes)
jedi.api.global_names_dict_generator(evaluator, scope, position)
jedi.api.load_grammar(file)
jedi.api.memoize_default(default, evaluator_is_first_arg, second_arg_is_evaluator)
jedi.api.names(source, path, encoding, all_scopes, definitions, references)
jedi.api.set_debug_function(func_cb, warnings, notices, speed)
jedi.api.source_tokens(source)

 
 
jedi.api.classes.BaseDefinition.parent(self)
jedi.api.classes.BaseDefinition.docstring(self, raw)
jedi.api.classes.BaseDefinition.goto_assignments(self)
jedi.api.classes.BaseDefinition.__init__(self, evaluator, name)
jedi.api.classes.BaseDefinition.in_builtin_module(self)
jedi.api.classes.BaseDefinition.__repr__(self)
jedi.api.classes.BaseDefinition.wrapper(obj)
jedi.api.classes.BaseDefinition._path(self)
jedi.api.classes.CachedMetaClass.wrapper(obj)
jedi.api.classes.CallSignature.__repr__(self)
jedi.api.classes.CallSignature.__init__(self, evaluator, executable_name, call_stmt, index, key_name)
jedi.api.classes.Completion.wrapper(obj)
jedi.api.classes.Completion.docstring(self, raw, fast)
jedi.api.classes.Completion.wrapper(obj)
jedi.api.classes.Completion.__repr__(self)
jedi.api.classes.Completion._complete(self, like_name)
jedi.api.classes.Completion.__init__(self, evaluator, name, needs_dot, like_name_length)
jedi.api.classes.Definition.__ne__(self, other)
jedi.api.classes.Definition.wrapper(obj)
jedi.api.classes.Definition.__hash__(self)
jedi.api.classes.Definition.__eq__(self, other)
jedi.api.classes.Definition.is_definition(self)
jedi.api.classes.Definition.__init__(self, evaluator, definition)
jedi.api.classes._Help.full(self)
jedi.api.classes._Help.raw(self)
jedi.api.classes._Help.__init__(self, definition)
jedi.api.classes._Param.get_code(self)
jedi.api.classes.defined_names(evaluator, scope)
jedi.api.classes.filter_definition_names(names, origin, position)
jedi.api.classes.memoize_default(default, evaluator_is_first_arg, second_arg_is_evaluator)
jedi.api.classes.use_metaclass(meta)

 
 
jedi.api.helpers.check_error_statements(module, pos)
jedi.api.helpers.completion_parts(path_until_cursor)
jedi.api.helpers.get_on_import_stmt(evaluator, user_context, user_stmt, is_like_search)
jedi.api.helpers.importer_from_error_statement(error_statement, pos)
jedi.api.helpers.sorted_definitions(defs)

 
 
jedi.api.interpreter.FastParser.update(self, source)
jedi.api.interpreter.FastParser._get_node(self, source, parser_code, line_offset, nodes)
jedi.api.interpreter.FastParser._split_parts(self, source)
jedi.api.interpreter.FastParser._parse(self, source)
jedi.api.interpreter.FastParser._reset_caches(self)
jedi.api.interpreter.FastParser.__init__(self, grammar, source, module_path)
jedi.api.interpreter.LazyName.__init__(self, evaluator, module, name, value)
jedi.api.interpreter.LazyName.is_definition(self)
jedi.api.interpreter.add_namespaces_to_parser(evaluator, namespaces, parser_module)
jedi.api.interpreter.get_module(obj)
jedi.api.interpreter.load_grammar(file)
jedi.api.interpreter.source_to_unicode(source, encoding)
jedi.api.interpreter.underscore_memoization(fun_apply)

 
 
jedi.api.keywords.FakeName.get_definition(self)
jedi.api.keywords.FakeName.is_definition(self)
jedi.api.keywords.FakeName.__init__(self, name_str, parent, start_pos, is_definition)
jedi.api.keywords.Keyword.__repr__(self)
jedi.api.keywords.Keyword.get_parent_until(self)
jedi.api.keywords.Keyword.__init__(self, name, pos)
jedi.api.keywords.get_operator(string, pos)
jedi.api.keywords.imitate_pydoc(string)
jedi.api.keywords.keywords(string, pos, all)

 
 

 
 
jedi.api.usages.usages(evaluator, definition_names, mods)
jedi.api.usages.usages_add_import_modules(evaluator, definitions)

 
 
jedi.cache.ParserCacheItem.__init__(self, parser, change_time)
jedi.cache._invalidate_star_import_cache_module(module, only_main)
jedi.cache.cache_star_import(fun_apply)
jedi.cache.clear_time_caches(delete_all)
jedi.cache.invalidate_star_import_cache(path)
jedi.cache.load_parser(path)
jedi.cache.memoize_method(method)
jedi.cache.save_parser(path, parser, pickling)
jedi.cache.time_cache(time_add_setting)
jedi.cache.underscore_memoization(fun_apply)

 
 
jedi.common.PushBackIterator.push_back(self, value)
jedi.common.PushBackIterator.__next__(self)
jedi.common.PushBackIterator.__iter__(self)
jedi.common.PushBackIterator.__init__(self, iterator)
jedi.common.PushBackIterator.next(self)
jedi.common.indent_block(text, indention)
jedi.common.literal_eval(node_or_string)
jedi.common.reraise(exception, traceback)
jedi.common.reraise_uncaught(fun_apply)
jedi.common.safe_property(fun_apply)
jedi.common.source_to_unicode(source, encoding)
jedi.common.splitlines(string)

 
 
jedi.debug.dbg(message)
jedi.debug.increase_indent(fun_apply)
jedi.debug.print_to_stdout(level, str_out)
jedi.debug.speed(name)
jedi.debug.u(string)
jedi.debug.warning(message)

 
 
jedi.evaluate.Evaluator.wrapper(obj)
jedi.evaluate.Evaluator.goto(self, name)
jedi.evaluate.Evaluator.execute_evaluated(self, obj)
jedi.evaluate.Evaluator.find_types(self, scope, name_str, position, search_global, is_goto)
jedi.evaluate.Evaluator.goto_definition(self, name)
jedi.evaluate.Evaluator._eval_atom(self, atom)
jedi.evaluate.Evaluator.wrapper(obj)
jedi.evaluate.Evaluator.wrap(self, element)
jedi.evaluate.Evaluator.__init__(self, grammar)
jedi.evaluate.Evaluator.eval_trailer(self, types, trailer)
jedi.evaluate.memoize_default(default, evaluator_is_first_arg, second_arg_is_evaluator)

 
 
jedi.evaluate.analysis.CompiledObject.wrapper(self)
jedi.evaluate.analysis.CompiledObject.get_self_attributes(self)
jedi.evaluate.analysis.CompiledObject.wrapper(self)

 
 
jedi.evaluate.cache.CachedMetaClass.wrapper(obj)
jedi.evaluate.cache.memoize_default(default, evaluator_is_first_arg, second_arg_is_evaluator)

 
 
jedi.evaluate.compiled.Base.is_scope(self)
jedi.evaluate.compiled.Base.get_parent_scope(self, include_flows)
jedi.evaluate.compiled.Base.isinstance(self)
jedi.evaluate.compiled.Builtin.wrapper(self)
jedi.evaluate.compiled.CheckAttribute.__init__(self, fun_apply)
jedi.evaluate.compiled.CheckAttribute.__get__(self, instance, owner)
jedi.evaluate.compiled.CompiledName.is_definition(self)
jedi.evaluate.compiled.CompiledName.__repr__(self)
jedi.evaluate.compiled.CompiledName.__init__(self, obj, name)
jedi.evaluate.compiled.CompiledObject.wrapper(self)
jedi.evaluate.compiled.CompiledObject.get_self_attributes(self)
jedi.evaluate.compiled.CompiledObject.wrapper(self)

 
 
jedi.evaluate.compiled.fake.FakeName.get_definition(self)
jedi.evaluate.compiled.fake.FakeName.is_definition(self)
jedi.evaluate.compiled.fake.FakeName.__init__(self, name_str, parent, start_pos, is_definition)
jedi.evaluate.compiled.fake.Parser._add_syntax_error(self, message, position)
jedi.evaluate.compiled.fake.Parser.remove_last_newline(self)
jedi.evaluate.compiled.fake.Parser.error_recovery(self, grammar, stack, typ, value, start_pos, prefix, add_token_callback)
jedi.evaluate.compiled.fake.Parser.__repr__(self)
jedi.evaluate.compiled.fake.Parser.__init__(self, grammar, source, module_path, tokenizer)
jedi.evaluate.compiled.fake.Parser.convert_node(self, grammar, type, children)
jedi.evaluate.compiled.fake.Parser.convert_leaf(self, grammar, type, value, prefix, start_pos)
jedi.evaluate.compiled.fake.Parser._stack_removal(self, grammar, stack, start_index, value, start_pos)
jedi.evaluate.compiled.fake.Parser._tokenize(self, tokenizer)
jedi.evaluate.compiled.fake._faked(module, obj, name)
jedi.evaluate.compiled.fake._load_faked_module(module)
jedi.evaluate.compiled.fake.get_faked(module, obj, name)
jedi.evaluate.compiled.fake.get_module(obj)
jedi.evaluate.compiled.fake.is_class_instance(obj)
jedi.evaluate.compiled.fake.load_grammar(file)
jedi.evaluate.compiled.fake.search_scope(scope, obj_name)

 
 
jedi.evaluate.docstrings.AlreadyEvaluated.get_code(self)
jedi.evaluate.docstrings.Array.wrapper(obj)
jedi.evaluate.docstrings.Array.get_exact_index_types(self, mixed_index)
jedi.evaluate.docstrings.Array.__getattr__(self, name)
jedi.evaluate.docstrings.Array.iter_content(self)
jedi.evaluate.docstrings.Array.__iter__(self)
jedi.evaluate.docstrings.Array.wrapper(obj)
jedi.evaluate.docstrings.Array.__init__(self, evaluator, atom)
jedi.evaluate.docstrings.Array._values(self)
jedi.evaluate.docstrings.Array._items(self)
jedi.evaluate.docstrings.Array.get_parent_until(self)
jedi.evaluate.docstrings.Array.__repr__(self)
jedi.evaluate.docstrings.FakeSequence._items(self)
jedi.evaluate.docstrings.FakeSequence.get_exact_index_types(self, index)
jedi.evaluate.docstrings.FakeSequence.__init__(self, evaluator, sequence_values, type)
jedi.evaluate.docstrings.NumpyDocString._str_header(self, name, symbol)
jedi.evaluate.docstrings.NumpyDocString._str_section(self, name)
jedi.evaluate.docstrings.NumpyDocString.__getitem__(self, key)
jedi.evaluate.docstrings.NumpyDocString.__str__(self, func_role)
jedi.evaluate.docstrings.NumpyDocString._str_indent(self, doc, indent)
jedi.evaluate.docstrings.NumpyDocString._read_to_next_section(self)
jedi.evaluate.docstrings.NumpyDocString._str_signature(self)
jedi.evaluate.docstrings.NumpyDocString.__setitem__(self, key, val)
jedi.evaluate.docstrings.NumpyDocString._read_sections(self)
jedi.evaluate.docstrings.NumpyDocString._str_extended_summary(self)
jedi.evaluate.docstrings.NumpyDocString._str_summary(self)
jedi.evaluate.docstrings.NumpyDocString.__init__(self, docstring, config)
jedi.evaluate.docstrings.NumpyDocString._strip(self, doc)
jedi.evaluate.docstrings.NumpyDocString._str_index(self)
jedi.evaluate.docstrings.NumpyDocString._parse_see_also(self, content)
jedi.evaluate.docstrings.NumpyDocString._parse_summary(self)
jedi.evaluate.docstrings.NumpyDocString._parse_index(self, section, content)
jedi.evaluate.docstrings.NumpyDocString._str_see_also(self, func_role)
jedi.evaluate.docstrings.NumpyDocString._parse_param_list(self, content)
jedi.evaluate.docstrings.NumpyDocString._parse(self)
jedi.evaluate.docstrings.NumpyDocString._str_param_list(self, name)
jedi.evaluate.docstrings.NumpyDocString._is_at_section(self)
jedi.evaluate.docstrings.Parser._add_syntax_error(self, message, position)
jedi.evaluate.docstrings.Parser.remove_last_newline(self)
jedi.evaluate.docstrings.Parser.error_recovery(self, grammar, stack, typ, value, start_pos, prefix, add_token_callback)
jedi.evaluate.docstrings.Parser.__repr__(self)
jedi.evaluate.docstrings.Parser.__init__(self, grammar, source, module_path, tokenizer)
jedi.evaluate.docstrings.Parser.convert_node(self, grammar, type, children)
jedi.evaluate.docstrings.Parser.convert_leaf(self, grammar, type, value, prefix, start_pos)
jedi.evaluate.docstrings.Parser._stack_removal(self, grammar, stack, start_index, value, start_pos)
jedi.evaluate.docstrings.Parser._tokenize(self, tokenizer)
jedi.evaluate.docstrings._evaluate_for_statement_string(evaluator, string, module)
jedi.evaluate.docstrings._execute_array_values(evaluator, array)
jedi.evaluate.docstrings._execute_types_in_stmt(evaluator, stmt)
jedi.evaluate.docstrings._search_param_in_docstr(docstr, param_str)
jedi.evaluate.docstrings._search_param_in_numpydocstr(docstr, param_str)
jedi.evaluate.docstrings._strip_rst_role(type_str)
jedi.evaluate.docstrings.dedent(text)
jedi.evaluate.docstrings.wrapper(obj)
jedi.evaluate.docstrings.wrapper(obj)
jedi.evaluate.docstrings.indent_block(text, indention)
jedi.evaluate.docstrings.literal_eval(node_or_string)
jedi.evaluate.docstrings.load_grammar(file)
jedi.evaluate.docstrings.memoize_default(default, evaluator_is_first_arg, second_arg_is_evaluator)

 
 
jedi.evaluate.dynamic.ParamListener.execute(self, params)
jedi.evaluate.dynamic.ParamListener.__init__(self)
jedi.evaluate.dynamic.memoize_default(default, evaluator_is_first_arg, second_arg_is_evaluator)
jedi.evaluate.dynamic.wrapper(obj)

 
 
jedi.evaluate.finder.NameFinder.scopes(self, search_global)
jedi.evaluate.finder.NameFinder.names_dict_lookup(self, names_dict, position)
jedi.evaluate.finder.NameFinder._resolve_descriptors(self, name, types)
jedi.evaluate.finder.NameFinder._check_getattr(self, inst)
jedi.evaluate.finder.NameFinder._names_to_types(self, names, search_global)
jedi.evaluate.finder.NameFinder.filter_name(self, names_dicts)
jedi.evaluate.finder.NameFinder.__init__(self, evaluator, scope, name_str, position)
jedi.evaluate.finder.NameFinder._clean_names(self, names)
jedi.evaluate.finder._check_isinstance_type(evaluator, element, search_name)
jedi.evaluate.finder._eval_param(evaluator, param, scope)
jedi.evaluate.finder.wrapper(obj)
jedi.evaluate.finder._remove_statements(evaluator, stmt, name)
jedi.evaluate.finder.check_flow_information(evaluator, flow, search_name, pos)
jedi.evaluate.finder.check_tuple_assignments(types, name)
jedi.evaluate.finder.filter_after_position(names, position)
jedi.evaluate.finder.filter_definition_names(names, origin, position)
jedi.evaluate.finder.filter_private_variable(scope, origin_node)
jedi.evaluate.finder.global_names_dict_generator(evaluator, scope, position)
jedi.evaluate.finder.memoize_default(default, evaluator_is_first_arg, second_arg_is_evaluator)
jedi.evaluate.finder.u(string)

 
 
jedi.evaluate.flow_analysis.Status.__repr__(self)
jedi.evaluate.flow_analysis.Status.__and__(self, other)
jedi.evaluate.flow_analysis.Status.invert(self)
jedi.evaluate.flow_analysis.Status.__init__(self, value, name)
jedi.evaluate.flow_analysis._break_check(evaluator, stmt, base_scope, element_scope)
jedi.evaluate.flow_analysis._check_if(evaluator, node)
jedi.evaluate.flow_analysis.break_check(evaluator, base_scope, stmt, origin_scope)

 
 
jedi.evaluate.helpers.FakeImport.paths(self)
jedi.evaluate.helpers.FakeImport.get_defined_names(self)
jedi.evaluate.helpers.FakeImport.is_definition(self)
jedi.evaluate.helpers.FakeImport.__init__(self, name, parent, level)
jedi.evaluate.helpers.FakeImport.aliases(self)
jedi.evaluate.helpers.FakeName.get_definition(self)
jedi.evaluate.helpers.FakeName.is_definition(self)
jedi.evaluate.helpers.FakeName.__init__(self, name_str, parent, start_pos, is_definition)
jedi.evaluate.helpers.LazyName.__init__(self, name, parent_callback, is_definition)
jedi.evaluate.helpers.call_of_name(name, cut_own_trailer)
jedi.evaluate.helpers.deep_ast_copy(obj, parent, new_elements)
jedi.evaluate.helpers.get_module_names(module, all_scopes)

 
 
jedi.evaluate.imports.ImportWrapper.wrapper(obj)
jedi.evaluate.imports.ImportWrapper.__init__(self, evaluator, name)
jedi.evaluate.imports.Importer._do_import(self, import_path, sys_path)
jedi.evaluate.imports.Importer.completion_names(self, evaluator, only_modules)
jedi.evaluate.imports.Importer.wrapper(obj)
jedi.evaluate.imports.Importer._get_module_names(self, search_path)
jedi.evaluate.imports.Importer.wrapper(obj)
jedi.evaluate.imports.Importer._generate_name(self, name)
jedi.evaluate.imports.Importer.__init__(self, evaluator, import_path, module, level)
jedi.evaluate.imports.NestedImportModule.__repr__(self)
jedi.evaluate.imports.NestedImportModule.__init__(self, module, nested_import)
jedi.evaluate.imports.NestedImportModule._get_nested_import_name(self)
jedi.evaluate.imports.NestedImportModule.__getattr__(self, name)
jedi.evaluate.imports._add_error(evaluator, name, message)
jedi.evaluate.imports._load_module(evaluator, path, source, sys_path)
jedi.evaluate.imports.add_module(evaluator, module_name, module)
jedi.evaluate.imports.completion_names(evaluator, imp, pos)
jedi.evaluate.imports.find_module_pre_py33(string, path)
jedi.evaluate.imports.get_init_path(directory_path)
jedi.evaluate.imports.get_modules_containing_name(evaluator, mods, name)
jedi.evaluate.imports.memoize_default(default, evaluator_is_first_arg, second_arg_is_evaluator)
jedi.evaluate.imports.source_to_unicode(source, encoding)

 
 
jedi.evaluate.iterable.AlreadyEvaluated.get_code(self)
jedi.evaluate.iterable.Array.wrapper(obj)
jedi.evaluate.iterable.Array.get_exact_index_types(self, mixed_index)
jedi.evaluate.iterable.Array.__getattr__(self, name)
jedi.evaluate.iterable.Array.iter_content(self)
jedi.evaluate.iterable.Array.__iter__(self)
jedi.evaluate.iterable.Array.wrapper(obj)
jedi.evaluate.iterable.Array.__init__(self, evaluator, atom)
jedi.evaluate.iterable.Array._values(self)
jedi.evaluate.iterable.Array._items(self)
jedi.evaluate.iterable.Array.get_parent_until(self)
jedi.evaluate.iterable.Array.__repr__(self)
jedi.evaluate.iterable.ArrayInstance.iter_content(self)
jedi.evaluate.iterable.ArrayInstance.__init__(self, evaluator, instance)
jedi.evaluate.iterable.ArrayMixin.py__bool__(self)
jedi.evaluate.iterable.ArrayMixin.wrapper(obj)
jedi.evaluate.iterable.CachedMetaClass.wrapper(obj)
jedi.evaluate.iterable.Comprehension.wrapper(obj)
jedi.evaluate.iterable.Comprehension.__repr__(self)
jedi.evaluate.iterable.Comprehension.get_exact_index_types(self, index)
jedi.evaluate.iterable.Comprehension.__init__(self, evaluator, atom)
jedi.evaluate.iterable.FakeDict._items(self)
jedi.evaluate.iterable.FakeDict.get_exact_index_types(self, index)
jedi.evaluate.iterable.FakeDict.__init__(self, evaluator, dct)
jedi.evaluate.iterable.FakeSequence._items(self)
jedi.evaluate.iterable.FakeSequence.get_exact_index_types(self, index)
jedi.evaluate.iterable.FakeSequence.__init__(self, evaluator, sequence_values, type)
jedi.evaluate.iterable.Generator.iter_content(self)
jedi.evaluate.iterable.Generator.__repr__(self)
jedi.evaluate.iterable.Generator.__init__(self, evaluator, fun_apply, var_args)
jedi.evaluate.iterable.Generator.__getattr__(self, name)
jedi.evaluate.iterable.GeneratorComprehension.iter_content(self)
jedi.evaluate.iterable.GeneratorMethod.py__call__(self, evaluator, params)
jedi.evaluate.iterable.GeneratorMethod.__init__(self, generator, builtin_func)
jedi.evaluate.iterable.GeneratorMethod.__getattr__(self, name)
jedi.evaluate.iterable.GeneratorMixin.get_index_types(self, evaluator, index_array)
jedi.evaluate.iterable.GeneratorMixin.py__bool__(self)
jedi.evaluate.iterable.GeneratorMixin.get_exact_index_types(self, index)
jedi.evaluate.iterable.GeneratorMixin.wrapper(obj)
jedi.evaluate.iterable.ImplicitTuple._items(self)
jedi.evaluate.iterable.ImplicitTuple.__init__(self, evaluator, testlist)
jedi.evaluate.iterable.IterableWrapper.is_class(self)
jedi.evaluate.iterable.ListComprehension.get_index_types(self, evaluator, index)
jedi.evaluate.iterable.ListComprehension.iter_content(self)
jedi.evaluate.iterable.MergedArray.values(self)
jedi.evaluate.iterable.MergedArray.__len__(self)
jedi.evaluate.iterable.MergedArray.get_exact_index_types(self, mixed_index)
jedi.evaluate.iterable.MergedArray.__iter__(self)
jedi.evaluate.iterable.MergedArray.__init__(self, evaluator, arrays)
jedi.evaluate.iterable.Slice.__init__(self, evaluator, start, stop, step)
jedi.evaluate.iterable._FakeArray.__init__(self, evaluator, container, type)
jedi.evaluate.iterable.wrapper(obj)
jedi.evaluate.iterable.check_array_additions(evaluator, array)
jedi.evaluate.iterable.check_array_instances(evaluator, instance)
jedi.evaluate.iterable.create_indexes_or_slices(evaluator, index)
jedi.evaluate.iterable.get_iterator_types(inputs)
jedi.evaluate.iterable.memoize_default(default, evaluator_is_first_arg, second_arg_is_evaluator)
jedi.evaluate.iterable.unite(iterable)
jedi.evaluate.iterable.use_metaclass(meta)

 
 
jedi.evaluate.param.Arguments.eval_argument_clinic(self, arguments)
jedi.evaluate.param.Arguments.get_parent_until(self)
jedi.evaluate.param.Arguments.as_tuple(self)
jedi.evaluate.param.Arguments.eval_args(self)
jedi.evaluate.param.Arguments.__repr__(self)
jedi.evaluate.param.Arguments.scope(self)
jedi.evaluate.param.Arguments.get_calling_var_args(self)
jedi.evaluate.param.Arguments._reorder_var_args(var_args)
jedi.evaluate.param.Arguments.unpack(self, fun_apply)
jedi.evaluate.param.Arguments.__init__(self, evaluator, argument_node, trailer)
jedi.evaluate.param.Arguments._split(self)
jedi.evaluate.param.ExecutedParam.__init__(self, original_param, var_args, values)
jedi.evaluate.param.ExecutedParam.eval(self, evaluator)
jedi.evaluate.param.ExecutedParam.__getattr__(self, name)
jedi.evaluate.param.FakeName.get_definition(self)
jedi.evaluate.param.FakeName.is_definition(self)
jedi.evaluate.param.FakeName.__init__(self, name_str, parent, start_pos, is_definition)
jedi.evaluate.param._error_argument_count(fun_apply, actual_count)
jedi.evaluate.param._get_calling_var_args(evaluator, var_args)
jedi.evaluate.param._iterate_star_args(evaluator, array, input_node, fun_apply)
jedi.evaluate.param._star_star_dict(evaluator, array, input_node, fun_apply)
jedi.evaluate.param.get_params(evaluator, fun_apply, var_args)
jedi.evaluate.param.underscore_memoization(fun_apply)

 
 
jedi.evaluate.precedence.CompiledObject.wrapper(self)
jedi.evaluate.precedence.CompiledObject.get_self_attributes(self)
jedi.evaluate.precedence.CompiledObject.wrapper(self)

 
 
jedi.evaluate.recursion.ExecutionRecursionDetector.pop_execution(cls)
jedi.evaluate.recursion.ExecutionRecursionDetector.__call__(self, execution)
jedi.evaluate.recursion.ExecutionRecursionDetector.__init__(self)
jedi.evaluate.recursion.ExecutionRecursionDetector.push_execution(cls, execution)
jedi.evaluate.recursion.RecursionDetector._check_recursion(self)
jedi.evaluate.recursion.RecursionDetector.node_statements(self)
jedi.evaluate.recursion.RecursionDetector.pop_stmt(self)
jedi.evaluate.recursion.RecursionDetector.push_stmt(self, stmt)
jedi.evaluate.recursion.RecursionDetector.__init__(self)
jedi.evaluate.recursion._RecursionNode.__eq__(self, other)
jedi.evaluate.recursion._RecursionNode.__init__(self, stmt, parent)
jedi.evaluate.recursion.execution_recursion_decorator(fun_apply)
jedi.evaluate.recursion.recursion_decorator(fun_apply)

 
 
jedi.evaluate.representation.CachedMetaClass.wrapper(obj)
jedi.evaluate.representation.Class.py__call__(self, evaluator, params)
jedi.evaluate.representation.Class.wrapper(obj)
jedi.evaluate.representation.Class.py__getattribute__(self, name)
jedi.evaluate.representation.Class.is_class(self)
jedi.evaluate.representation.Class.__getattr__(self, name)
jedi.evaluate.representation.Class.__repr__(self)
jedi.evaluate.representation.Class.names_dicts(self, search_global, is_instance)
jedi.evaluate.representation.Class.wrapper(obj)
jedi.evaluate.representation.Class.get_subscope_by_name(self, name)
jedi.evaluate.representation.Class.__init__(self, evaluator, base)
jedi.evaluate.representation.Executed.is_scope(self)
jedi.evaluate.representation.Executed.get_parent_until(self)
jedi.evaluate.representation.Executed.__init__(self, evaluator, base, var_args)
jedi.evaluate.representation.Function.wrapper(obj)
jedi.evaluate.representation.Function.__getattr__(self, name)
jedi.evaluate.representation.Function.__repr__(self)
jedi.evaluate.representation.Function.names_dicts(self, search_global)
jedi.evaluate.representation.Function.__init__(self, evaluator, fun_apply, is_decorated)
jedi.evaluate.representation.FunctionExecution.param_by_name(self, name)
jedi.evaluate.representation.FunctionExecution.name_for_position(self, position)
jedi.evaluate.representation.FunctionExecution.names_dicts(self, search_global)
jedi.evaluate.representation.FunctionExecution.wrapper(obj)
jedi.evaluate.representation.FunctionExecution.__init__(self, evaluator, base)
jedi.evaluate.representation.FunctionExecution.wrapper(obj)
jedi.evaluate.representation.FunctionExecution.__getattr__(self, name)
jedi.evaluate.representation.FunctionExecution.__repr__(self)
jedi.evaluate.representation.FunctionExecution._scope_copy(self, scope)
jedi.evaluate.representation.FunctionExecution._copy_list(self, lst)
jedi.evaluate.representation.GlobalName.__init__(self, name)
jedi.evaluate.representation.Instance.py__bool__(self)
jedi.evaluate.representation.Instance.execute_subscope_by_name(self, name)
jedi.evaluate.representation.Instance.wrapper(obj)
jedi.evaluate.representation.Instance.get_subscope_by_name(self, name)
jedi.evaluate.representation.Instance.get_descriptor_returns(self, obj)
jedi.evaluate.representation.Instance.py__class__(self, evaluator)
jedi.evaluate.representation.Instance.get_index_types(self, evaluator, index_array)
jedi.evaluate.representation.Instance._self_names_dict(self, add_mro)
jedi.evaluate.representation.Instance.__getattr__(self, name)
jedi.evaluate.representation.Instance.__init__(self, evaluator, base, var_args, is_generated)
jedi.evaluate.representation.Instance._get_func_self_name(self, fun_apply)
jedi.evaluate.representation.Instance.wrapper(obj)
jedi.evaluate.representation.Instance.__repr__(self)
jedi.evaluate.representation.InstanceElement.is_scope(self)
jedi.evaluate.representation.InstanceElement.get_decorated_func(self)
jedi.evaluate.representation.InstanceElement.__iter__(self)
jedi.evaluate.representation.InstanceElement.isinstance(self)
jedi.evaluate.representation.InstanceElement.__init__(self, evaluator, instance, var, is_class_var)
jedi.evaluate.representation.InstanceElement.py__call__(self, evaluator, params)
jedi.evaluate.representation.InstanceElement.__getitem__(self, index)
jedi.evaluate.representation.InstanceElement.__getattr__(self, name)
jedi.evaluate.representation.InstanceElement.get_rhs(self)
jedi.evaluate.representation.InstanceElement.__repr__(self)
jedi.evaluate.representation.InstanceElement.get_definition(self)
jedi.evaluate.representation.InstanceElement.is_definition(self)
jedi.evaluate.representation.InstanceElement.get_parent_until(self)
jedi.evaluate.representation.InstanceName.is_definition(self)
jedi.evaluate.representation.InstanceName.__init__(self, origin_name, parent)
jedi.evaluate.representation.LambdaWrapper.get_decorated_func(self)
jedi.evaluate.representation.LazyInstanceDict.__getitem__(self, name)
jedi.evaluate.representation.LazyInstanceDict.values(self)
jedi.evaluate.representation.LazyInstanceDict.__init__(self, evaluator, instance, dct)
jedi.evaluate.representation.ModuleWrapper.wrapper(obj)
jedi.evaluate.representation.ModuleWrapper._get_init_directory(self)
jedi.evaluate.representation.ModuleWrapper.py__package__(self)
jedi.evaluate.representation.ModuleWrapper.wrapper(obj)
jedi.evaluate.representation.ModuleWrapper.__getattr__(self, name)
jedi.evaluate.representation.ModuleWrapper.wrapper(obj)
jedi.evaluate.representation.ModuleWrapper.py__file__(self)
jedi.evaluate.representation.ModuleWrapper.names_dicts(self, search_global)
jedi.evaluate.representation.ModuleWrapper.__repr__(self)
jedi.evaluate.representation.ModuleWrapper.__init__(self, evaluator, module)
jedi.evaluate.representation.ModuleWrapper.py__name__(self)
jedi.evaluate.representation.Python3Method.__init__(self, fun_apply)
jedi.evaluate.representation.Python3Method.__get__(self, obj, objtype)
jedi.evaluate.representation.Wrapper.is_scope(self)
jedi.evaluate.representation.Wrapper.py__bool__(self)
jedi.evaluate.representation.Wrapper.is_class(self)
jedi.evaluate.representation.cache_star_import(fun_apply)
jedi.evaluate.representation.get_instance_el(evaluator, instance, var, is_class_var)
jedi.evaluate.representation.memoize_default(default, evaluator_is_first_arg, second_arg_is_evaluator)
jedi.evaluate.representation.underscore_memoization(fun_apply)
jedi.evaluate.representation.use_metaclass(meta)

 
 
jedi.evaluate.stdlib.Parser._add_syntax_error(self, message, position)
jedi.evaluate.stdlib.Parser.remove_last_newline(self)
jedi.evaluate.stdlib.Parser.error_recovery(self, grammar, stack, typ, value, start_pos, prefix, add_token_callback)
jedi.evaluate.stdlib.Parser.__repr__(self)
jedi.evaluate.stdlib.Parser.__init__(self, grammar, source, module_path, tokenizer)
jedi.evaluate.stdlib.Parser.convert_node(self, grammar, type, children)
jedi.evaluate.stdlib.Parser.convert_leaf(self, grammar, type, value, prefix, start_pos)
jedi.evaluate.stdlib.Parser._stack_removal(self, grammar, stack, start_index, value, start_pos)
jedi.evaluate.stdlib.Parser._tokenize(self, tokenizer)
jedi.evaluate.stdlib.SuperInstance.__init__(self, evaluator, cls)
jedi.evaluate.stdlib._follow_param(evaluator, params, index)
jedi.evaluate.stdlib.wrapper(evaluator, obj, arguments)
jedi.evaluate.stdlib.argument_clinic(string, want_obj, want_scope)
jedi.evaluate.stdlib.wrapper(evaluator, obj, arguments)
jedi.evaluate.stdlib.wrapper(evaluator, obj, arguments)
jedi.evaluate.stdlib.wrapper(evaluator, obj, arguments)
jedi.evaluate.stdlib.wrapper(evaluator, obj, arguments)
jedi.evaluate.stdlib.wrapper(evaluator, obj, arguments)
jedi.evaluate.stdlib.collections_namedtuple(evaluator, obj, params)
jedi.evaluate.stdlib.execute(evaluator, obj, params)

 
 
jedi.evaluate.sys_path.Parser._add_syntax_error(self, message, position)
jedi.evaluate.sys_path.Parser.remove_last_newline(self)
jedi.evaluate.sys_path.Parser.error_recovery(self, grammar, stack, typ, value, start_pos, prefix, add_token_callback)
jedi.evaluate.sys_path.Parser.__repr__(self)
jedi.evaluate.sys_path.Parser.__init__(self, grammar, source, module_path, tokenizer)
jedi.evaluate.sys_path.Parser.convert_node(self, grammar, type, children)
jedi.evaluate.sys_path.Parser.convert_leaf(self, grammar, type, value, prefix, start_pos)
jedi.evaluate.sys_path.Parser._stack_removal(self, grammar, stack, start_index, value, start_pos)
jedi.evaluate.sys_path.Parser._tokenize(self, tokenizer)
jedi.evaluate.sys_path._check_module(evaluator, module)
jedi.evaluate.sys_path._detect_django_path(module_path)
jedi.evaluate.sys_path._execute_code(module_path, code)
jedi.evaluate.sys_path._get_buildout_scripts(module_path)
jedi.evaluate.sys_path._get_parent_dir_with_file(path, filename)
jedi.evaluate.sys_path._get_paths_from_buildout_script(evaluator, buildout_script)
jedi.evaluate.sys_path._get_venv_sitepackages(venv)
jedi.evaluate.sys_path._paths_from_assignment(evaluator, expr_stmt)
jedi.evaluate.sys_path._paths_from_list_modifications(module_path, trailer1, trailer2)
jedi.evaluate.sys_path.exec_function(source, global_map)
jedi.evaluate.sys_path.memoize_default(default, evaluator_is_first_arg, second_arg_is_evaluator)
jedi.evaluate.sys_path.wrapper(obj)
jedi.evaluate.sys_path.traverse_parents(path)

 
 
jedi.parser.ErrorStatement.__init__(self, stack, next_token, position_modifier, next_start_pos)
jedi.parser.Parser._add_syntax_error(self, message, position)
jedi.parser.Parser.remove_last_newline(self)
jedi.parser.Parser.error_recovery(self, grammar, stack, typ, value, start_pos, prefix, add_token_callback)
jedi.parser.Parser.__repr__(self)
jedi.parser.Parser.__init__(self, grammar, source, module_path, tokenizer)
jedi.parser.Parser.convert_node(self, grammar, type, children)
jedi.parser.Parser.convert_leaf(self, grammar, type, value, prefix, start_pos)
jedi.parser.Parser._stack_removal(self, grammar, stack, start_index, value, start_pos)
jedi.parser.Parser._tokenize(self, tokenizer)
jedi.parser.ParserSyntaxError.__init__(self, message, position)
jedi.parser.PgenParser.shift(self, type, value, newstate, prefix, start_pos)
jedi.parser.PgenParser.pop(self)
jedi.parser.PgenParser.parse(self, tokenizer)
jedi.parser.PgenParser.addtoken(self, type, value, prefix, start_pos)
jedi.parser.PgenParser.push(self, type, newdfa, newstate)
jedi.parser.PgenParser.__init__(self, grammar, convert_node, convert_leaf, error_recovery)
jedi.parser.generate_grammar(filename)
jedi.parser.load_grammar(file)

 
 
jedi.parser.fast.CachedFastParser.__call__(self, grammar, source, module_path)
jedi.parser.fast.FastModule.reset_caches(self)
jedi.parser.fast.FastModule.__repr__(self)
jedi.parser.fast.FastModule.__init__(self, module_path)
jedi.parser.fast.FastParser.update(self, source)
jedi.parser.fast.FastParser._get_node(self, source, parser_code, line_offset, nodes)
jedi.parser.fast.FastParser._split_parts(self, source)
jedi.parser.fast.FastParser._parse(self, source)
jedi.parser.fast.FastParser._reset_caches(self)
jedi.parser.fast.FastParser.__init__(self, grammar, source, module_path)
jedi.parser.fast.FastTokenizer.__next__(self)
jedi.parser.fast.FastTokenizer.next(self)
jedi.parser.fast.FastTokenizer._finish_dedents(self)
jedi.parser.fast.FastTokenizer.__iter__(self)
jedi.parser.fast.FastTokenizer._close(self)
jedi.parser.fast.FastTokenizer.__init__(self, source)
jedi.parser.fast.FastTokenizer._get_prefix(self)
jedi.parser.fast.MergedNamesDict.values(self)
jedi.parser.fast.MergedNamesDict.__getitem__(self, value)
jedi.parser.fast.MergedNamesDict.items(self)
jedi.parser.fast.MergedNamesDict.__iter__(self)
jedi.parser.fast.MergedNamesDict.__init__(self, dicts)
jedi.parser.fast.Parser._add_syntax_error(self, message, position)
jedi.parser.fast.Parser.remove_last_newline(self)
jedi.parser.fast.Parser.error_recovery(self, grammar, stack, typ, value, start_pos, prefix, add_token_callback)
jedi.parser.fast.Parser.__repr__(self)
jedi.parser.fast.Parser.__init__(self, grammar, source, module_path, tokenizer)
jedi.parser.fast.Parser.convert_node(self, grammar, type, children)
jedi.parser.fast.Parser.convert_leaf(self, grammar, type, value, prefix, start_pos)
jedi.parser.fast.Parser._stack_removal(self, grammar, stack, start_index, value, start_pos)
jedi.parser.fast.Parser._tokenize(self, tokenizer)
jedi.parser.fast.ParserNode.all_sub_nodes(self)
jedi.parser.fast.ParserNode.add_node(self, node, line_offset)
jedi.parser.fast.ParserNode.wrapper(self)
jedi.parser.fast.ParserNode._rewrite_last_newline(self)
jedi.parser.fast.ParserNode.__repr__(self)
jedi.parser.fast.ParserNode.close(self)
jedi.parser.fast.ParserNode.parent_until_indent(self, indent)
jedi.parser.fast.ParserNode.reset_node(self)
jedi.parser.fast.ParserNode.__init__(self, fast_module, parser, source)
jedi.parser.fast.source_tokens(source)
jedi.parser.fast.use_metaclass(meta)

 
 

 
 
jedi.parser.pgen2.grammar.Grammar.load(self, filename)
jedi.parser.pgen2.grammar.Grammar.dump(self, filename)
jedi.parser.pgen2.grammar.Grammar.report(self)
jedi.parser.pgen2.grammar.Grammar.copy(self)
jedi.parser.pgen2.grammar.Grammar.__init__(self)

 
 
jedi.parser.pgen2.parse.ParseError.__init__(self, msg, type, value, start_pos)
jedi.parser.pgen2.parse.PgenParser.shift(self, type, value, newstate, prefix, start_pos)
jedi.parser.pgen2.parse.PgenParser.pop(self)
jedi.parser.pgen2.parse.PgenParser.parse(self, tokenizer)
jedi.parser.pgen2.parse.PgenParser.addtoken(self, type, value, prefix, start_pos)
jedi.parser.pgen2.parse.PgenParser.push(self, type, newdfa, newstate)
jedi.parser.pgen2.parse.PgenParser.__init__(self, grammar, convert_node, convert_leaf, error_recovery)

 
 
jedi.parser.pgen2.pgen.DFAState.unifystate(self, old, new)
jedi.parser.pgen2.pgen.DFAState.addarc(self, next, label)
jedi.parser.pgen2.pgen.DFAState.__eq__(self, other)
jedi.parser.pgen2.pgen.DFAState.__init__(self, nfaset, final)
jedi.parser.pgen2.pgen.NFAState.addarc(self, next, label)
jedi.parser.pgen2.pgen.NFAState.__init__(self)
jedi.parser.pgen2.pgen.ParserGenerator.raise_error(self, msg)
jedi.parser.pgen2.pgen.ParserGenerator.make_grammar(self)
jedi.parser.pgen2.pgen.ParserGenerator.calcfirst(self, name)
jedi.parser.pgen2.pgen.ParserGenerator.make_dfa(self, start, finish)
jedi.parser.pgen2.pgen.ParserGenerator.parse(self)
jedi.parser.pgen2.pgen.ParserGenerator.addfirstsets(self)
jedi.parser.pgen2.pgen.ParserGenerator.make_first(self, c, name)
jedi.parser.pgen2.pgen.ParserGenerator.expect(self, type, value)
jedi.parser.pgen2.pgen.ParserGenerator.__init__(self, filename, stream)
jedi.parser.pgen2.pgen.ParserGenerator.parse_item(self)
jedi.parser.pgen2.pgen.ParserGenerator.parse_rhs(self)
jedi.parser.pgen2.pgen.ParserGenerator.simplify_dfa(self, dfa)
jedi.parser.pgen2.pgen.ParserGenerator.dump_dfa(self, name, dfa)
jedi.parser.pgen2.pgen.ParserGenerator.parse_alt(self)
jedi.parser.pgen2.pgen.ParserGenerator.dump_nfa(self, name, start, finish)
jedi.parser.pgen2.pgen.ParserGenerator.make_label(self, c, label)
jedi.parser.pgen2.pgen.ParserGenerator.parse_atom(self)
jedi.parser.pgen2.pgen.ParserGenerator.gettoken(self)
jedi.parser.pgen2.pgen.generate_grammar(filename)

 
 
jedi.parser.token.ISEOF(x)
jedi.parser.token.ISNONTERMINAL(x)
jedi.parser.token.ISTERMINAL(x)

 
 
jedi.parser.tokenize.generate_tokens(readline)
jedi.parser.tokenize.<lambda>(s)
jedi.parser.tokenize.source_tokens(source)

 
 
jedi.parser.tree.AssertStmt.assertion(self)
jedi.parser.tree.Base.is_scope(self)
jedi.parser.tree.Base.get_parent_scope(self, include_flows)
jedi.parser.tree.Base.isinstance(self)
jedi.parser.tree.BaseNode.move(self, line_offset, column_offset)
jedi.parser.tree.BaseNode.__init__(self, children)
jedi.parser.tree.BaseNode.first_leaf(self)
jedi.parser.tree.BaseNode.get_code(self)
jedi.parser.tree.BaseNode.wrapper(self)
jedi.parser.tree.Class.get_super_arglist(self)
jedi.parser.tree.Class.__init__(self, children)
jedi.parser.tree.ClassOrFunc.get_decorators(self)
jedi.parser.tree.CompFor.is_scope(self)
jedi.parser.tree.CompFor.names_dicts(self, search_global)
jedi.parser.tree.CompFor.get_defined_names(self)
jedi.parser.tree.ExprStmt.get_defined_names(self)
jedi.parser.tree.ExprStmt.get_rhs(self)
jedi.parser.tree.ExprStmt.first_operation(self)
jedi.parser.tree.Function.is_generator(self)
jedi.parser.tree.Function.get_call_signature(self, width, func_name)
jedi.parser.tree.Function.annotation(self)
jedi.parser.tree.Function.__init__(self, children)
jedi.parser.tree.GlobalStmt.get_global_names(self)
jedi.parser.tree.GlobalStmt.get_defined_names(self)
jedi.parser.tree.IfStmt.check_nodes(self)
jedi.parser.tree.IfStmt.node_in_which_check_node(self, node)
jedi.parser.tree.IfStmt.node_after_else(self, node)
jedi.parser.tree.Import.is_star_import(self)
jedi.parser.tree.Import.is_nested(self)
jedi.parser.tree.Import.path_for_name(self, name)
jedi.parser.tree.ImportFrom.paths(self)
jedi.parser.tree.ImportFrom.star_import_name(self)
jedi.parser.tree.ImportFrom.get_from_names(self)
jedi.parser.tree.ImportFrom._as_name_tuples(self)
jedi.parser.tree.ImportFrom.get_defined_names(self)
jedi.parser.tree.ImportFrom.aliases(self)
jedi.parser.tree.ImportName.paths(self)
jedi.parser.tree.ImportName.is_nested(self)
jedi.parser.tree.ImportName._dotted_as_names(self)
jedi.parser.tree.ImportName.get_defined_names(self)
jedi.parser.tree.ImportName.aliases(self)
jedi.parser.tree.IsScopeMeta.__instancecheck__(self, other)
jedi.parser.tree.Keyword.__ne__(self, other)
jedi.parser.tree.Keyword.__hash__(self)
jedi.parser.tree.Keyword.__eq__(self, other)
jedi.parser.tree.Lambda.is_generator(self)
jedi.parser.tree.Lambda.__repr__(self)
jedi.parser.tree.Lambda.yields(self)
jedi.parser.tree.Lambda.__init__(self, children)
jedi.parser.tree.Leaf.next_sibling(self)
jedi.parser.tree.Leaf.move(self, line_offset, column_offset)
jedi.parser.tree.Leaf.prev_sibling(self)
jedi.parser.tree.Leaf.get_previous(self)
jedi.parser.tree.Leaf.wrapper(self)
jedi.parser.tree.Leaf.get_code(self)
jedi.parser.tree.Leaf.__init__(self, position_modifier, value, start_pos, prefix)
jedi.parser.tree.Literal.eval(self)
jedi.parser.tree.Module.__init__(self, children)
jedi.parser.tree.Name.assignment_indexes(self)
jedi.parser.tree.Name.__str__(self)
jedi.parser.tree.Name.__repr__(self)
jedi.parser.tree.Name.get_definition(self)
jedi.parser.tree.Name.is_definition(self)
jedi.parser.tree.Name.__unicode__(self)
jedi.parser.tree.Node.__repr__(self)
jedi.parser.tree.Node.__init__(self, type, children)
jedi.parser.tree.Operator.__ne__(self, other)
jedi.parser.tree.Operator.__hash__(self)
jedi.parser.tree.Operator.__eq__(self, other)
jedi.parser.tree.Operator.__str__(self)
jedi.parser.tree.Param._tfpdef(self)
jedi.parser.tree.Param.__repr__(self)
jedi.parser.tree.Param.annotation(self)
jedi.parser.tree.Param.__init__(self, children, parent)
jedi.parser.tree.PositionModifier.__init__(self)
jedi.parser.tree.Python3Method.__init__(self, fun_apply)
jedi.parser.tree.Python3Method.__get__(self, obj, objtype)
jedi.parser.tree.Scope.is_scope(self)
jedi.parser.tree.Scope.walk(self)
jedi.parser.tree.Scope.__init__(self, children)
jedi.parser.tree.Scope.__repr__(self)
jedi.parser.tree.TryStmt.except_clauses(self)
jedi.parser.tree.WithStmt.node_from_name(self, name)
jedi.parser.tree.WithStmt.get_defined_names(self)
jedi.parser.tree._create_params(parent, argslist_list)
jedi.parser.tree._defined_names(current)
jedi.parser.tree.cleandoc(doc)
jedi.parser.tree.is_node(node)
jedi.parser.tree.literal_eval(string)
jedi.parser.tree.use_metaclass(meta)
jedi.parser.tree.utf8_repr(fun_apply)

 
 
jedi.parser.user_context.FastParser.update(self, source)
jedi.parser.user_context.FastParser._get_node(self, source, parser_code, line_offset, nodes)
jedi.parser.user_context.FastParser._split_parts(self, source)
jedi.parser.user_context.FastParser._parse(self, source)
jedi.parser.user_context.FastParser._reset_caches(self)
jedi.parser.user_context.FastParser.__init__(self, grammar, source, module_path)
jedi.parser.user_context.Parser._add_syntax_error(self, message, position)
jedi.parser.user_context.Parser.remove_last_newline(self)
jedi.parser.user_context.Parser.error_recovery(self, grammar, stack, typ, value, start_pos, prefix, add_token_callback)
jedi.parser.user_context.Parser.__repr__(self)
jedi.parser.user_context.Parser.__init__(self, grammar, source, module_path, tokenizer)
jedi.parser.user_context.Parser.convert_node(self, grammar, type, children)
jedi.parser.user_context.Parser.convert_leaf(self, grammar, type, value, prefix, start_pos)
jedi.parser.user_context.Parser._stack_removal(self, grammar, stack, start_index, value, start_pos)
jedi.parser.user_context.Parser._tokenize(self, tokenizer)
jedi.parser.user_context.PushBackIterator.push_back(self, value)
jedi.parser.user_context.PushBackIterator.__next__(self)
jedi.parser.user_context.PushBackIterator.__iter__(self)
jedi.parser.user_context.PushBackIterator.__init__(self, iterator)
jedi.parser.user_context.PushBackIterator.next(self)
jedi.parser.user_context.UserContext._backwards_line_generator(self, start_pos)
jedi.parser.user_context.UserContext.get_path_under_cursor(self)
jedi.parser.user_context.UserContext.get_line(self, line_nr)
jedi.parser.user_context.UserContext._calc_path_until_cursor(self, start_pos)
jedi.parser.user_context.UserContext._get_backwards_tokenizer(self, start_pos, line_gen)
jedi.parser.user_context.UserContext.get_position_line(self)
jedi.parser.user_context.UserContext.get_path_after_cursor(self)
jedi.parser.user_context.UserContext.get_operator_under_cursor(self)
jedi.parser.user_context.UserContext.call_signature(self)
jedi.parser.user_context.UserContext.get_context(self, yield_positions)
jedi.parser.user_context.UserContext.wrapper(self)
jedi.parser.user_context.UserContext.__init__(self, source, position)
jedi.parser.user_context.UserContextParser.wrapper(self)
jedi.parser.user_context.UserContextParser.module(self)
jedi.parser.user_context.UserContextParser.__init__(self, grammar, source, path, position, user_context, parser_done_callback, use_fast_parser)
jedi.parser.user_context.UserContextParser.wrapper(self)
jedi.parser.user_context.UserContextParser.wrapper(self)
jedi.parser.user_context.UserContextParser.wrapper(self)
jedi.parser.user_context.u(string)

 
 
jedi.refactoring.Refactoring.old_files(self)
jedi.refactoring.Refactoring.new_files(self)
jedi.refactoring.Refactoring.diff(self)
jedi.refactoring.Refactoring.__init__(self, change_dct)
jedi.refactoring._rename(names, replace_str)
jedi.refactoring.extract(script, new_name)
jedi.refactoring.inline(script)
jedi.refactoring.rename(script, new_name)

 
 

 
 
jedi.utils.Interpreter.__init__(self, source, namespaces)
jedi.utils.Interpreter._simple_complete(self, path, dot, like)
jedi.utils.UserContext._backwards_line_generator(self, start_pos)
jedi.utils.UserContext.get_path_under_cursor(self)
jedi.utils.UserContext.get_line(self, line_nr)
jedi.utils.UserContext._calc_path_until_cursor(self, start_pos)
jedi.utils.UserContext._get_backwards_tokenizer(self, start_pos, line_gen)
jedi.utils.UserContext.get_position_line(self)
jedi.utils.UserContext.get_path_after_cursor(self)
jedi.utils.UserContext.get_operator_under_cursor(self)
jedi.utils.UserContext.call_signature(self)
jedi.utils.UserContext.get_context(self, yield_positions)
jedi.utils.UserContext.wrapper(self)
jedi.utils.UserContext.__init__(self, source, position)
jedi.utils.completion_parts(path_until_cursor)
jedi.utils.namedtuple(typename, field_names, verbose, rename)
jedi.utils.setup_readline(namespace_module)








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





#------http://stackoverflow.com/questions/8718885/import-module-from-string-variable
list and process it with pydoc
pprint:


import pydoc
!pydoc sys





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


  
  
 '''
  
  
  