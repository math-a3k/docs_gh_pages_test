# -*- coding: utf-8 -*---------------------------------------------
#------JPYPE for JAVA JAR integration-------------------------------
'''
!pip install JPype1-py3
#Java Class
#https://docs.oracle.com/javase/8/docs/api/java/lang/Math.html
import jpype as jp
jp.startJVM(jp.getDefaultJVMPath(), "-ea")
jp.java.lang.System.out.println("hello world")
jp.shutdownJVM()
'''
import jpype as jp; import numpy as np; import os as os
jarpath= r"D:\_devs\Python01\project\zjavajar\\"
mavenurl= r"http://mvnrepository.com/artifact/org.springframework"


#http://www.ibm.com/developerworks/library/j-5things6/




#--------------------JPYPE library--------------------------------------------
# Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
def importJAR(path1="", path2="", path3="", path4=""):
   classpath = path1
   if path2 != "":  classpath = os.pathsep.join((classpath, path2))   
   if path3 != "":  classpath = os.pathsep.join((classpath, path3))
   if path4 != "":  classpath = os.pathsep.join((classpath, path4))        
   jp.startJVM(jp.getDefaultJVMPath(),"-ea", "-Djava.class.path=%s" % classpath)

#path1= r"D:\_devs\Python01\project\zjavajar\tika-app-1.12.jar"
#importJAR(path1)

#  looping  on the folder , loop on jar and maven     

def listallfile(some_dir, pattern="*.*", dirlevel=1):
  import fnmatch; import os;  matches = []
  some_dir = some_dir.rstrip(os.path.sep)
  assert os.path.isdir(some_dir);  num_sep = some_dir.count(os.path.sep)
  for root, dirs, files in os.walk(some_dir):
 #   yield root, dirs, files
    num_sep_this = root.count(os.path.sep)
    if num_sep + dirlevel <= num_sep_this: del dirs[:]
    for files in fnmatch.filter(files, pattern):
      matches.append(os.path.join(root, files))     
  return matches            
     
# DIRCWD=r"D:\_devs\Python01\project"
# listallfile(DIRCWD, "*.*", 2)


def importFolderJAR(dir1="", dirlevel=1):
   vv=  listallfile(dir1, "*.jar", dirlevel);    classpath = ""
   for jar1 in vv:
       classpath = os.pathsep.join((classpath, jar1))     
   jp.startJVM(jp.getDefaultJVMPath(),"-ea", "-Djava.class.path=%s" % classpath)

#path1= r"D:\_devs\Python01\project\zjavajar\"
#importFolderJAR(path1)


def importFromMaven():
    return 0
    
    
    

def showLoadedClass(): #Code to see the JAR loaded.
   classloader = jp.java.lang.ClassLoader.getSystemClassLoader(); vv= []; 
   for x in classloader.getURLs():  vv.append(x.toString());         
   return vv
     
#showLoadedClass()
#vv= ["" for x in range(10)]


def inspectJAR(dir1) :
 import zipfile
 archive = zipfile.ZipFile('<path to jar file>/test.jar', 'r')
 list = archive.namelist()





def loadSingleton(class1):  single= jp.JClass(class1);  return Single.getInstance()


def java_print(x):  jp.java.lang.System.out.println(x)   #Print in Java Console

 
def compileJAVA(javafile):
 import  subprocess, os
 compath1=  os.getenv("JAVA_HOME")+"javac"
 subprocess.check_call([compath1, javafile])
   
   
def writeText(text, filename) :
 text_file = open(filename, "w"); text_file.write(text); text_file.close()


def compileJAVAtext(classname, javatxt, path1=""):  
 import os 
 if path1 != "": path1=  os.getcwd()
 path1= path1+"\\"+classname+".java"
 text_file=open(path1, "w"); text_file.write(javatxt); text_file.close()
 compileJAVA(path1)


javacode1=""" 
public class Test {
    private String msg;
    public Test() {   msg = "nothing so far...";  }
    public static void speak(String msg) {    System.out.println(msg);    }
    public void setString(String s) {     msg = s; }
    public String getString() {  return msg; }
}
"""

#compileJAVAtext("Test", javacode1, path1="")


#---------------Execute in Command Line --> JAVA MAIN-----------------------
def execute_javamain(java_file): 
  import STDOUT, PIPE, subprocess 
  jvmpath1=  os.getenv("JAVA_HOME")+"java"
  cmd=[jvmpath1, java_file]
  proc=subprocess.Popen(cmd, stdout = PIPE, stderr = STDOUT) 
 # input = subprocess.Popen(cmd, stdin = PIPE)
  print(proc.stdout.read()) 

#  execute_java("CsMain")



def javaerror(jpJavaException):
    print("Caught the runtime exception : ", jpJavaException.message())
    print(jpJavaException.stackTrace())


#--------Error Catch for Java runtime--------------------------
'''
try :
    
 # Code that throws a java.lang.RuntimeException
 jp.java.lang.System.out.println("its meetup running Java Code")
 aa= jp.java.lang.Math.cos(5/0.0)
    
except jp.JException(jp.java.lang.RuntimeException) : javaerror(jp.JavaException)
'''




#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------










#-----------------------PDF BOX ---------------------------------------------
def launchPDFbox():
 path1= jarpath + r"commons-logging-1.2.jar"
 path2= jarpath + r"pdfbox-app-1.8.11.jar"
 classpath = os.pathsep.join((path1, path2))
 jp.startJVM(jp.getDefaultJVMPath(),"-ea", "-Djava.class.path=%s" % classpath)
 showLoadedClass()
 
 
def getfpdffulltext(pdfile1):
 pd = jp.JClass("org.apache.pdfbox.pdmodel.PDDocument")
 document = pd.load(pdfile1)
 Text1 = jp.JClass("org.apache.pdfbox.util.PDFTextStripper");  text = Text1()
 textDoc = text.getText(document); document.close()
 return textDoc

#launchPDFbox()
#txt= getpdffulltext(r"D:\_devs\Python01\project\zjavajar\japanese01.pdf")
#vtxt= txt.split('。')  # --> ['Line 1', 'Line 2', 'Line 3']
#print(txt)






#----------- TIKA for Parsing-------------------------------------------
def launchTIKA():
 path1= jarpath + r"commons-logging-1.2.jar"; path2= jarpath + r"tika-app-1.12.jar"
 importJAR(path1, path2);   showLoadedClass()

#    launchTIKA()

def getfulltext(file1,withMeta=0 ): 
  file2= jp.java.io.FileInputStream(file1);
  handler =   jp.JClass("org.apache.tika.sax.BodyContentHandler")(1010241024); #MaxSizeText
  metadata =  jp.JClass("org.apache.tika.metadata.Metadata")();
  pcontext  = jp.JClass("org.apache.tika.parser.ParseContext")();
  parser= jp.JClass("org.apache.tika.parser.AutoDetectParser")() 

  parser.parse(file2, handler, metadata,pcontext); txt1=  handler.toString();
  
  if withMeta ==1 : #Print meta
   metatxt= "";  lname= metadata.names()
   for name in lname : metatxt= metatxt + name + ": "+ metadata.get(name) + "\n"
   txt1= metatxt + "\n\n" + txt1
  return txt1 

#  txt2= getfulltext(r"D:\\_devs\\Python01\\project\\detail.doc",1)


#------------Get Full Text for one directory, Store in Panda---------------------
def directorygetalltext(dir1, filetype1="*.*", withMeta=0, fileout=""):
 import glob; import pandas as pd;  vv= []
 for file1 in glob.glob(dir1 + "\\" + filetype1) :
     txt= getfulltext(file1,withMeta);  
     vv.append(txt) 
 if fileout!="":  #Store in Pandas the content
   pdf= pd.DataFrame(vv);  st= pd.HDFStore(fileout);  st.append('data', pdf); del pd    
 return vv


 
#  DIRCWD=r"D:\_devs\Python01\project";     fileout= r'E:\_data\all_text01.h5'
#  vv2= directorygetalltext(DIRCWD, "*.*", 0, fileout)  #=No Metadata




#------------Get Full Text for one directory, Store in Panda---------------------
def directorygetalltext2(dir1, filetype1="*.*", type1=0, fileout=""):
 import glob; import pandas as pd;  vv= [[]]; i=0
 for file1 in glob.glob(dir1 + "\\" + filetype1) :
     txt= getfulltext(file1,type1);  
     vv[i,0]= file1;  vv[i,1]= txt;  i+=1
 if fileout!="":  #Store in Pandas the content
   pdf= pd.DataFrame(vv);  st= pd.HDFStore(fileout);  st.append('data', pdf); del pd    
 return vv
 
 
#  DIRCWD=r"D:\_devs\Python01\project";    fileout= r'E:\_data\all_text01.h5'
#  vv2= directorygetalltext(DIRCWD, "*.*", 0, fileout)







#------------Spring DTS Load--------------------------------------------------
# http://mvnrepository.com/artifact/org.springframework
















#---------------Example 2 ----------------------------------
javacode1=""" 
package org.wg3i.test;
 
public class Test {
    private String msg;
    public Test() {   msg = "nothing so far...";  }
    public static void speak(String msg) {    System.out.println(msg);    }
    public void setString(String s) {     msg = s; }
    public String getString() {  return msg; }
}
"""




'''

import jpype as jp; import numpy as np

jp.startJVM(jp.getDefaultJVMPath(), "-ea")
jp.java.lang.System.out.println("hello world")

jp.java.lang.System.out.println("it's meetup running Java Code")

aa=8; 
aa= jp.java.lang.Math.abs(-5)
aa


values = np.arange(7)
java_array = jp.JArray(jp.JDouble, 1)(values.tolist())

for item in java_array:
 jp.java.lang.System.out.println(item)
 
 
 util = jpype.JPackage("java.util")
4
al = util.ArrayList()
5
al.add(1)
6
al.add(2)


The problem is due to a missing class, sadly JNI doesn't give enough information to find where the problem is.
After some tests, I got a working version.

1/ The classpath option is case sensitive, so you should use -Djava.class.path.
2/ PDFBox requires Apache Commons Logging, so it must be added in the classpath.
3/ jp.JString shoudln't be used for conversion: it represents a string from the Java world, but doesn't act like it. As the value is already in the Python world, it should be converted using str().

'''


#---------------Tutorial ----------------------------------------
#http://www.tutorialspoint.com/tika/


javacode1=""" 
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

import org.apache.tika.exception.TikaException;
import org.apache.tika.metadata.Metadata;
import org.apache.tika.parser.ParseContext;
import org.apache.tika.parser.pdf.PDFParser;
import org.apache.tika.sax.BodyContentHandler;

import org.xml.sax.SAXException;

public class PdfParse {

   public static void main(final String[] args) throws IOException,TikaException {

      BodyContentHandler handler = new BodyContentHandler();
      Metadata metadata = new Metadata();
      FileInputStream inputstream = new FileInputStream(new File("Example.pdf"));
      ParseContext pcontext = new ParseContext();
      
      //parsing the document using PDF parser
      PDFParser pdfparser = new PDFParser(); 
      pdfparser.parse(inputstream, handler, metadata,pcontext);
      
      //getting the content of the document
      System.out.println("Contents of the PDF :" + handler.toString());
      
      //getting metadata of the document
      System.out.println("Metadata of the PDF:");
      String[] metadataNames = metadata.names();
      
      for(String name : metadataNames) {
         System.out.println(name+ " : " + metadata.get(name));
      }
   }
}


# Command Line parson
# javac PdfParse.java
# java PdfParse

""" 























