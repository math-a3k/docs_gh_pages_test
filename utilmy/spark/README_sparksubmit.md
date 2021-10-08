To create egg file:
===================
1. Change your current folder to root folder of the package
cd <zesresr>
2. Run below command
setup.py bdist_egg


Steps to run Job:
================
Note: Please make sure that below file have executable permissions, if not then please run below command:
	chmod +x script/sparkrunscript.sh 

1. Change current folder to root of the package:
cd /home/testuser/zesresr
b. Run below shell script:
./script/sparkrunscript.sh 
























```
######  Load userlog table                                                                   
                              0                    1                                                  
loggeddate           2015-07-22           2015-07-22                                                  
ds          22-07-2015 11:40:06  22-07-2015 11:40:07                                                  
y                            27                   62                                                  
                              0                    1                                                  
loggeddate           2015-07-22           2015-07-22                                                  
ds          22-07-2015 11:40:06  22-07-2015 11:40:07                                                  
y                            27                   62                                                  
####### Train  Run  ##############################                                                    
   loggeddate                   ds   y                                                                
0  2015-07-22  22-07-2015 11:40:06  27                                                                
1  2015-07-22  22-07-2015 11:40:07  62                                                                
####### Prediction Run ###########################                                                    
####### Prediction output ########################                                                    
+-------------------+-----+----------+-----------+-------------+                                      
|                 ds|    y|y_pred_max|y_pred_mean|training_date|                                      
+-------------------+-----+----------+-----------+-------------+                                      
|22-07-2015 11:40:06| 27.0|   272.092|       85.0|   2021-06-05|                                      
|22-07-2015 11:40:07| 62.0|   272.092|       85.0|   2021-06-05|                                      
|22-07-2015 11:40:08| 56.0|   272.092|       85.0|   2021-06-05|                                      
|22-07-2015 11:40:09|112.0|   272.092|       85.0|   2021-06-05|                                      
|22-07-2015 11:40:10| 58.0|   272.092|       85.0|   2021-06-05|                                      
|22-07-2015 11:40:11| 58.0|   272.092|       85.0|   2021-06-05|                                      
|22-07-2015 11:40:12| 67.0|   272.092|       85.0|   2021-06-05|                                      
|22-07-2015 11:40:13| 85.0|   272.092|       85.0|   2021-06-05|                                      
|22-07-2015 11:40:14|160.0|   272.092|       85.0|   2021-06-05|                                      
|22-07-2015 11:40:15| 57.0|   272.092|       85.0|   2021-06-05|                                      
+-------------------+-----+----------+-----------+-------------+                                      
only showing top 10 rows                                                                              
                                                                                                      
####### Predict check: 4269                                                                           
                                 0                    1                                               
ds             22-07-2015 11:40:06  22-07-2015 11:40:07                                               
y                               27                   62                                               
y_pred_max                 272.092              272.092                                               
y_pred_mean                     85                   85                                               
training_date           2021-06-05           2021-06-05                                               
                                                                                                      
SUCCESS: The process with PID 17996 (child process of PID 15576) has been terminated.                 
SUCCESS: The process with PID 15576 (child process of PID 3156) has been terminated.                  
SUCCESS: The process with PID 3156 (child process of PID 13808) has been terminated. 
```




