


### source init.sh

####Only show current directory
 PS1="[\W]\\$ "


pwd
which python 


alias aa='PS1="[\W]\\$ "'

alias ipy="ipython --no-autoindent"



################################################################################################
function git_autocommit {
   git  add -A
   git  commit -m "auto commit"
}




function cron_autocommit {

  while true; do git_autocommit; sleep 3600; done
}



################################################################################################
#### Auto Batch



#The $0 variable is reserved for the function’s name.
#The $# variable holds the number of positional parameters/arguments passed to the function.