from params import *


def log(msg,file_name):
    print(msg)
    file_log = open(LOG_PATH+file_name+'.log',"a") 
    file_log.writelines(msg+ '\n')
    file_log.close()