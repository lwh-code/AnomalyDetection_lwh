import shutil
import matplotlib

if __name__ == '__main__':
    #解决中文问题是对matplotlib库之前的信息缓存进行清除
    shutil.rmtree(matplotlib.get_cachedir())