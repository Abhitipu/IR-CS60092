# Python program to illustrate the concept
# of threading
# importing the threading module
import threading
import time

lock = threading.Lock()
def print_cube(num, name, mapofall):
    # function to print cube of given num
    time.sleep(2)
    lock.acquire()
    print(f"Cube: {num} {name}")
    mapofall[name] = num
    lock.release()
    

def print_square(num, mapofall):
    # function to print square of given num
    time.sleep(2)
    lock.acquire()
    mapofall[num] = num*num
    print("Square: {}" .format(num * num))
    lock.release()


if __name__ =="__main__":
    # creating thread
    mapofall = {}
    t1 = threading.Thread(target=print_square, args=(10,mapofall))
    t2 = threading.Thread(target=print_cube, args=(10,"longondon", mapofall))

    # starting thread 1
    t1.start()
    # starting thread 2
    t2.start()

    # wait until thread 1 is completely executed
    t1.join()
    # wait until thread 2 is completely executed
    t2.join()
    # print(apple)
    print(mapofall)
    # both threads completely executed
    print("Done!")
