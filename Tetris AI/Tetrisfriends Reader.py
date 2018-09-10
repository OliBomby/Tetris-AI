from tkinter import *
from PIL import Image, ImageGrab, ImageTk
from scipy.misc.pilutil import imresize
import numpy as np
import tensorflow as tf
import time

def run():
    global label
    global img4
    global img2
    global img3
    global img
    global root
    global n 
    root=Tk()

    nw = [561,636]
    areasize = [180,360]
    
    esg = [54,72]
    poi = [[[-42,66],[0,0]],[[238,66],[1,0]],[[238,126],[2,0]],
           [[238,171],[3,0]],[[238,216],[4,0]],[[238,261],[5,0]]]
    
    grabarea = [nw[0]-esg[0],nw[1],nw[0]+areasize[0]+esg[1],nw[1]+areasize[1]]
    grabsize = [areasize[0]+esg[0]+esg[1],areasize[1]]

    img = ImageGrab.grab(bbox=grabarea)
    img2 = imresize(img, (1,1))
    img3 = Image.fromarray(img2)
    img4 = ImageTk.PhotoImage(img)

    n = 1/18
    label = Label(root)
    label.pack()

    def update():
        global label
        global img4
        global img2
        global img3
        global img
        global root
        global n 
        img = ImageGrab.grab(bbox=grabarea)
        array = np.array(img)
        img2 = imresize(array, n, 'nearest')
        for p in poi:
            stuff = array[p[0][1]][p[0][0]+esg[0]]
            img2[p[1][0]][p[1][1]] = stuff
            
        img5 = imresize(img2, tuple(grabsize[::-1]), 'nearest')
        img3 = Image.fromarray(img5)
        img4 = ImageTk.PhotoImage(img3)
        label.configure(image=img4)
        root.after(1, update)
        
    root.after(0, update)
    root.mainloop()


run()

