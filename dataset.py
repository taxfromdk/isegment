import os
import time
import glob
import random
import pickle

class Dataset:
    def __init__(s, folder):
        #init memory
        s.memory = {}
        
        #Make sure storage folder exist
        s.folder = folder
        os.makedirs(s.folder, exist_ok=True)

        s.last_check = None
        s.checkForUpdates()

        s.active_pixels = 0

        s.randlist = []

    def getModTime(s, fn):
        return os.stat(fn)[8]

    def load(s, fn):
        #print('loading', fn)
        s.memory[fn] = {}
        s.memory[fn]['datapoint'] = pickle.load( open( fn, "rb" ) )
        s.memory[fn]['modtime'] = s.getModTime(fn)
        s.active_pixels = s.getPixels()
        
    def check(s, fn):
        changed = False
        #verify it exist
        if not os.path.exists(fn):
            print('removing', fn)
            del s.memory[fn]
            changed = True
        else:
            modtime = s.getModTime(fn)
            diff = modtime - s.memory[fn]['modtime']
            if diff != 0:
                print('updating', fn)
                s.load(fn)
                changed = True
        return changed

    def checkForUpdates(s):        
        changed = False
        n = time.time()
        if s.last_check == None or n > s.last_check + 10.0:
            print("Checking")
            s.last_check = n
            keys = s.getKeys()
            #check already known samples for updates
            for fn in keys:
                changed = s.check(fn) or changed
            
            #Load datapoints already in folder
            for fn in glob.iglob(s.folder + '/*.datapoint', recursive=False):
                if fn not in keys:
                    changed = True
                    s.load(fn)

        return changed
            
    def getFN(s, i):
        return '%s/%08d.datapoint'%(s.folder, i)

    def getAvailableFN(s):
        i = 0
        while os.path.exists(s.getFN(i)):
            i += 1
        return s.getFN(i)

    def put(s, datapoint, fn=None):
        if fn == None:
            fn = s.getAvailableFN()
        pickle.dump( datapoint, open( fn + '_tmp', "wb" ) )
        if os.path.exists(fn):
            os.remove(fn)
        os.rename(fn + '_tmp', fn)
        s.load(fn)
        s.active_pixels = s.getPixels()
        return fn

    def get(s, fn):
        if fn in s.memory:
            s.check(fn)
            return s.memory[fn]['datapoint']
        else:
            return None
    
    def delete(s, fn):
        os.remove(fn)
        s.check(fn)
        s.active_pixels = s.getPixels()

    def getKeys(s):
        return sorted(list(s.memory.keys()))
    
    def getPixels(s):
        acc = 0
        for k in list(s.memory.keys()):
            acc += s.memory[k]['datapoint']["active_pixels"]
        return acc

    def getRandom(s):
        keys = s.getKeys()
        if len(keys):
            if len(s.randlist) == 0:
                s.randlist = keys
                random.shuffle(s.randlist)
            fn = s.randlist[0]
            s.randlist = s.randlist[1:]
            return s.get(fn), fn
        return None