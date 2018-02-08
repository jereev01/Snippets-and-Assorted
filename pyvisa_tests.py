# -*- coding: utf-8 -*-
"""
Setting up a Script File to interact with various USB DMM 

Takes a reading every time the command cycle is completed.
Each command takes about 0.350 seconds preventing this from being super useful.
Faster options would require data logging or some other device.
There must be a way using VISAs to quickly measure voltages.
"""

import visa
import time


rm =  visa.ResourceManager()

reslist = rm.list_resources()

def getnames(resource_list,kind):
    """
    Given list of available resources return names of devices
    that match a specific kind e.g. 'USB' , 'GPIB' etc.
    """
    names = []
    for res in reslist:
        if 'USB' in res:
            inst = rm.open_resource(res)
            names.append(inst.query('*IDN?'))
    return(names)

inst_names = getnames(reslist,'USB')

print(inst_names)

my_instrument = rm.open_resource(reslist[0])

t0 = time.time()
print(time.time()-t0)


t = []
v = []

for x in range(10):
    tm = time.time()-t0
    vm = my_instrument.query_ascii_values('MEAS:VOLT:DC? 100mV')[0] 
    t.append(tm)
    v.append(vm)
    
print(t)
print(v)


