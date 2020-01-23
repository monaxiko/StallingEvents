import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pcapng import FileScanner

filename = '2.pcapng'
packet_blocks = []
file = open(filename,'rb')
pcapfile = savefile.load_savefile(file,verbose=True)