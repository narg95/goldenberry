from fitfunctions.fitnessFuncs import onemax, zero
from searchers.cga import *
import sys

def main():
    c = cga()
    c.config(zero(),10, 15)
    result = c.find()
    print(result)
     

main()