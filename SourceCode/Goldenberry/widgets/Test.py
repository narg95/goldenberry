from Search.OWcga import testWidget
import sys
def main():
    try:
        testWidget()
    except:
        print "Unexpected error:", sys.exc_info()[0]
        raise   
    
main()