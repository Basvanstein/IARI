import sys
import pyximport; pyximport.install()
import testrepair

if (len(sys.argv) != 2  and len(sys.argv) != 3 and len(sys.argv) != 4):
	print "Usage: python ",sys.argv[0], "dataset_name","preffered model"
	exit()

model_prefered = sys.argv[2]
data_name = sys.argv[1]

testrepair.executeRun(data_name,model_prefered)