import numpy as np




import math
from sklearn import ensemble
import copy
from sklearn.preprocessing import Imputer
from sklearn.gaussian_process import GaussianProcess

from sklearn import cross_validation
from sklearn import datasets

from sklearn.datasets import fetch_mldata
import time
import os

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import sys
from scipy import stats




from sklearn import svm

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor



###
# Creates a training set with max 40k points
###
def createTraingSet(datasetname,seed):
	if len(datasetname.data) > 40000:
		datasetname.data = datasetname.data[:40000,:]
		datasetname.target = datasetname.target[:40000]

	std_scaler = StandardScaler()
	datasetname.data = std_scaler.fit_transform(datasetname.data,y=datasetname.target)
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(datasetname.data, datasetname.target, test_size=0.2, random_state=seed)
	return X_train, X_test, y_train, y_test

###
# Creates a random forest with 100 estimators and no out-of-bag sampling
###
def createClffast(training_x, training_y):
	clf = ensemble.RandomForestRegressor(n_estimators=100,oob_score=True)
	clf.fit(training_x, training_y)
	return clf

###
# Creates a regression or classification model and trains it on the data given
###
def createClf(training_x, training_y, test_x, test_y, printscore=False, regression=True,model_prefered="RandomForest"):
	if (regression):
		#print "regressor"
		if (model_prefered == "SVM"):
			clf = svm.SVR()
		elif (model_prefered=="Gaussian"):
			clf = GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
		elif (model_prefered=="Gradient"):
			clf = GradientBoostingRegressor(n_estimators=50)
		else:
			clf = ensemble.RandomForestRegressor(n_estimators=50)
	else:
		#print "classifier"
		if (model_prefered == "SVM"):
			clf = svm.SVC()
		elif (model_prefered=="Gaussian"):
			exit() #cannot use gaussian for classification
			clf = GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
		elif (model_prefered=="Gradient"):
			clf = GradientBoostingClassifier(n_estimators=50)
		else:
			clf = ensemble.RandomForestClassifier(n_estimators=50)
		 
	clf.fit(training_x, training_y)
	score = clf.score(test_x, test_y)
	if (printscore):
		print "Score:", score
	return clf, score


#returns prediction of a model
def clfPredict(clf,x):
	ant = clf.predict(x)
	return ant[0],1


###
# Inpute a data set using 2k nearest neighbour PVI
###
from sklearn.neighbors import NearestNeighbors
def impute_NN2(trainingset, imputedmeanset):
	x = copy.deepcopy(trainingset)
	
	nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(imputedmeanset)
	imputed = 0
	
	for i in range(len(x)):
		for j in range(len(x[i])):
			if (math.isnan(x[i][j])):
				distances, indices = nbrs.kneighbors(imputedmeanset[i])
				#count the number of missing values in the neighbours
				n1 = x[indices[0][1]]
				n2 = x[indices[0][2]]
				n1_m = float(len(n1))
				n2_m = float(len(n2))
				for n1_1 in n1:
					if math.isnan(n1_1):
						n1_m -= 1.0 
				for n2_1 in n2:
					if math.isnan(n2_1):
						n2_m -= 1.0 
				to = float(n1_m+n2_m)

				imputed += 1
				x[i][j] = imputedmeanset[indices[0][1]][j] * (n1_m / to) +  imputedmeanset[indices[0][2]][j] * (n2_m / to)
	return x

###
# Inpute a data set using regression imputation
###
def impute_MODELS(trainingset, targetset, imputedmeanset, start_column_with_missing_data,end_column_with_missing_data):
	x = copy.deepcopy(trainingset)
	x = np.hstack((x,np.array([targetset]).T)) #add the target to the features
	imputedmeanset = np.hstack((imputedmeanset,np.array([targetset]).T))

	modelarray = []
	modelinputs = []
	for j in range(start_column_with_missing_data,end_column_with_missing_data):
		model_training_set = np.hstack((imputedmeanset[:,:j],imputedmeanset[:,(j+1):]))
		modelinputs.append(model_training_set)
		model_training_target = imputedmeanset[:,j]
		clf,score = createClf(model_training_set, model_training_target,model_training_set,model_training_target)
		modelarray.append(clf)

	#imputation
	for i in range(len(x)):
		for j in range(start_column_with_missing_data,end_column_with_missing_data):
			if (math.isnan(x[i][j])):
				x[i][j] = modelarray[j-start_column_with_missing_data].predict(modelinputs[j-start_column_with_missing_data][i])
	return x[:,:-1]



#######
# MAIN ALGORITHM
#######
def runtest(datasetname, name, seed, missing_perc=4.,MAR = True, verbose = False, regression=True,model_prefered="RandomForest"):
	np.random.seed(seed)
	missing_perc = float(missing_perc)
	start_time = time.time()
	start_all = start_time
	sys.stderr.write("--- Started test "+name+" "+`seed`+" ---\n" )
	

	#Load the training and test set
	training_x, test_x, training_y, test_y = createTraingSet(datasetname,seed)

	#shuffle the x columns 
	randomorder = np.random.permutation(len(training_x[0]))
	training_x = training_x[:,randomorder]
	test_x = test_x[:,randomorder]

	#(samples, features)
	X = copy.deepcopy(training_x)
	Y = copy.deepcopy(training_y)

	#test settings
	start_row_with_missing_data = 0 #len(datasetname)/4 				#25% of the data is always complete
	start_column_with_missing_data = len(datasetname.data[0]) / 4 	#25% of the features is never missing
	end_column_with_missing_data = len(datasetname.data[0])

	#screw the training data by creating missing data.
	median_of_training_features = np.median(training_x,axis=0)
	missing=0
	j_size = len(training_x)
	max_missing = j_size * missing_perc / 10
	temp_j = np.arange(j_size)


	for j in range(start_column_with_missing_data,end_column_with_missing_data): 
		np.random.shuffle(temp_j)
		missing_j = 0;
		refvar =  median_of_training_features[j] #this is the non random part for MAR == False
		for i in temp_j:
			if ( missing_j < max_missing and (MAR or  training_x[i][j] > refvar ) ): 
				missing += 1
				missing_j += 1
				training_x[i][j] = np.nan
	N_missing = missing

	#Fix the missing data with our algorithm.
	missing = np.zeros(start_column_with_missing_data)
	for j in range(start_column_with_missing_data,end_column_with_missing_data): 	# For all features that something is missing, count the number of misses
		missing = np.append(missing,0)
		for i in range(len(training_x)):
			if (math.isnan(training_x[i][j])):
				missing[j]+=1
	if verbose:
		print "Missing data in each column: ",missing

	if verbose:
		print("--- Training set created in %s seconds ---" % (time.time() - start_time))
	start_time = time.time()
	timescores = []

	column_numbers = np.arange(start_column_with_missing_data,end_column_with_missing_data)
	training_x_first = copy.deepcopy(training_x[:,0:start_column_with_missing_data])
	training_x_first = np.append(training_x_first, np.array([training_y]).T, axis=1)
	
	usedcolumns = list(xrange(start_column_with_missing_data))
	while len(column_numbers)!=0:
		bestscore = -float('inf')
		bestcolumn = -1
		bestclf = None
		calcx = None
		calci = None
		for missing in column_numbers: # range(start_column_with_missing_data,end_column_with_missing_data):
			#try to add the column
			training_y_first = copy.deepcopy(training_x[:,missing])
			
			to_calc_i = []
			to_calc_x = []

			i = len(training_y_first)
			
			while i > 0:
				i -= 1
				if (math.isnan(training_y_first[i]) ):
					to_calc_x.append(training_x_first[i])
					to_calc_i.append(i)

			mask = np.ones(len(training_x_first), dtype=bool)
			mask[to_calc_i] = False
			training_x_first_mask=training_x_first[mask]
			
			mask = np.ones(len(training_y_first), dtype=bool)
			mask[to_calc_i] = False
			training_y_first_mask = training_y_first[mask]

			clf = createClffast(training_x_first_mask, training_y_first_mask)
			newscore = clf.oob_score_
			if newscore > bestscore:
				bestscore = newscore
				bestcolumn = missing
				bestclf = clf
				calcx = to_calc_x
				calci = to_calc_i
			if newscore == 1 :
				break
			
		#impute the real set
		imputed = 0
		for i in calci:
			training_x[i,bestcolumn] = bestclf.predict(calcx[imputed])
			imputed += 1

		cpos = np.where(column_numbers==bestcolumn)[0]
		column_numbers = np.delete(column_numbers,cpos)
		usedcolumns.append(bestcolumn)
		training_x_first = np.append(training_x_first, np.array([training_x[:,bestcolumn]]).T, axis=1) 

	#print "--- used ", len(usedcolumns), " features ---"
	if verbose:
		print("--- Imputation with Reduced Feature Models in %s seconds ---" % (time.time() - start_time))
	timescores.append((time.time() - start_time))
	start_time = time.time()
			
	outputscores = []
	mean_squared_errors = []
	if verbose:
		print "Complete model:",


	clf2, score = createClf(X,Y,test_x,test_y,verbose, regression=regression,model_prefered=model_prefered)
	outputscores.append(score)

	
	if verbose:
		print "imputation with GIARI:",
	clf2, score = createClf(training_x, training_y,test_x,test_y,verbose, regression=regression,model_prefered=model_prefered)
	outputscores.append(score)
	mean_squared_errors.append(sqrt(mean_squared_error(training_x, X)))


	if verbose:
		print("--- Run done in %s seconds ---" % (time.time() - start_all))
	return outputscores, timescores, mean_squared_errors, N_missing





######## START OF THE PROGRAM #######
def executeRun(dataname="Digits", model_prefered= "RandomForest"):
	#uci-20070111 kdd_internet_usage

	indn = 0
	print "Results for the imputation algorithms. Per test run the score of the models generated using the imputed training sets is given in the order"
	print "Reference model (no missing data), Imputation with mean, Imputation with median, Imputation with most freq., PVI using NN2, Imputation with RFMs"
	print ""

	regression = True

	MAR_T = "2"
	missing_perc_in = [1,2,3,4,5,6] #,7,8] #0.1,0.5,
	MAR_IN = [True, False]

	if (MAR_T=="1"):
		MAR_IN = [True]
	elif (MAR_T=="0"):
		MAR_IN = [False]

	model_prefered = "Gradient" #controlls if we use SVM, Gaussian processes, Gradient boosting or Random Forest as final model

	print "MAR_IN=",MAR_IN

	d = None
	if dataname == "allhouses":
		d = fetch_mldata('uci-20070111 house_16H')
		d.target = d.data[:,(len(d.data[0])-1)]
		d.data = d.data[:,:(len(d.data[0])-1)]
		regression = True

	
	elif dataname == "eye":
		d = lambda:0 
		# https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State
		d.data = np.loadtxt("Eye/EEG Eye State.csv", delimiter=",")
		d.target = d.data[:,(len(d.data[0])-1)]
		d.data = d.data[:,:(len(d.data[0])-1)]
		regression = False

	elif dataname == "page":
		d = lambda:0 
		# https://archive.ics.uci.edu/ml/datasets/Page+Blocks+Classification
		d.data = np.loadtxt("../page-blocks/page-blocks.data", delimiter=",")
		d.target = d.data[:,(len(d.data[0])-1)]
		d.data = d.data[:,:(len(d.data[0])-1)]
		regression = False
	elif dataname == "skill": #fails to load
		d = lambda:0 
		# https://archive.ics.uci.edu/ml/datasets/SkillCraft1+Master+Table+Dataset
		d.data = np.loadtxt("SkillCraft/SkillCraft1_Dataset.csv", delimiter=",")
		d.target = d.data[:,(len(d.data[0])-1)]
		d.data = d.data[:,:(len(d.data[0])-1)]
		regression = True
		exit()

	elif dataname == "concrete":
		d = lambda:0 
		# https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
		d.data = np.loadtxt("../Concrete/Concrete_Data.csv", delimiter=",")
		
		d.target = d.data[:,(len(d.data[0])-1)]
		d.data = d.data[:,:(len(d.data[0])-1)]
		regression = True
	elif dataname == "digits":
		d = datasets.load_digits()
		regression = False
	elif dataname == "iris":#too small
		d = datasets.load_iris()
		regression = False
		
	elif dataname == "cover":
		d = datasets.fetch_covtype()
		regression = False
	else:
		print "No such dataset";
		exit()

	print d.data.shape
	print d.target.shape

	all_all_scores = []
	all_all_rmse = []
	all_all_times = []
	allbestalgs = []

	final_model = model_prefered

	seed_times = 20
	for MAR in MAR_IN:
		for missing_perc in missing_perc_in:
			allscores = []
			alltimes = []
			allerrors = []
			allmissing = []
			for seed in range(seed_times):
				time_run_start = time.time()
				scores, times, mean_squared_errors, N_missing = runtest(d,dataname + " dataset "+`missing_perc`,seed, missing_perc,MAR=MAR, regression=regression, model_prefered=model_prefered)
				time_run = time.time() - time_run_start
				sys.stderr.write("--- Test finished in "+`time_run`+" ---\n" )
				time_left = time_run * (seed_times-seed) +  time_run * (len(missing_perc_in) - missing_perc_in.index(missing_perc))*seed_times
				time_left_hour = int(time_left / 3600)
				time_left_minute = int((time_left - 3600*time_left_hour)/60)
				sys.stderr.write("--- Aproximate time left (HH:MM): "+`time_left_hour`+":"+`time_left_minute`+" ---\n" )
				allscores.append(scores)
				alltimes.append(times)
				allmissing.append(N_missing)
				allerrors.append(mean_squared_errors)

			avgscores = np.average(allscores,axis=0)
			avgtimes = np.average(alltimes,axis=0)
			avgmeansquarederrors = np.average(allerrors,axis=0)
			avgmissing = np.average(allmissing)
			all_all_scores.append((avgmissing,MAR, ' & '.join(map(str, avgscores)) ))
			all_all_rmse.append((avgmissing,MAR, ' & '.join(map(str, avgmeansquarederrors)) ))
			all_all_times.append((avgmissing,MAR, ' & '.join(map(str, avgtimes))))


			missing_perc_2 = missing_perc * 10.0
			width = 0.3
			ind = np.arange(indn)
			comp_scores = np.zeros(indn)
			for i in range(indn):
				comp_scores[i] = avgscores[0] - avgscores[i+1]
			

			if os.path.isdir('arraysG/') == False:
				os.makedirs('arraysG/')

			allscores = np.array(allscores)
			allerrors = np.array(allerrors)
			alltimes = np.array(alltimes)
			stdscores = np.std(allscores,axis=0)
			allscores = np.array(allscores)

			np.save( 'arraysG/allscores_'+final_model+'_'+dataname+'_missing_'+`missing_perc_2`+'_percent_MNAR.npy', allscores)
			np.save( 'arraysG/allmrse_'+final_model+'_'+dataname+'_missing_'+`missing_perc_2`+'_percent_MNAR.npy', allerrors)
			np.save( 'arraysG/alltimes_'+final_model+'_'+dataname+'_missing_'+`missing_perc_2`+'_percent_MNAR.npy', alltimes)
			
	print "Scores"
	for i in all_all_scores:
		print ' & '.join(map(str, i)),"\\\\"

	print "RMSE"
	for i in all_all_rmse:
		print ' & '.join(map(str, i)),"\\\\"
	print "Times"
	for i in all_all_times:
		print ' & '.join(map(str, i)),"\\\\"


	exit()



