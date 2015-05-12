import numpy as np

from sklearn import tree
from sklearn.externals import joblib
from deap import benchmarks
from sklearn import preprocessing
import math
from sklearn import ensemble
import copy
from sklearn.preprocessing import Imputer
from sklearn.gaussian_process import GaussianProcess

from sklearn import cross_validation
from sklearn import datasets

from sklearn.datasets import fetch_mldata
import gc
import time
import matplotlib.pyplot as pl
import os

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import sys

from sklearn.feature_selection import f_classif, f_regression

import urllib
from sklearn import svm

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

from scipy import stats



#test function
def rastrigin_arg_vari(sol):
	#D1,D2,D3,D4,D5,D6,D7,D8,D9,D10 = sol[0], sol[1], sol[2], sol[3], sol[4], sol[5], sol[6], sol[7], sol[8], sol[9]
	Z = np.zeros(sol.shape[0])
	#print sol.shape[0]
	for i in xrange(sol.shape[0]):
		#print sol[i]
		Z[i] = benchmarks.rastrigin(sol[i])[0]
		#print Z[i]
	return Z

def createTraingSet(datasetname,seed):
	if len(datasetname.data) > 40000:
		datasetname.data = datasetname.data[:40000,:]
		datasetname.target = datasetname.target[:40000]

	std_scaler = StandardScaler()
	datasetname.data = std_scaler.fit_transform(datasetname.data,y=datasetname.target)

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(datasetname.data, datasetname.target, test_size=0.2, random_state=seed)

	return X_train, X_test, y_train, y_test


def createClf(training_x, training_y, test_x, test_y, printscore=False, regression=True,model_prefered="RandomForest"):
	#print "Initializing process"
	if (regression):
		#print "regression"
		if (model_prefered == "SVM"):
			clf = svm.SVR()
		elif (model_prefered=="Gaussian"):
			clf = GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
		elif (model_prefered=="Gradient"):
			clf = GradientBoostingRegressor(n_estimators=100)
		else:
			clf = ensemble.RandomForestRegressor(n_estimators=100)
	else:
		#print "classifier"
		if (model_prefered == "SVM"):
			clf = svm.SVC()
		elif (model_prefered=="Gaussian"):
			exit() #cannot use gaussian for classification
			clf = GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
		elif (model_prefered=="Gradient"):
			clf = GradientBoostingClassifier(n_estimators=100)
		else:
			clf = ensemble.RandomForestClassifier(n_estimators=100)
	clf.fit(training_x, training_y)
	#print "Done training"
	

	score = clf.score(test_x, test_y)
	if (printscore):
		print "Score:", score
	return clf, score


def clfPredict(clf,x):
	ant = clf.predict(x)
	return ant[0],1


from sklearn.neighbors import NearestNeighbors
def impute_NN2(trainingset, imputedmeanset):
	x = copy.deepcopy(trainingset)
	
	nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(imputedmeanset)
	#print x
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

				
	#print "imputed", x, imputed
	return x


def impute_MODELS(trainingset, targetset, imputedmeanset, start_column_with_missing_data,end_column_with_missing_data):
	x = copy.deepcopy(trainingset)
	#print x.shape, targetset.shape

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
				#model_input = np.hstack((imputedmeanset[i,:j],imputedmeanset[i,(j+1):]))
				x[i][j] = modelarray[j-start_column_with_missing_data].predict(modelinputs[j-start_column_with_missing_data][i])
	return x[:,:-1]




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
	#print "Performing run with dataset " + name, datasetname.data.shape
	start_row_with_missing_data = 0 #len(datasetname)/4 				#25% of the data is always complete
	start_column_with_missing_data = len(datasetname.data[0]) / 4 	#25% of the features is never missing
	end_column_with_missing_data = len(datasetname.data[0])
	#percentage_missing_per_column = [.1,.2,.5]

	#screw the training data by creating missing data.
	median_of_training_features = np.median(training_x,axis=0)
	missing=0
	j_size = len(training_x)
	max_missing = j_size * missing_perc / 10
	temp_j = np.arange(j_size)


	for j in range(start_column_with_missing_data,end_column_with_missing_data): 
		#print j_size
		np.random.shuffle(temp_j)
		missing_j = 0;
		refvar =  median_of_training_features[j] #this is the non random part for MAR == False
		for i in temp_j:
			if ( missing_j < max_missing and (MAR or  training_x[i][j] > refvar ) ): 
				missing += 1
				missing_j += 1
				training_x[i][j] = np.nang
	N_missing = missing

	missing = np.zeros(start_column_with_missing_data)
	for j in range(start_column_with_missing_data,end_column_with_missing_data): 	# For all features that something is missing, count the number of misses
		missing = np.append(missing,0)
		#print missing
		for i in range(len(training_x)):
			if (math.isnan(training_x[i][j])):
				missing[j]+=1
	if verbose:
		print "Missing data in each column: ",missing

	if verbose:
		print("--- Training set created in %s seconds ---" % (time.time() - start_time))
	start_time = time.time()
	timescores = []

	imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
	imp.fit(training_x)
	imputed_x_1 = imp.transform(training_x)

	if verbose:
		print("--- Imputation with mean in %s seconds ---" % (time.time() - start_time))
	timescores.append((time.time() - start_time))
	start_time = time.time()

	imp = Imputer(missing_values='NaN', strategy='median', axis=0)
	imp.fit(training_x)
	imputed_x_2 = imp.transform(training_x)
	
	if verbose:
		print("--- Imputation with median in %s seconds ---" % (time.time() - start_time))
	timescores.append((time.time() - start_time))
	start_time = time.time()

	imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
	imp.fit(training_x)
	imputed_x_3 = imp.transform(training_x)
	
	if verbose:
		print("--- Imputation with most frequent in %s seconds ---" % (time.time() - start_time))
	timescores.append((time.time() - start_time))
	start_time = time.time()
	imputed_x_1_temp = copy.deepcopy(imputed_x_1)
	imputed_x_4 = impute_NN2(training_x,imputed_x_1_temp)

	if verbose:
		print("--- Imputation with NN2 in %s seconds ---" % (time.time() - start_time))
	timescores.append((time.time() - start_time))
	start_time = time.time()

	imputed_x_1_temp = copy.deepcopy(imputed_x_1)
	imputed_x_5 = impute_MODELS(training_x, training_y, imputed_x_1_temp, start_column_with_missing_data,end_column_with_missing_data)

	timescores.append((time.time() - start_time))
	start_time = time.time()

	# find the Univariate score of each feature using random forest feature importance
	imputed_x_1_temp = copy.deepcopy(imputed_x_1)

	extraforest,score = createClf(imputed_x_1_temp[:,start_column_with_missing_data:end_column_with_missing_data], training_y,imputed_x_1_temp[:,start_column_with_missing_data:end_column_with_missing_data],training_y,regression=regression)
	F = extraforest.feature_importances_

	for i in range(len(F)):
		if np.isnan(F[i]):
			F[i] = 0
	column_numbers = np.arange(start_column_with_missing_data,end_column_with_missing_data)
	sorted_column_numbers = sorted(column_numbers, key=lambda best: -F[best-start_column_with_missing_data]) 

	training_x_first = copy.deepcopy(training_x[:,0:start_column_with_missing_data])
	training_x_first = np.append(training_x_first, np.array([training_y]).T, axis=1)
	
	for missing in sorted_column_numbers: 
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
		training_y_first = training_y_first[mask]#np.delete(training_y_first,(to_calc_i), axis=0)

		clf, score = createClf(training_x_first_mask, training_y_first,training_x_first_mask,training_y_first)

		imputed = 0
		for i in to_calc_i:
			training_x[i,missing] = clf.predict(to_calc_x[imputed])
			imputed += 1

		training_x_first = np.append(training_x_first, np.array([training_x[:,missing]]).T, axis=1) 

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
		print "imputation with mean:",
	clf2, score = createClf(imputed_x_1,Y,test_x,test_y,verbose, regression=regression,model_prefered=model_prefered)
	outputscores.append(score)
	mean_squared_errors.append(sqrt(mean_squared_error(imputed_x_1, X)))

	if verbose:
		print "imputation with median:",
	clf2, score = createClf(imputed_x_2,Y,test_x,test_y,verbose, regression=regression,model_prefered=model_prefered)
	outputscores.append(score)
	mean_squared_errors.append(sqrt(mean_squared_error(imputed_x_2, X)))

	if verbose:
		print "imputation with most frequent:",
	clf2, score = createClf(imputed_x_3,Y,test_x,test_y,verbose, regression=regression,model_prefered=model_prefered)
	outputscores.append(score)
	mean_squared_errors.append(sqrt(mean_squared_error(imputed_x_3, X)))

	if verbose:
		print "imputation with PVI (predictive value imputation) using NN2:",
	clf2, score = createClf(imputed_x_4,Y,test_x,test_y,verbose, regression=regression,model_prefered=model_prefered)
	outputscores.append(score)
	mean_squared_errors.append(sqrt(mean_squared_error(imputed_x_4, X)))


	if verbose:
		print "imputation with MODELS:",
	clf2, score = createClf(imputed_x_5, Y,test_x,test_y,verbose, regression=regression,model_prefered=model_prefered)
	outputscores.append(score)
	mean_squared_errors.append(sqrt(mean_squared_error(imputed_x_5, X)))

	if verbose:
		print "imputation with IARI:",
	clf2, score = createClf(training_x, training_y,test_x,test_y,verbose, regression=regression,model_prefered=model_prefered)
	outputscores.append(score)
	mean_squared_errors.append(sqrt(mean_squared_error(training_x, X)))

	

	if verbose:
		print("--- Run done in %s seconds ---" % (time.time() - start_all))
	return outputscores, timescores, mean_squared_errors, N_missing


indn = 6
print "Results for the imputation algorithms. Per test run the score of the models generated using the imputed training sets is given in the order"
print "Reference model (no missing data), Imputation with mean, Imputation with median, Imputation with most freq., PVI using NN2, Imputation with RFMs"
print ""
regression = True

if (len(sys.argv) != 2  and len(sys.argv) != 3 and len(sys.argv) != 4):
	print "Usage: python ",sys.argv[0], "dataset_name","[MAR:true/false]","[percentage]"
	exit()

dataname = sys.argv[1]

if (len(sys.argv) > 2):
	MAR_T = sys.argv[2]
else:
	MAR_T = "2"
missing_perc_in = [1,2,3,4,5,6] #0.1,0.5,
MAR_IN = [True, False]

if (len(sys.argv) == 4):
	missing_perc_in = [float(sys.argv[3])]

if (MAR_T=="1"):
	MAR_IN = [True]
elif (MAR_T=="0"):
	MAR_IN = [False]

model_prefered = "Gradient" #controlls if we use SVM, Gaussian processes, Gradient boosting or Random Forest as final model

if (MAR_T=="Gradient"):
	model_prefered = "Gradient"
elif (MAR_T=="RandomForest"):
	model_prefered = "RandomForest"
elif (MAR_T=="SVM"):
	model_prefered = "SVM"
elif (MAR_T=="Gaussian"):
	model_prefered = "Gaussian"

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
	# https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State
	d.data = np.loadtxt("page-blocks/page-blocks.data", delimiter=",")
	d.target = d.data[:,(len(d.data[0])-1)]
	d.data = d.data[:,:(len(d.data[0])-1)]
	regression = False
elif dataname == "concrete":
	d = lambda:0 
	# https://archive.ics.uci.edu/ml/datasets/SkillCraft1+Master+Table+Dataset
	d.data = np.loadtxt("Concrete/Concrete_Data.csv", delimiter=",")
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

seed_times = 10
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
		stdscores = np.std(allscores,axis=0)
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


		allscores = np.array(allscores)

		#determine the significance of the scores
		#calculate the t-test on our algorithm and the rest
		best_algorithm = 0
		for test in range(1,7):
			T_score, prob = stats.ttest_ind(allscores[:,test],allscores)
			significant_better = True;
			for i in range(1,len(T_score)):
				if T_score[i]<stdscores[test]*3 and i!=test:
					significant_better = False
			if (significant_better):
				best_algorithm = test

		allbestalgs.append( best_algorithm )


		if os.path.isdir('/home/promimoocbas/arrays/') == False:
			os.makedirs('/home/promimoocbas/arrays/')

		if os.path.isdir('/home/promimoocbas/img/') == False:
			os.makedirs('/home/promimoocbas/img/')
			
		
		#print allscores.shape
		pl.figure(figsize=(10,6))
		pl.boxplot(allscores)
		pl.ylabel('Performance of each model')
		pl.title('Performance of different imputation methods')
		pl.xticks(np.arange(7)+1, ( 'Reference','Mean', 'Median', 'Most Frequent', 'PVI', 'RI', 'IARI') )
		if (MAR):
			pl.savefig( '/home/promimoocbas/img/'+final_model+'_avg_result_'+dataname+'_missing_'+`missing_perc_2`+'_percent_MAR.png')
		else:
			pl.savefig( '/home/promimoocbas/img/'+final_model+'_avg_result_'+dataname+'_missing_'+`missing_perc_2`+'_percent_MNAR.png')
		pl.clf()
		
		np.save( '/home/promimoocbas/img/Z'+final_model+'_allscores_'+dataname+'_missing_'+`missing_perc_2`+'_percent_MNAR.npy', allscores)

		pl.figure(figsize=(10,6))
		pl.bar(ind, avgmeansquarederrors,   0.4, color='r')
		pl.ylabel('Root Mean Squared Error')
		pl.title('RMSE per imputation method')
		pl.xticks(ind+0.4/2., ( 'Mean', 'Median', 'Most Frequent', 'PVI', 'RI', 'IARI') )
		if (MAR):
			pl.savefig('/home/promimoocbas/img/'+final_model+'_avg_rmse_'+dataname+'_missing_'+`missing_perc_2`+'_percent_MAR.png')
		else:
			pl.savefig('/home/promimoocbas/img/'+final_model+'_avg_rmse_'+dataname+'_missing_'+`missing_perc_2`+'_percent_MNAR.png')
		pl.clf()


print "Scores"
for i in all_all_scores:
	print ' & '.join(map(str, i)),"\\\\"

print "RMSE"
for i in all_all_rmse:
	print ' & '.join(map(str, i)),"\\\\"
print "Times"
for i in all_all_times:
	print ' & '.join(map(str, i)),"\\\\"

print "Significant best algorithm per test:"
print allbestalgs

exit()



