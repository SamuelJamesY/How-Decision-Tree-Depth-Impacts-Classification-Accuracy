'''
Decision Tree Model for Pima Indian Dataset
'''

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier 
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

def load_data():
	'''
	Load the data and seperate into predictor and response variables
	'''
	df = pd.read_csv('pima.csv',header=None)
	df.columns = ['preg','gluc','bp','skin','insulin','BMI','dbf','age','diabetes']
	# note there aren't any categorical elements for dtree so just take as is
	features = df.iloc[:,:-1].to_numpy()
	targets = df.iloc[:,-1].to_numpy()
	return features,targets

def train_test_split_data(features,targets,expnum):
	'''
	Split the data into train and test dataset
	'''
	xtrain,xtest,ytrain,ytest = train_test_split(features,targets,test_size=0.25,random_state=expnum)
	return xtrain,xtest,ytrain,ytest

def decision_tree(xtrain,xtest,ytrain,ytest,expnum,depth):
	'''
	Build a decision tree model and print the rules of the decision tree
	'''
	dt = DecisionTreeClassifier(max_features='auto',min_samples_leaf=3,max_depth=depth,random_state=expnum)
	dtree = dt.fit(xtrain,ytrain)
	dt_rules = export_text(dtree,show_weights=True)
	#print(dt_rules)
	ypredtrain = dtree.predict(xtrain)
	ypredtest = dtree.predict(xtest)
	acc_train = accuracy_score(ytrain,ypredtrain)
	acc_test = accuracy_score(ytest,ypredtest)
	#print(acc_train,' Accuracy training Score',acc_test,' Accuracy test Score')
	return acc_train, acc_test

def main():
	'''
	Note the deeper the tree, the better performance on training data set but decrease in performance in test dataset
	Also min_sample_leaf=3 gives a good balance between overfitting and underfitting
	'''
	features,targets=load_data()
	depth_lst = list(range(1,11))
	acc_train_mean = []
	acc_test_mean = []
	for depth in depth_lst:
		acc_train_lst = np.empty(10)
		acc_test_lst = np.empty(10)
		for exp in range(10):
			xtrain,xtest,ytrain,ytest = train_test_split_data(features,targets,exp)
			acc_train_lst[exp],acc_test_lst[exp]=decision_tree(xtrain,xtest,ytrain,ytest,exp,depth)
		#print(acc_train_lst,' 10 Experinments training Accuracy')
		#print(acc_test_lst, ' 10 Experinments test Accuracy')
		acc_train_mean.append(acc_train_lst.mean())
		acc_test_mean.append(acc_test_lst.mean())
	# Shows a depth of six does best in test dataset
	print(acc_train_mean,'Mean of 10 Exps for each training of depths 1 to 10')
	print(acc_test_mean,'Mean of 10 Exp for each testing of depths 1 to 10')

if __name__ == '__main__':
	main()