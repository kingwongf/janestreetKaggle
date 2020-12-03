# Pipeline and machine learning algorithms
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier


# Model fine-tuning and evaluation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn import model_selection


def multi_model_scores(X_train, y_train, cv):
	'''
	params: cv: An iterable yielding (train, test) splits as arrays of indices.
	'''
	# Initiate 11 classifier models
	ran = RandomForestClassifier(random_state=1)
	knn = KNeighborsClassifier()
	log = LogisticRegression()
	xgb = XGBClassifier()
	gbc = GradientBoostingClassifier()
	svc = SVC(probability=True)
	ext = ExtraTreesClassifier()
	ada = AdaBoostClassifier()
	gnb = GaussianNB()
	gpc = GaussianProcessClassifier()
	bag = BaggingClassifier()

	# Prepare lists
	models = [ran, knn, log, xgb, gbc, svc, ext, ada, gnb, gpc, bag]         
	scores = []

	# Sequentially fit and cross validate all models
	for mod in models:
	    mod.fit(X_train, y_train)
	    acc = cross_val_score(mod, X_train, y_train, scoring = "accuracy", cv = cv)
	    scores.append(acc.mean())
	# Creating a table of results, ranked highest to lowest
	results = pd.DataFrame({
	    'Model': ['Random Forest', 'K Nearest Neighbour', 'Logistic Regression', 
	    'XGBoost', 'Gradient Boosting', 'SVC', 'Extra Trees', 
	    'AdaBoost', 'Gaussian Naive Bayes', 'Gaussian Process', 'Bagging Classifier'],
	    'Score': scores})

	result_df = results.sort_values(by='Score', ascending=False).reset_index(drop=True)
	return res