
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Sequence, Union
import lzma
import pickle
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import sklearn.metrics

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import math
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize

from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import label_binarize
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier


class DataStorage:

    FEATURES_KEY = "segmentation_features"
    LABELS_KEY = "segmentation_labels_num"
    FEATURE_NAMES_KEY = "feature_names"

    def __init__(self, features : np.ndarray, labels : np.ndarray, feature_names : np.ndarray = None, label_names : Sequence[str] = None) -> None:
        """
        Initialises the instance with data arrays.

        Arguments:
        - 'features' - Data feature values.
        - 'labels' - Data labels.
        - 'feature_names' - The names of the features.
        - 'label_names' - The names of the class labels.
        """
        self.features = features
        self.data_count : int = self.features.shape[0]
        self.feature_count : int = self.features.shape[1]
        self.labels = labels
        self.feature_names = feature_names
        self.label_names = label_names

    def getSubset(self, indices : np.ndarray) -> DataStorage:
        """
        Creates a new instance of this class with a subset of the data at the given indices.

        Arguments:
        - 'indices' - Indices to the subset of the data.

        Returns:
        - New instance of the class with data selected according to 'indices'.
        """
        return DataStorage(self.features[indices], self.labels[indices], self.feature_names, self.label_names)

    @classmethod
    def fromFile(cls, data_path : Path) -> DataStorage:
        """
        Loads the data from the given file.

        Arguments:
        - 'data_path' - Path to the file with the segmentation data.

        Returns:
        - An instance of this class with the data from the given file.
        """
        # Load the matlab file.
        data = scipy.io.loadmat(data_path)
        # Extract the feature values.
        features : np.ndarray = data[DataStorage.FEATURES_KEY]
        # Numerical labels in 1D array starting at index 0. 
        labels : np.ndarray = (data[DataStorage.LABELS_KEY] - 1).ravel()
        # Extract the feature names - this is a nested array.
        feature_names : np.ndarray = data[DataStorage.FEATURE_NAMES_KEY]       
        # Names of the classes from 0 to 6.
        label_names : List[str] = ["BRICKFACE", "CEMENT", "FOLIAGE", "GRASS", "PATH", "SKY", "WINDOW"]
        return cls(features, labels, feature_names, label_names)

class DataAnalysis:

    def __init__(self, rng : np.random.RandomState, args : argparse.Namespace):
        """
        Initialises the base class for data analysis classes.

        Arguments:
        - 'rng' - Random number generator for reproducible behaviour.
        - 'args' - Script arguments (passed down from the evaluator during evaluation).
        """
        self._rng = rng
        self._args = args

    def save(self, path : Union[Path, str], model : object) -> None:
        """
        Saves the given object, possibly a model or a list of models using pickle.

        Arguments:
        - 'path' - Path with the file name where the given object will be saved.
        - 'model' - Object to save using pickle.
        """
        with lzma.open(path, "wb") as model_file:
            pickle.dump(model, model_file)

    def load(self, path : Union[Path, str]) -> object:
        """
        Loads the pickled object from the given file.

        Arguments:
        - 'path' - Path with the file name to load.

        Returns:
        - 'The unpickled object.'
        """
        with lzma.open(path, "rb") as model_file:
            model = pickle.load(model_file)
        return model

class FeatureAnalysis(DataAnalysis):

    def __init__(self, rng : np.random.RandomState, args : argparse.Namespace):
        """
        Initialises the class for feature analysis and transformation.

        Arguments:
        - 'rng' - Random number generator for reproducible behaviour.
        - 'args' - Script arguments (passed down from the evaluator during evaluation).
        """
        super().__init__(rng, args)

        self.pca = PCA(n_components=4)

    def fit(self, data : DataStorage):
        """Implementation of feature selection/transformation fitting."""

        self.pca.fit(data.features)
    
    def transform(self, data : DataStorage):
        """Application of feature/transformation."""

        self.fit(data)
        data.features = self.pca.transform(data.features)

    def analyze(self, train_data_path: Path):
        """
        Performs feature analysis and visualizes PCA explained variance ratio.

        Arguments:
        - 'train_data_path' - Path to the training data file for analysis.
        """
        # TODO: You should implement your feature analysis and visualisation here.
        # - Do not forget to apply your final feature selection/transformation on the data before both training
        #   and testing. This method is only for finding an appropriate feature selection/transformation and
        #   its visualisation.

        data: DataStorage = DataStorage.fromFile(train_data_path)
        self.pca = PCA(n_components=None)
        self.fit(data)

        var = np.cumsum(self.pca.explained_variance_ratio_)

        plt.figure(figsize=(10, 6))
        plt.xlabel("n_componets")
        plt.ylabel('var')
        plt.plot(range(1,len(var)+1),var, marker='o', linestyle='-')
        plt.xticks(ticks=range(1,len(var)+1), labels=range(1,len(var)+1))
        plt.grid()
        plt.show()

        self.pca = PCA(n_components=4)

class ClusterAnalysis(DataAnalysis):

    def __init__(self, rng : np.random.RandomState, args : argparse.Namespace):
        """
        Initialises the class for cluster analysis.

        Arguments:
        - 'rng' - Random number generator for reproducible behaviour.
        - 'args' - Script arguments (passed down from the evaluator during evaluation).
        """
        super().__init__(rng, args)
        self._feature_processor = FeatureAnalysis(self._rng, self._args)
    
    def get_WSS(self,features):
        WSS =  []

        for clusters in range(1,20):

            model =  KMeans(n_clusters=clusters, random_state=self._rng).fit(features)

            WSS.append(model.inertia_)
        
        return WSS
    
    def find_elbow(self, WSS):
        distancies = []

        a = WSS[-1] - WSS[0]
        b = - (len(WSS) - 1)
        c = - (a + b* WSS[0])

        for i in range(len(WSS)):
            d = abs(a*(i+1) + b* WSS[i] + c) / np.sqrt(np.square(a)+np.square(b))
            distancies.append(float(d))

        elbow_point = np.argmax(distancies) + 1

        return elbow_point

    def analyze(self, train_data_path: Path):
        """Implementation of cluster analysis."""
        # TODO: You should implement your feature analysis and visualisation here.
        # - Do not forget to apply your final feature selection/transformation on the data before both training
        #   and testing. This method is only for finding an appropriate feature selection/transformation and
        #   its visualisation.
        
        data: DataStorage = DataStorage.fromFile(train_data_path)
        self._feature_processor.fit(data)
        self._feature_processor.transform(data)

        WSS = self.get_WSS(data.features)

        count_of_clusters = self.find_elbow(WSS)

        plt.figure(figsize=(9, 6))
        plt.plot(range(1, len(WSS) + 1), WSS, marker='o')
        plt.axvline(x=count_of_clusters, color='r', linestyle='-')
        plt.grid()
        plt.show()

class SupervisedClassificator:
    def __init__(self, parameters, method, name):
        self.parameters = parameters
        self.method = method
        self.name = name

    def searching_for_best_parameters(self, search,train_1,train_2):
        if search:

            grid = GridSearchCV(self.method, self.parameters, cv=10)

            grid.fit(train_1,train_2)

            self.method = grid.best_estimator_

            # print("**************************************************************************************************")
            # with open("parametry.txt", "a") as file:
            #      file.write(f"{self.name}: {grid.best_params_}")

            #  return 
        
        self.method.fit(train_1,train_2)

    def save_classificator(self,modelAnalysis):
        modelAnalysis.save( "{}_model.pkl".format(self.name), self.method)

class DecisionTree(SupervisedClassificator):
    def __init__(self, parameters):
        super().__init__(parameters,DecisionTreeClassifier(criterion='entropy'),
                        "DecisionTree_")

class MLPmodel(SupervisedClassificator):
    def __init__(self, parameters):
        super().__init__(parameters,MLPClassifier(activation='tanh'), 
                        "MLPClassifier_")

class GradientBoostingModel(SupervisedClassificator):
    def __init__(self, parameters):
                super().__init__(parameters,GradientBoostingClassifier(max_depth=5,subsample=0.8), 
                                "GradientBoosting_")
class Printer:
    def __init__(self):
        self.results = {}

    def get_confusionMatrix_precission_recall(self,model_name,labels,scores):

        confusionMatrix = confusion_matrix(labels, scores.argmax(axis=1))
        precision = precision_score(labels, scores.argmax(axis=1), average="macro")
        recall = recall_score(labels, scores.argmax(axis=1), average="macro")

        self.results[model_name] = (precision,recall)

        print(model_name)
        print("-------------")
        print(confusionMatrix)
        print("-------------")
        print("precision: " + str(precision))
        print("-------------")
        print("recall: " + str(recall))
        print("-------------")

    def plot_precision_recall_space(self):
        plt.figure(figsize=(9,7))

        for model_type,values in self.results.items():
            plt.scatter(values[0],values[1],label=model_type)

        plt.legend()
        plt.grid()
        plt.show()

class ModelAnalysis(DataAnalysis):

    def __init__(self, rng: np.random.RandomState, args: argparse.Namespace, evaluate_on_train: bool = True):
        """
        Initialises the class for machine learning model analysis.

        Arguments:
        - 'rng' - Random number generator for reproducible behaviour.
        - 'args' - Script arguments (passed down from the evaluator during evaluation).
        - 'evaluate_on_train' - Whether the implementation should be tested on the train data.
        """
        super().__init__(rng, args)
        self._evaluate_on_train = evaluate_on_train
        self._feature_processor = FeatureAnalysis(self._rng, self._args)
        self.test_data = None
        self.test_labels = None

    def train(self, data_path: Path, search: bool = True):
        """Implementation of model training and parameter search."""
        # TODO: You should implement your parameter search and training of the selected models in this function.
        # - If 'search' is 'False' then you should simply train your models with the best parameters that you
        #   discovered - possibly with hard-coded values.
        # - Otherwise, you should run your parameter search implementation - if you look through the parameter
        #   space manually then write the tested parameter combinations and model comparison code here.
        # - You can saved the trained models and objects with 'self.save'.
        # NOTE: If 'self._evaluate_on_train' is 'True' then you should split the training data into a train
        # and test subsets. Use the train subset for training and the test subset in the 'evaluation' function.
        # It is guaranteed that 'evaluate' will be called on the same instance (object) as 'train' if
        # 'self._evaluate_on_train' is 'True'.
        # - This way you can test your 'evaluate' implementation on independent data.
        # - You can create a subset of 'DataStorage' by calling 'data.getSubset' with an array of indices.
        # NOTE: Use 'self._feature_processor' to fit and apply feature selection/transformation.
        data = DataStorage.fromFile(data_path)

        self._feature_processor = FeatureAnalysis(self._rng, self._args)
        self._feature_processor.fit(data)
        self._feature_processor.transform(data)

        if self._evaluate_on_train:
            train_1_data,_, train_1_labels,_ = train_test_split(data.features, data.labels, test_size=0.2, random_state=self._rng) # Použil jsem místo data.getsubset
            self.test_data = train_1_data
            self.test_labels = train_1_labels

        else:
            train_1_data,self.test_data,train_1_labels, self.test_labels = train_test_split(data.features, data.labels, test_size=0.2, random_state=self._rng)

        parameters_decisionTree = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'],'max_depth': [5, 10, 15, 20, None],'min_samples_split': [2, 5, 10, 20],'min_samples_leaf': [1, 2, 5, 10],
            'max_features': [None, 'sqrt', 'log2'],'max_leaf_nodes': [None, 10, 20, 50],'min_impurity_decrease': [0.0, 0.01, 0.1]}
        
        decisionTree = DecisionTree(parameters_decisionTree)

        parameters_MLP = {'hidden_layer_sizes': [(50,), (100,),(150,)],'activation': ['relu', 'tanh'],'alpha': [0.00001,0.0001, 0.001],'learning_rate_init': [0.0001,0.001, 0.01],'early_stopping': [False,True]}

        mlp = MLPmodel(parameters_MLP)

        parameters_gradientBoosting = {'max_depth': [7,14,20],'subsample': [0.5,0.8,1.0,1.2]}

        gradientBoostingModel = GradientBoostingModel(parameters_gradientBoosting)

        for model in [decisionTree,mlp,gradientBoostingModel]:
            model.searching_for_best_parameters(search, train_1_data, train_1_labels)
            model.save_classificator(self)

    def evaluate(self, data_path: Path):
        """Implementation of model evaluation."""
        # TODO: Implement your model evaluation code in this function.
        # - This is the function mentioned in the 'Task 6' of the assignment PDF.
        # - You can load models and objects saved in 'train' with 'self.load'.
        data : DataStorage = DataStorage.fromFile(data_path)
        self._feature_processor.transform(data)

        if self.test_data is None:
            data_for_evaluate = data
        
        else:
            data_for_evaluate = DataStorage(features=self.test_data, labels=self.test_labels)

        models = ["DecisionTree__model.pkl", "MLPClassifier__model.pkl", "GradientBoosting__model.pkl"]

        printer = Printer()

        for model_type in models:
            model_type_name = model_type.split("__")[0]
            model = self.load(model_type)

            scores = model.predict_proba( data_for_evaluate.features)

            printer.get_confusionMatrix_precission_recall(model_type_name,data_for_evaluate.labels,scores)

        printer.plot_precision_recall_space()
