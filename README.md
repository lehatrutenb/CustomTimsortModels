# TimSort

## Methods:

## *SortingBenchmarkFramework(algo, model=None, save=None, epochs=None, model_params=None)*
Class for working with models

:param algo string: algorithm of training neural networks like fast, secondary, hard, specific

:param model: trained model

:param save bool: true, if during training program should save logs about modules, 
                                                           otherwise - false

:param epochs int: num of epochs for training

:param model_params: SortingBenchmarkFramework model params

:param data NamedTuple("train", "train_labels", "test",
                        "test_labels", "exam", "exam_labels"): just data

:param parameters NamedTuple("layer", "num_layer", "cheks_num", "neirons"): 
                             parameters from config file for generating neural network

:param data_params NamedTuple("std", "mean"): info about data


## *CreateAlgo(specific_params=None)*
Generates model with given parameters

:param specific_params NamedTuple("layer", "num_layer", "cheks_num", "neirons"):
                                            specific parameters for generating model


## *SaveModel(self, model_name_json=None, model_name=None)*
Save model

:param model_name_json string: save model settings in .json

:param model_name string: save trained model in tensorflow style 
                   (creates dirictory with name model_name)


## *LoadModel(model_name)*
Load model

:param model_name string: name of .json file with model parameters


## *LoadModel(model_name)*
Load model

:param model_name string: name of .json file with model parameters

## *LoadConfig(config_name)*
Loads config for generating models

:param config_name string: name of .json file with parameters for generating models


## *LoadData(path_to_data)*
Loads data for training and testing models

:param path_to_data string: path to .csv file with data with column DataPairs and 
                            (size.0, minrun.0) in each row


## *LoadDataConfig(path_to_data)*
Loads mean, std for given data in LoadData

:param path_to_data string: path to .json file with keys "mean", "std"


## *Train()*
Fits model on data from LoadData


## *Validate()*
Score model on exam data


## *PredictMinrunBySize(size)*
Predict minrun for array with given size

:param size int: size of array


## *Predict(test, test_labels=None)*
Predict minruns for Plots()

:param test [int]: array of arrays for predicting minruns

:param test_labels [int]: minruns for model where algo == "best algorithm"


## *Plots(models, name_of_result, dpi=100)*
Compares models on exem_data from LoadData()

:param models [SortingBenchmarkFramework]: array of models for comparing

:param name_of_result string: .jpg filename where plot will be saved in

:param dpi int: dpi for jpg (the more dpi the more datailed pic is)


## *CreateMinrunData(data, name_of_csv=None, k=3, minrun_step=1)*
Counts best minruns for given arrays

:param name_of_csv string: name of .csv file to save data to

:param k number of resorting for one minrun value. (the more the better)
:param minrun_step int: step of checking minrun


## *DivideArr(arr)*
Divide array to smaller arrays

:param arr []: arr to divide



## To generate and save model:

```
Model = SortingBenchmarkFramework(algo="fast", save=False)  # you can used algo="fast|secondary|hard"
Model.LoadData("Minruns_data.csv")  # path to data with minruns
Model.LoadDataConfig("data.json")  # path to data with mean and std
Model.LoadConfig("config.json")  # path to config file for generating models
Model.CreateAlgo()  # generate model
Model.Train()
Model.SaveModel("network.json", "network")
```

## To load model:

```
Model = SortingBenchmarkFramework(algo="fast", save=False)  # you can used algo="fast|secondary|hard"
Model.LoadData("Minruns_data.csv")  # path to data with minruns
Model.LoadDataConfig("data.json")  # path to data with mean and std
Model.LoadConfig("config.json")  # path to config file for generating models
Model.LoadModel("network.json") 
Model.Train()
```

## To generate minrun data:

```
CreateMinrunData(divideArr(some_array), name_of_csv="minrun_data.csv", minrun_step=10)  # better to use minrun step = some procents of len(some_array)
```
