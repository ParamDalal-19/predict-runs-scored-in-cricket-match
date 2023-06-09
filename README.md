# Predict Runs Scored in a Cricket Match

This project aims to predict the number of runs scored in a cricket match using different machine learning algorithms. The dataset used for training and testing the models is stored in the "final_dataset.csv" file.


## Files

The project consists of the following files:

- `decisiontree.py`: Implements the Decision Tree regression algorithm to predict runs scored in a cricket match.
- `KNN.py`: Implements the K-Nearest Neighbors regression algorithm to predict runs scored in a cricket match.
- `lasso.py`: Implements the Lasso regression algorithm to predict runs scored in a cricket match.
- `PCA.py`: Applies Principal Component Analysis (PCA) to the dataset and visualizes the explained variance ratio and principal components.
- `random_forest.py`: Uses the Random Forest regression algorithm to predict runs scored in a cricket match.


## Dependencies

The following Python libraries are required to run the project:

- pandas
- numpy
- sklearn
- matplotlib

You can install the dependencies using the following command:

```bash
pip install pandas numpy scikit-learn matplotlib
```


## Usage

To use the project, follow these steps:

1. Make sure you have installed the required dependencies.
2. Place the `final_dataset.csv` file in the same directory as the Python files.
3. Open the desired file (decisiontree.py, KNN.py, lasso.py, PCA.py, or random_forest.py) in your Python editor.
4. Run the file to see the results, including accuracy metrics and visualizations.


## Credits

The project is based on the idea of predicting runs scored in a cricket match and is inspired by an article on machine learning in sports analytics.


## Future Scope

The project can be extended to predict the outcome of a cricket match, detect player performance patterns, or analyze various other aspects of the game. By modifying the input dataset and applying appropriate algorithms, it can also be adapted to analyze and predict outcomes in other sports.
