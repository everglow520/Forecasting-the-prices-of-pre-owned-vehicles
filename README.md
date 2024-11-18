# Readme 
In this project,we mainly use the **random forest** after comparing with other model,which is include Linear Regression,Decision Tree,Adaboost, Gradient Boosting.
After compared with it's score,we choose the best one to be our project's main model.
The picture is the random forest's schematic diagram:
![Random-Forest-Algorithm](https://github.com/user-attachments/assets/47a4f1bf-e351-45dd-9f06-3ff11f1e4c1e)
We will explore this principle in more detail in a later section.
# Dataset
## Form
We choose the website https://sz.58.com/ershouche/?PGTID=0d30001d-0000-4622-24da-a55ad27728f0&ClickID=57 and crawl the data.
The below code is the method how we crawl it:
```
def parse_page(soup):
     cars = soup.find_all('li', class_='cards-li')
     for car in cars:
        try:
            title = car.find('h4', class_='card-name').get_text(strip=True)
            price = car.find('span', class_='pirce').get_text(strip=True)
            str = car.find('p', class_='cards-unit').get_text(strip=True)
            box = str.split('')
            mileage = box[0]
            year = box[1]
            city = box[2]
            
            car_data.append([title, price, mileage, year, city])
        except AttributeError:
            continue
```
Then we get 3641 rows and 8 columns data.
## Preprocessing
In this project,we mainly use **one-hot encoder** and **standard scaler** to preprocessing our data to make it more comprehendible for the computer.
And what is it? Why we use it?
The reason is that the computer can't comprehend the data we crwal directly.We must convert it to computer language.
### Onehotencoder
One-hot encoding, also known as one-bit effective encoding, mainly uses N-bit status registers to encode N states, each state is composed of its own independent register bits, and only One bit is valid at any time.  
It is a representation of categorical variables as binary vectors. This first requires mapping class values to integer values. Each integer value is then represented as a binary vector that is zero except for the index of the integer, which is labeled 1.  
Unique thermal coding is because most algorithms are calculated based on the metric in the vector space, in order to make the values of variables that are not partial ordered, and are equidistant to the circle point. Using one-hot coding, the value of discrete feature is extended to Euclidean space, and a value of discrete feature corresponds to a point in Euclidean space. Using one-hot encoding for discrete features makes the distance between features more reasonable. After the discrete features are one-hot encoded, the encoded features, in fact, can be regarded as continuous features in each dimension. Each dimension can be normalized in the same way as the normalization method for continuous features. For example, normalization to [-1,1] or normalization to mean 0 and variance 1.  
The discrete features are mapped to the Euclidean space through one-hot coding, because, in the regression, classification, clustering and other machine learning algorithms, the calculation of the distance between the features or the calculation of the similarity is very important, and our commonly used distance or similarity calculation is the similarity calculation in the Euclidean space, the calculation of cosine similarity, based on the Euclidean space.
```
oneHotEncoder = OneHotEncoder(drop='first')
categorical_scaler = oneHotEncoder.fit_transform(df[categorical_features]).toarray()
print(categorical_scaler)
```
The result is below:
![254ff9f5492fb32c424f1219be13ea1](https://github.com/user-attachments/assets/468c41f3-5fde-487d-af8d-e9d16ab50c90)

### Standardscaler
In machine learning, data preprocessing is a crucial step. StandardScaler is a class in the sklearn.preprocessing module that is used to standardize data processing. Standardization is the process of scaling features to a given range (usually mean 0, standard deviation 1), StandardScaler achieves this transformation by calculating the mean and standard deviation of feature columns, it is a common technique in data preprocessing, and is crucial to the performance improvement of many machine learning algorithms.  
The main benefits of standardization include, but are not limited to:  
  1.Improve model performance: Many machine learning algorithms, especially those based on distance (such as K-nearest neighbors, support vector machines, etc.) and algorithms that need to compute covariance matrices (such as PCA, linear regression, etc.), will perform better when the input data has the same scale. Standardized processing can help these algorithms achieve better performance.  
  2.Preventing gradient disappearance or explosion: For deep learning models, especially those that use gradient descent optimization, too large a difference in the scale of the input features can lead to problems with gradient disappearance or explosion. Standardization can solve this problem effectively and make the model training more stable.  
  3.Makes the data easier to interpret: Standardized data has a mean of 0 and a standard deviation of 1, which makes the data distribution more concentrated and easier to understand and interpret.  
The principle of StandardScaler is based on the normalization of the data, which is a linear transformation of the data so that the data for each feature (that is, each attribute or each column) follows the standard normal distribution, that is, the mean is 0 and the standard deviation is 1. This conversion is done by calculating the mean and standard deviation of the features.  
A key benefit of the standardized treatment is that it is not affected by the original scale of the features, because the transformation is based on the statistical properties (mean and standard deviation) of each feature itself. This means that no matter how large or small the original value of the feature is, it will all have the same scale after normalization, which can help improve the performance of many machine learning algorithms.
```
standardScaler = StandardScaler()
numerical_scaler = standardScaler.fit_transform(df[numerical_features])
print(numerical_scaler)
```
The result is below:
![b85ed778b37ecc48756f8d6e9895bbf](https://github.com/user-attachments/assets/7b792ffc-7b5a-4e8f-b201-fd147aa85817)

# Random forest
## What is the Random Forest Algorithm?
Random Forest algorithm is a powerful tree learning technique in Machine Learning. It works by creating a number of Decision Trees during the training phase. Each tree is constructed using a random subset of the data set to measure a random subset of features in each partition. This randomness introduces variability among individual trees, reducing the risk of overfitting and improving overall prediction performance.

In prediction, the algorithm aggregates the results of all trees, either by voting (for classification tasks) or by averaging (for regression tasks) This collaborative decision-making process, supported by multiple trees with their insights, provides an example stable and precise results. Random forests are widely used for classification and regression functions, which are known for their ability to handle complex data, reduce overfitting, and provide reliable forecasts in different environments.
## What are Ensemble Learning models?
Ensemble learning models work just like a group of diverse experts teaming up to make decisions – think of them as a bunch of friends with different strengths tackling a problem together. Picture it as a group of friends with different skills working on a project. Each friend excels in a particular area, and by combining their strengths, they create a more robust solution than any individual could achieve alone.

Similarly, in ensemble learning, different models, often of the same type or different types, team up to enhance predictive performance. It's all about leveraging the collective wisdom of the group to overcome individual limitations and make more informed decisions in various machine learning tasks. Some popular ensemble models include- XGBoost, AdaBoost, LightGBM, Random Forest, Bagging, Voting etc.

## What is Bagging and Boosting?
Bagging is an ensemble learning model, where multiple week models are trained on different subsets of the training data. Each subset is sampled with replacement and prediction is made by averaging the prediction of the week models for regression problem and considering majority vote for classification problem.

Boosting trains multiple based models sequentially. In this method, each model tries to correct the errors made by the previous models. Each model is trained on a modified version of the dataset, the instances that were misclassified by the previous models are given more weight. The final prediction is made by weighted voting.

## Algorithm for Random Forest Work:

Step 1: Select random K data points from the training set.
Step 2:Build the decision trees associated with the selected data points(Subsets).
Step 3:Choose the number N for decision trees that you want to build.
Step 4:Repeat Step 1 and 2.
Step 5: For new data points, find the predictions of each decision tree, and assign the new data points to the category that wins the majority votes.

## How Does Random Forest Work?
The random Forest algorithm works in several steps which are discussed below-->

Ensemble of Decision Trees: Random Forest leverages the power of ensemble learning by constructing an army of Decision Trees. These trees are like individual experts, each specializing in a particular aspect of the data. Importantly, they operate independently, minimizing the risk of the model being overly influenced by the nuances of a single tree.
Random Feature Selection: To ensure that each decision tree in the ensemble brings a unique perspective, Random Forest employs random feature selection. During the training of each tree, a random subset of features is chosen. This randomness ensures that each tree focuses on different aspects of the data, fostering a diverse set of predictors within the ensemble.
Bootstrap Aggregating or Bagging: The technique of bagging is a cornerstone of Random Forest's training strategy which involves creating multiple bootstrap samples from the original dataset, allowing instances to be sampled with replacement. This results in different subsets of data for each decision tree, introducing variability in the training process and making the model more robust.
Decision Making and Voting: When it comes to making predictions, each decision tree in the Random Forest casts its vote. For classification tasks, the final prediction is determined by the mode (most frequent prediction) across all the trees. In regression tasks, the average of the individual tree predictions is taken. This internal voting mechanism ensures a balanced and collective decision-making process.
