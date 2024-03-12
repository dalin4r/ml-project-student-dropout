# ml-project-student-dropout

## Overview

For our ML Project, we decided to explore certain variables responsible for predicting academic success in a college setting. After scanning over the data set, we came to the conclusion that we wanted to use debtors, occupations of parents, and scholarship holders as the variables that best predict academic success. 

We chose the occupations of parents as the primary attribute because higher levels of education may have a substantial influence on children. This is related to academic success as it is somewhat reliant on income levels. For example, students with higher levels of income have greater access to tutors and academic resources, which boost academic performance. The model aims to make comparisons between students of lower and higher social strata, predicting their academic performance. When it comes to academic success, we were given the performance metrics of students from a university for the first two semesters, so based on their economic backgrounds, we’ll predict grades for both. 

The stress of debt can influence how well a student does well in school because of the potential need to partake in a part-time job lowering the amount of time they are able to commit towards academics. 

Scholarships provide students with financial support, possibly reducing the necessity to work solely to pay for school. It additionally can serve as motivation or a stressor for the student to perform well, as scholarships commonly will require a student to attain a certain GPA to continue receiving support. 

With the described variables, we will conduct exploratory data analysis to understand the relationships between them and a student’s academic success while describing patterns, correlation between variables, and potential outliers. We will determine the most significant factor in gauging academic success by utilizing regression analysis to build an accurate predictive model. Through methods learned in class, we will validate our model’s accuracy and reliability, ensuring it is not biased, and provides meaningful insight on how academic success is influenced by the selected variables. 

## How to run frontend

1. Make sure you are in the frontend directory.
2. In the terminal, type `flask run` to start the app. The output should have the link to the development server the app runs on.
3. Choose a model to run and press the `submit` button in the app. Note that running the SVM model will take around 3 minutes because it is being run multiple times with a variable number of features.
4. `CTRL+C` to stop the app.

## Folders & Files

**frontend** - Directory that contains the flask app

**notebook** - Directory that contains the Jupyter Notebook where we ran our models

**README.md** -  This file explains the context of this project and has instructions on how to run the main app

## Contributing

If you would like to contribute, follow these steps:
1. Fork this repo.
2. Make your changes and commit.
3. Push changes to your fork.
4. Submit a pull request against the `main` branch of this repo.
