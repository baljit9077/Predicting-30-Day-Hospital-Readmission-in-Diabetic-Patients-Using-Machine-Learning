Predicting 30-Day Hospital Readmission in Diabetic Patients Using Machine Learning
Chapter 1: Introduction
Diabetes is one of the most common long-term health conditions worldwide. It can lead to serious complications such as heart disease, kidney problems, nerve damage, eye disease, and repeated hospital admissions. Many diabetic patients require regular clinical monitoring, medication adjustments, and follow-up care after discharge.
Hospital readmission within 30 days is an important healthcare quality indicator. If a patient returns to hospital shortly after discharge, it may suggest that the patient required additional care, better discharge planning, or closer monitoring. Readmissions also increase healthcare costs and create pressure on hospital services.
Predicting 30-day readmission is therefore important because it can help healthcare professionals identify high-risk patients earlier. If hospitals know which patients are more likely to be readmitted, they can provide extra support, arrange follow-up appointments, review medication plans, and reduce avoidable readmissions.
The main aim of this project is to use machine learning techniques to predict whether diabetic patients will be readmitted within 30 days.
Research Objectives
The objectives of this project are:
1.	To clean and preprocess the diabetes hospital readmission dataset.
2.	To explore patterns in patient readmission, age, hospital stay, and medication variables.
3.	To transform the readmission outcome into a binary classification problem.
4.	To encode categorical variables into numerical form.
5.	To create new useful features from existing variables.
6.	To handle severe class imbalance using SMOTE and class weighting.
7.	To train and compare Logistic Regression, Random Forest, XGBoost, and XGBoost with class weighting.
8.	To identify the most important predictors of 30-day readmission.
9.	To discuss the strengths, limitations, and healthcare relevance of the models.



Chapter 2: Literature Review
Machine learning is increasingly used in healthcare because hospitals collect large amounts of patient data. These data can be used to predict outcomes such as disease risk, hospital length of stay, mortality, and readmission.
Traditional statistical models such as Logistic Regression are commonly used in healthcare because they are easy to interpret. However, healthcare data is often complex and contains non-linear relationships. For this reason, machine learning models such as Random Forest and XGBoost are often used because they can capture more complex patterns.
Hospital readmission prediction is challenging because readmitted patients usually make up a small proportion of the total dataset. This creates a class imbalance problem. In this project, only 4.55% of patients were readmitted within 30 days. This means that a model can achieve high accuracy by mostly predicting the majority class, but still fail to identify high-risk patients.
To address this issue, techniques such as SMOTE and class weighting can be used. SMOTE creates synthetic minority class examples to balance the training data. Class weighting gives more importance to the minority class during model training.
Previous studies show that ensemble models such as Random Forest and XGBoost often perform well in healthcare prediction tasks. However, for clinical problems, accuracy alone is not enough. Recall and F1-score are very important because the goal is to correctly identify high-risk patients.





Chapter 3: Dataset Description and Data Preprocessing
3.1 Dataset Description
The dataset used in this project is the Diabetes 130-US Hospitals dataset. It contains hospital records of diabetic patients collected from 130 hospitals in the United States between 1999 and 2008.
The dataset includes patient demographic details, admission information, laboratory tests, medication details, diagnosis codes, and readmission outcomes.
The original dataset contained:
Attribute	Value
Original records	101,766
Original variables	50
Final records	70,305
Final predictor variables	35
Target variable	readmitted
Important types of variables in the dataset included:
•	Age
•	Race
•	Gender
•	Admission type
•	Discharge disposition
•	Admission source
•	Time in hospital
•	Number of laboratory procedures
•	Number of procedures
•	Number of medications
•	Diagnosis codes
•	Diabetes medication variables
•	A1C result
•	Readmission status
The target variable was readmitted, which originally contained three categories:
•	NO
•	30
•	<30
The purpose of this project was to predict whether a patient would be readmitted within 30 days.
3.2 Removal of Duplicate Patients
The dataset contained multiple encounters for some patients. This means that the same patient could appear more than once. If duplicate patients are included in both training and testing data, it can lead to data leakage. Data leakage occurs when the model has already seen information about the same patient during training and then performs better during testing than it would in real life.
To avoid this issue, duplicate patients were removed using the patient_nbr column. The last record for each patient was kept.
After removing duplicate patient records, the dataset was reduced from 101,766 records to approximately 71,518 unique patient records.
This step ensured that each patient appeared only once in the final dataset.

3.3 Missing Value Handling
The dataset contained missing values represented by the symbol "?". These values were first replaced with NaN so that they could be handled properly in Python.
Several columns had a large number of missing values. Examples included:
Variable	Reason for concern
weight	Very high missing values
payer_code	Many missing values
medical_specialty	Many missing values
max_glu_serum	Very high missing values
race	Some missing values
diag_1	Small number of missing values
diag_2	Small number of missing values
diag_3	Small number of missing values
The columns weight, payer_code, medical_specialty, and max_glu_serum were removed because they had too many missing values and were not reliable for machine learning.
The columns encounter_id and patient_nbr were also removed because they were identification variables. These columns do not provide useful clinical information for predicting readmission.
The A1Cresult column had many missing values. Instead of removing it, missing values were replaced with NotTested. This was useful because not having an A1C test may itself provide important information about patient care.
Rows with missing values in diag_1, diag_2, and diag_3 were removed because these columns had only a small number of missing values.
3.4 Removal of Demographic Variables
The race and gender columns were removed from the dataset. This was done to simplify the model and reduce the risk of demographic bias.
Removing demographic variables can help focus the model more on clinical and hospital-related features, such as admission source, diagnoses, medication use, hospital visits, and time in hospital.

3.5 Target Variable Transformation
The original readmitted variable had three categories:
Original category	Meaning
NO	Patient was not readmitted
>30	Patient was readmitted after more than 30 days
<30	Patient was readmitted within 30 days
The aim of this project was to predict 30-day readmission. Therefore, the target variable was converted into binary form:
New value	Meaning
0	Not readmitted or readmitted after 30 days
1	Readmitted within 30 days
The transformation was:
•	NO = 0
•	30 = 0
•	<30 = 1
After transformation, the class distribution was:
Class	Meaning	Count	Percentage
0	Not readmitted or >30 days	67,104	95.45%
1	Readmitted within 30 days	3,201	4.55%
This showed that the dataset was highly imbalanced.


3.6 Removal of Low-Variance Variables
Some medication variables contained almost the same value for all patients. These columns were not useful because they provided little or no variation for the model to learn from.
Examples included:
•	examide
•	citoglipton
•	glimepiride-pioglitazone
•	acetohexamide
•	troglitazone
•	metformin-pioglitazone
•	metformin-rosiglitazone
•	glipizide-metformin
For example, examide and citoglipton had the value No for all 70,305 patients. Because these variables contained almost no information, they were removed from the dataset.
After removing low-variance variables, the dataset was reduced from 42 columns to 34 main features before later feature engineering.

Chapter 4: Exploratory Data Analysis
Exploratory Data Analysis was performed to understand the structure of the dataset, the distribution of the target variable, and the behaviour of important features.
4.1 Readmission Distribution Before Binary Transformation
Before converting the target variable, the readmission categories were:
Readmission category	Count	Percentage
NO	53,330	75.86%
>30	13,774	19.59%
<30	3,201	4.55%
Most patients were not readmitted. A smaller group was readmitted after more than 30 days, and only a very small group was readmitted within 30 days.
This showed that early readmission was rare in the dataset.
4.2 Readmission Distribution After Binary Transformation
After binary transformation, the distribution became:
Class	Count	Percentage
0	67,104	95.45%
1	3,201	4.55%
This showed a severe class imbalance. The majority class was much larger than the minority class.
This is important because machine learning models may become biased toward predicting class 0. For example, a model could achieve high accuracy by predicting almost all patients as not readmitted, but this would not be useful in healthcare because it would miss high-risk patients.
4.3 Age Distribution
The age distribution showed that most diabetic patients were older adults. The largest age groups were between 60 and 90 years old, especially the 70–80 age group.
This suggests that diabetes-related hospital admissions are more common among older patients. Age is therefore an important predictor because older patients may have more health complications, more diagnoses, and a higher risk of readmission.
The age categories were also converted into numerical midpoint values:
Age group	Encoded value
[0-10)	5
[10-20)	15
[20-30)	25
[30-40)	35
[40-50)	45
[50-60)	55
[60-70)	65
[70-80)	75
[80-90)	85
[90-100)	95
This allowed the age variable to be used in machine learning models.
4.4 Time in Hospital Distribution
The time in hospital variable showed that most patients stayed in hospital for a short period, mainly between 1 and 5 days. The number of patients decreased as length of stay increased.
This suggests that long hospital stays were less common. However, time in hospital may still be an important predictor because longer stays may indicate more serious illness or more complex treatment needs.
4.5 A1C Result and Readmission
The A1C result analysis showed that many patients were in the NotTested category. This group had the highest number of both readmitted and non-readmitted patients.
A1C testing is clinically important because it gives information about blood glucose control. However, the high number of NotTested values suggests that testing was not recorded for many patients.
Instead of removing this variable, NotTested was kept as a separate category because it may provide useful information about care patterns.
4.6 Medication Distributions
Medication columns were checked to understand how diabetes drugs were recorded.
Some medication variables showed useful variation. For example, insulin had the following values:
Insulin status	Count
No	33,853
Steady	21,775
Down	7,598
Up	7,079
This shows that insulin was an important medication variable because it had different values across patients.
Other variables had almost no variation. For example, examide and citoglipton had only the value No for all patients. These variables were removed.

Chapter 5: Encoding and Feature Engineering
5.1 Encoding Categorical Variables
Machine learning models require numerical input. Therefore, categorical variables were converted into numerical format.
5.2 Medication Encoding
Medication variables were encoded using the following mapping:
Original value	Encoded value
No	0
Steady	1
Up	2
Down	3
This encoding was applied to medication columns such as:
•	metformin
•	repaglinide
•	nateglinide
•	chlorpropamide
•	glimepiride
•	glipizide
•	glyburide
•	tolbutamide
•	pioglitazone
•	rosiglitazone
•	acarbose
•	miglitol
•	tolazamide
•	insulin
•	glyburide-metformin
This allowed the model to understand whether a medication was not used, remained steady, increased, or decreased.
5.3 A1C Result Encoding
The A1Cresult variable was encoded as:
A1C result	Encoded value
NotTested	0
Norm	1
>7	2
>8	3
This transformation allowed the model to use A1C result information as a numerical predictor.
5.4 Change Variable Encoding
The change variable shows whether there was a change in diabetes medication.
It was encoded as:
Original value	Encoded value
No	0
Ch	1
5.5 Diabetes Medication Encoding
The diabetesMed variable shows whether the patient was taking diabetes medication.
It was encoded as:
Original value	Encoded value
No	0
Yes	1
5.6 Diagnosis Encoding
The diagnosis variables diag_1, diag_2, and diag_3 contained many diagnosis codes. These were encoded using Label Encoding.
Label Encoding converts each diagnosis category into a numerical value. This made the diagnosis variables suitable for machine learning.
5.7 Feature Engineering
Two new features were created.
Total Visits
The first new feature was total_visits.
It was calculated as:
total_visits = number_outpatient + number_emergency + number_inpatient
This variable represents the total number of hospital-related visits made by a patient. Patients with more previous visits may have more complex health conditions and may be more likely to be readmitted.
Medications Per Day
The second new feature was meds_per_day.
It was calculated as:
meds_per_day = num_medications / time_in_hospital
This variable represents medication intensity during the hospital stay. A higher value may suggest that a patient required more treatment in a shorter period.
After encoding and feature engineering, all variables were numerical. The dataset contained 70,305 rows and 36 columns, including the target variable.

Chapter 6: Data Shuffling and Train-Test Split
Before building the machine learning models, the dataset was randomly shuffled. This was done to ensure that the rows were randomly distributed and that there was no ordering effect in the dataset.
A fixed random seed was used to make the results reproducible.
The dataset was then divided into predictor variables and the target variable.
The readmitted column was selected as the target variable y. All other variables were selected as predictor variables X.
The resulting shapes were:
Data	Shape
X	70,305 rows × 35 features
y	70,305 rows
This step is important because supervised machine learning models learn the relationship between the input features and the target outcome.
The data was then split into training and testing sets using an 80:20 split. Stratified sampling was used so that both the training and testing sets kept the same class distribution as the original dataset.
The training set was used to train the models, while the testing set was used to evaluate model performance on unseen data.

Chapter 7: Handling Class Imbalance
7.1 Class Imbalance Problem
The dataset had a severe class imbalance problem. Only 4.55% of patients were readmitted within 30 days.
This is a serious issue because machine learning models can become biased toward the majority class. For example, if a model predicts every patient as not readmitted, it would still achieve high accuracy because most patients belong to class 0. However, this model would be clinically useless because it would fail to identify high-risk patients.
Therefore, accuracy alone was not enough to evaluate performance. Recall and F1-score for class 1 were more important.
7.2 SMOTE
SMOTE stands for Synthetic Minority Oversampling Technique. It creates synthetic examples of the minority class by using nearest neighbours from existing minority class samples.
In this project, SMOTE was applied only to the training data. This is important because applying SMOTE before train-test splitting would cause data leakage.
After SMOTE, the minority class in the training set was increased to match the majority class. This helped the models learn patterns from both classes more equally.
7.3 SMOTEENN
SMOTEENN was also investigated. It combines SMOTE oversampling with Edited Nearest Neighbours cleaning. This method creates synthetic minority examples and removes noisy examples from the dataset.
7.4 XGBoost Class Weighting
Class weighting was also used with XGBoost. The class weight was calculated using the ratio between the majority and minority classes.
This method gives more importance to the minority class during training without creating synthetic data.

Chapter 8: Machine Learning Models
Four main models were developed and compared.
8.1 Logistic Regression
Logistic Regression was used as a baseline model. It is a simple and interpretable classification model commonly used in healthcare prediction.
Because Logistic Regression is easier to explain, it is useful for understanding the basic relationship between patient features and readmission.
8.2 Random Forest
Random Forest is an ensemble model that builds many decision trees and combines their predictions.
It is useful because it can capture non-linear relationships and interactions between variables. However, in imbalanced datasets, Random Forest may still favour the majority class.
8.3 XGBoost with SMOTE
XGBoost is a powerful gradient boosting algorithm. It builds trees sequentially, where each new tree tries to correct the errors made by previous trees.
XGBoost was trained using the SMOTE-balanced training data. This was done to improve the model’s ability to detect readmitted patients.
8.4 XGBoost with Class Weighting
XGBoost was also trained using class weighting. This approach gives more importance to the minority class during model training.
This model was important because it aimed to improve recall for patients readmitted within 30 days.

Chapter 9: Model Evaluation and Results
The models were evaluated using accuracy, recall, and F1-score.
In this healthcare prediction problem, recall for class 1 is very important. Recall shows how many actual readmitted patients were correctly identified by the model.
The results were:
Model	Accuracy	Recall for Class 1	F1-score for Class 1
Logistic Regression	68%	0.40	0.10
Random Forest	93%	0.05	0.07
XGBoost + SMOTE	94%	0.06	0.08
XGBoost + Class Weight	81%	0.36	0.14
9.1 Logistic Regression Result
Logistic Regression achieved lower accuracy compared with Random Forest and XGBoost. However, it achieved a better recall for class 1 than the high-accuracy models.
This means Logistic Regression identified more readmitted patients, but it also made more false positive predictions.
9.2 Random Forest Result
Random Forest achieved high accuracy of 93%. However, its recall for class 1 was only 0.05. This means it failed to identify most patients who were readmitted within 30 days.
This shows that high accuracy can be misleading in imbalanced healthcare datasets.
9.3 XGBoost with SMOTE Result
XGBoost with SMOTE achieved high accuracy of 94%, but the recall for class 1 was only 0.06. This means that despite balancing the training data with SMOTE, the model still struggled to identify the minority class.
9.4 XGBoost with Class Weighting Result
XGBoost with class weighting achieved 81% accuracy, 0.36 recall, and 0.14 F1-score for class 1.
Although the accuracy was lower than Random Forest and XGBoost with SMOTE, the model performed better at identifying readmitted patients. In healthcare, this is more useful because missing a high-risk patient can have serious consequences.
Therefore, XGBoost with class weighting was selected as the best model overall.


Chapter 10: Feature Importance
Feature importance analysis was performed to identify which variables had the strongest influence on readmission prediction.
The most important predictors were:
1.	Age
2.	Admission source
3.	Metformin
4.	Glyburide
5.	Rosiglitazone
6.	Change in medication
7.	Number of diagnoses
8.	Total visits
10.1 Age
Age was one of the strongest predictors. Older patients are more likely to have multiple health conditions and complications, which may increase readmission risk.
10.2 Admission Source
Admission source was important because it shows how the patient entered the hospital. Patients admitted through emergency or referral routes may have different levels of risk.
10.3 Medication Variables
Medication variables such as metformin, glyburide, and rosiglitazone were important. These variables may reflect the patient’s diabetes treatment plan and severity of condition.
10.4 Change in Medication
Medication change was also important. A change in medication may indicate that the patient’s diabetes was unstable or required adjustment during admission.
10.5 Number of Diagnoses
Patients with a higher number of diagnoses may have more complex medical conditions. This increases the risk of complications and possible readmission.
10.6 Total Visits
Total visits was an engineered feature and appeared among the important predictors. This shows that previous healthcare usage is useful for predicting readmission risk.

Discussion
The results show that predicting 30-day readmission in diabetic patients is challenging because the dataset is highly imbalanced. Only a small percentage of patients were readmitted within 30 days.
Random Forest and XGBoost with SMOTE achieved high accuracy, but their recall for class 1 was very low. This means they mainly predicted the majority class and failed to identify many high-risk patients.
This is a common problem in healthcare machine learning. A model with high accuracy may not be clinically useful if it cannot detect the patients who need the most attention.
XGBoost with class weighting provided the best balance because it improved recall for the minority class. It identified more readmitted patients than Random Forest and XGBoost with SMOTE.
The feature importance results also made sense from a healthcare perspective. Age, admission source, medication use, medication changes, number of diagnoses, and total visits are all relevant to patient health and readmission risk.
The engineered feature total_visits was useful because it captured overall healthcare usage. Patients with more outpatient, emergency, and inpatient visits may have more serious or ongoing health problems.
The meds_per_day feature also added useful information about medication intensity during the hospital stay.
Overall, this project shows that machine learning can help identify patients at risk of early readmission, but model evaluation must focus on recall and F1-score rather than accuracy alone.

Limitations
This project has several limitations.
First, the dataset was highly imbalanced. Only 4.55% of patients were readmitted within 30 days. This made it difficult for models to learn patterns from the minority class.
Second, some important clinical variables had many missing values and had to be removed. For example, weight and max_glu_serum may have been clinically useful, but they could not be used because of excessive missing data.
Third, diagnosis codes were encoded using Label Encoding. This method converts categories into numbers, but it may not fully capture the medical meaning of diagnosis codes.
Fourth, the model was trained and tested on one dataset only. No external hospital dataset was used for validation.
Fifth, demographic variables such as race and gender were removed to reduce bias and simplify the model. However, removing them may also remove some information related to patient risk.
Finally, the project did not include advanced explainability techniques such as SHAP values. SHAP could provide a clearer explanation of how each feature affects individual predictions.


Conclusion
This project developed machine learning models to predict 30-day hospital readmission among diabetic patients.
The dataset was cleaned by removing duplicate patients, handling missing values, removing low-information columns, encoding categorical variables, and creating new features. The target variable was converted into binary form, where class 1 represented readmission within 30 days.
Exploratory analysis showed that the dataset was highly imbalanced, with only 4.55% of patients readmitted within 30 days. This made model evaluation challenging.
Several models were trained, including Logistic Regression, Random Forest, XGBoost with SMOTE, and XGBoost with class weighting. Although Random Forest and XGBoost with SMOTE achieved high accuracy, they performed poorly in identifying readmitted patients.
XGBoost with class weighting achieved the best overall balance because it improved recall for the minority class. This made it the most suitable model for this healthcare prediction problem.
The most important predictors included age, admission source, medication variables, medication change, number of diagnoses, and total visits.
Overall, this study shows that machine learning can support healthcare professionals by helping identify diabetic patients who may be at higher risk of early readmission. However, further improvement is needed before such a model could be used in real clinical practice.

