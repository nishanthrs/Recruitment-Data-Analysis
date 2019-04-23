
# coding: utf-8

# In[13]:

# Import libraries
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns


# In[14]:

# Load csv as pandas dataframes
traders_df = pd.read_excel("xxx_2018 Quant Trader Full Time and Internship_Data.xlsx", header=0)
traders_df = traders_df.drop([0])

# Split into full-time and intern applications
'''
grads_df = traders_df[traders_df['Job_Posting_Title'] == 'REQ-00438 Graduate Quant Trader (Open)']
interns_df = traders_df[traders_df['Job_Posting_Title'] == 'REQ-00446 Quant Trader Intern - Summer 2019 (Fill Date: 06/10/2019)']
'''

# Separate the features of the dataset into continuous and categorical
features = np.array(traders_df.columns)
continuous_features = ['Codility Battleship', 'Saville Diagramatic', 'Saville Numerical', 'Best Score', 'GPA', 'Jobs Applied', ' Applications']
categorical_features = np.setdiff1d(features, continuous_features)

# Print descriptions of the dataset: mean, std, quartiles, etc. of each feature of dataset
print(traders_df.describe())


# In[15]:

# Get unique vals of features of dataset to gain better understanding of each feature
for categorical_feature in categorical_features:
    # print(traders_df[categorical_feature].describe())
    unique_vals = traders_df[categorical_feature].unique()
    # print("List of unique vals for " + categorical_feature + ": " + str(unique_vals))
    
for continuous_feature in continuous_features:
    # print(traders_df[continuous_feature].describe())
    unique_vals = traders_df[continuous_feature].unique()
    # print("List of unique vals for " + continuous_feature + ": " + str(unique_vals))
    
sat_df = traders_df[traders_df['Best standardized test'] == 'SAT']
sat_df.describe()


# ## Cleaning Data Pipeline <br>
# 
# <font size=3>
# 1. Fill in missing values of University, Last Degree, Will Require Visa Sponsorship, Sponsorship, Best Standardized Test, Best Score, GPA as Not Listed or 0 <br> <br>
# 
# 2. Break up double majors by adding in new row with second major (treat double major as two applicants, each one with different major) <br> <br>
# 
# 3. Change type of and standardize Best Score <br> <br>
# 
# 4. Bin graduation date by year and month (exclude day) <br> <br>
# 
# 5. Bin colleges of each university (non-standardized university names) together: <br> 
# Baruch, CMU, MIT, UIUC, Caltech, Courant Institute, Columbia, Georgia Tech, IIT, Indiana University, John Hopkins, New York University, Northwestern University, University of Toronto, Rutgers, Stevens Institude, Stony Brook, Texas AM, UPenn, Ohio State, UChicago, UT Austin, UCLA, UMichigan, University of North Carolina, University of Pittsburgh, University of Waterloo, University of Wisconsin, WashU <br> <br>
# 
# 6. Change data type of and standardize GPA: <br> 
# Scale from 5 to 4 for MIT, University of Warsaw <br>
# Scale from 100 to 4 for all GPAs above 5 <br> <br>
# 
# 7. Fill in missing values in Last Stage and Disposition Reason <br> <br>
# 
# 8. Encode categorical features (reference sklearn.preprocessing and articles on how to properly do this) <br> <br>
# </font>

# In[16]:

# 1. Iterate through each feature and fill in missing values with 'Not Listed' for discrete features and -1 for categorical features
def fill_missing_discrete_val(x):
    x = str(x)
    if x == 'nan' or x == '-' or x == '':
        return 'Not Listed'
    return x

for categorical_feature in categorical_features:
    traders_df[categorical_feature] = traders_df[categorical_feature].apply(lambda x : fill_missing_discrete_val(x))
    
def fill_missing_continuous_val(x):
    x = str(x)
    if x == 'nan' or x == '-' or x == '':
        return 0.0
    return float(x)
    
for continuous_feature in continuous_features:
    traders_df[continuous_feature] = traders_df[continuous_feature].apply(lambda x : fill_missing_continuous_val(x))


# In[17]:

# 2. If an applicant has double major (; exists under Major), then create additional row with second major
# Treat an applicant with double major as two applicants with first and second major
majors_cleaned_df = pd.DataFrame(columns=traders_df.columns)

def add_row_for_double_major(majors_cleaned_df, traders_df):
    for applicant in traders_df.iterrows():
        applicant = applicant[1]
        if ';' in applicant['Major']:
            majors = applicant['Major'].split(';')
            for major in majors:
                major = major.strip()
                applicant['Major'] = major
                majors_cleaned_df = majors_cleaned_df.append(applicant)
        else:
            majors_cleaned_df = majors_cleaned_df.append(applicant)
    return majors_cleaned_df
    
majors_cleaned_df = add_row_for_double_major(majors_cleaned_df, traders_df)
majors_cleaned_df = majors_cleaned_df.reset_index()
majors_cleaned_df


# In[18]:

# 3. Normalize SAT/ACT scores on a scale from 0 to 100
scores_cleaned_df = majors_cleaned_df

# Based on table at https://blog.prepscholar.com/act-to-sat-conversion
def normalize_score(applicant):
    score_type = applicant['Best standardized test']
    score = applicant['Best Score']
    if score == '-' or score == '' or score_type == 'Other':
        return 0
    else:
        score = int(score)
        if score_type == 'SAT':
            if score > 2310:
                score = 36
            elif score > 2240:
                score = 35
            elif score > 2160:
                score = 34
            elif score > 2090:
                score = 33
            elif score > 2030:
                score = 32
            elif score > 1980:
                score = 31
            elif score > 1930:
                score = 30
            elif score > 1880:
                score = 29
        score = (score / 36) * 100
    return round(score, 2)

scores_cleaned_df['Best Score'] = pd.Series([normalize_score(scores_cleaned_df.iloc[i]) for i in range(0, scores_cleaned_df.shape[0])])
scores_cleaned_df


# In[19]:

# 4. Bin graduation date by year and month (exclude day since there's too much unnecessary variance) 
grad_dates_cleaned_df = scores_cleaned_df

def bucket_grad_dates(grad_date):
    return grad_date[0:7]

def bucket_applied_dates(applied_date):
    return applied_date[0:7]

grad_dates_cleaned_df['Graduation Date'] = grad_dates_cleaned_df['Graduation Date'].apply(lambda date: bucket_grad_dates(date))
grad_dates_cleaned_df['Applied On'] = grad_dates_cleaned_df['Applied On'].apply(lambda date: bucket_applied_dates(date))


# In[20]:

# 5. Bin colleges of each university (non-standardized university names) together: 
# Baruch, CMU, MIT, UIUC, Caltech, Columbia, Georgia Tech, IIT, Indiana University, 
# John Hopkins, New York University, Northwestern University, University of Toronto, Rutgers, 
# Stevens Institute, Stony Brook, Texas AM, UPenn, Ohio State, UChicago, UT Austin, UCLA, UMichigan, 
# University of North Carolina, University of Pittsburgh, University of Waterloo, University of Wisconsin, 
# WashU, Western University

universities_cleaned_df = grad_dates_cleaned_df

def bucket_universities(applicant):
    university = applicant['University'].lower()
    if university == '' or university == '-' or university == 'nan':
        return 'Not Listed'
    if 'baruch college' in university:
        return 'Baruch College'
    if 'california institute' in university:
        return 'Caltech'
    if 'carnegie mellon' in university:
        return 'CMU'
    if ('champaign' in university or 'university of illinois' in university) and 'chicago' not in university:
        return 'UIUC'
    if 'columbia' in university:
        return 'Columbia University'
    if 'duke' in university:
        return 'Duke University'
    if 'fordham' in university:
        return 'Fordham University'
    if 'georgia institute of technology' in university or 'georgia tech' in university:
        return 'Georgia Tech'
    if 'harvard' in university:
        return 'Harvard University'
    if 'indian institute of technology' in university:
        return 'Indian Institute of Technology'
    if 'indiana university' in university:
        return 'Indiana University'
    if 'johns hopkins' in university:
        return 'Johns Hopkins University'
    if 'new york university' in university or 'leonard n. stern' in university or 'nyu' in university:
        return 'New York University'
    if 'london school of economics' in university:
        return 'London School of Economics'
    if 'louisiana state university' in university:
        return 'Louisiana State University'
    if 'loyola' in university:
        return 'Loyola University'
    if 'manhattan' in university:
        return 'Manhattan University'
    if 'massachusetts' in university and 'technology' in university:
        return 'MIT'
    if 'michigan state university' in university:
        return 'Michigan State University'
    if 'northwestern' in university:
        return 'Northwestern University'
    if 'university of toronto' in university:
        return 'University of Toronto'
    if 'berkeley' in university or 'uc, berkeley' in university:
        return 'UC Berkeley'
    if 'rutgers' in university:
        return 'Rutgers University'
    if 'ohio state' in university:
        return 'Ohio State University'
    if 'stevens institute of technology' in university:
        return 'Stevens Institute of Technology'
    if 'stony brook university' in university:
        return 'Stony Brook University'
    if 'texas a&m' in university:
        return 'Texas A&M University'
    if 'george washington' in university:
        return 'George Washington University'
    if 'university of chicago' in university:
        return 'University of Chicago'
    if 'university of hong kong' in university:
        return 'University of Hong Kong'
    if 'north carolina' in university:
        return 'University of North Carolina'
    if 'ucla' in university or ('los angeles' in university and 'california' in university):
        return 'UCLA'
    if 'davis' in university:
        return 'UC Davis'
    if 'riverside' in university:
        return 'UC Riverside'
    if 'santa barbara' in university:
        return 'UC Santa Barbara'
    if 'santa cruz' in university:
        return 'UC Santa Cruz'
    if 'irvine' in university:
        return 'UC Irvine'
    if 'san diego' in university:
        return 'UC San Diego'
    if 'university of maryland' in university:
        return 'University of Maryland'
    if 'amherst' in university:
        return 'Amherst College'
    if 'university of michigan' in university:
        return 'University of Michigan'
    if 'university of missouri' in university:
        return 'University of Missouri'
    if 'university of minnesota' in university:
        return 'University of Minnesota'
    if 'university of pennsylvania' in university:
        return 'University of Pennsylvania'
    if 'university of virginia' in university:
        return 'University of Virginia'
    if 'waterloo' in university:
        return 'University of Waterloo'
    if 'university of wisconsin' in university:
        return 'University of Wisconsin'
    if 'university of south carolina' in university:
        return 'University of South Carolina'
    if 'university of southern california' in university:
        return 'USC'
    if 'washington university' in university:
        return 'Washington University in St. Louis'
    if 'western university' in university:
        return 'Western University'
    return applicant['University']

# TODO: Apply lambda transformation instead since transformation only uses university feature
universities_cleaned_df['University'] = pd.Series([bucket_universities(universities_cleaned_df.iloc[i]) for i in range(0, universities_cleaned_df.shape[0])])
universities_cleaned_df


# In[21]:

# 6. Change data type of and standardize GPA: 
# Scale from 5 to 4 for MIT, University of Warsaw 
# Scale from 100 to 4 for all GPAs above 5: https://pages.collegeboard.org/how-to-convert-gpa-4.0-scale
# Scale from 10 to 4: 
# https://www.scholaro.com/Forum/yaf_postst4489_BITS-Pilani---India.aspx (India)
# https://student.uva.nl/en/content/az/grading-scheme/grading-scheme.html?1554308653321 (Amsterdam)


cleaned_gpas_df = universities_cleaned_df

def normalize_gpa(applicant):
    gpa = applicant['GPA']
    university = applicant['University'].lower()
    
    if gpa == '-' or gpa == '':
        return 0.0
    
    gpa = float(gpa)
    
    # Different scaling for some international (on 10.0 scale)
    if university == 'indian institute of technology' or 'bits pilani' in university:
        if gpa > 4.0: # Not converted to 4.0 scale
            if gpa > 9.5:
                return 4.0
            elif gpa > 9.0:
                return 3.67
            elif gpa > 8.0:
                return 3.0
            elif gpa > 7.0:
                return 2.7
            elif gpa > 6.0:
                return 2.3
            elif gpa > 5.0:
                return 2.0
            return 1.7
        return gpa
    elif 'amsterdam' in university:
        if gpa > 4.0:
            if gpa > 8.3:
                return 4.0
            elif gpa > 7.8:
                return 3.7
            elif gpa > 7.3:
                return 3.3
            elif gpa > 7.0:
                return 3.0
            elif gpa > 6.7:
                return 2.7
            elif gpa > 6.4:
                return 2.3
            elif gpa > 5.5:
                return 2.0
            return 1.7
    # Hard-coded scaling: terrible coding practice!, but figured I would just scale these 2 stupid GPA anomalies from CMU
    if applicant['ID'] == 'C43521':
        return 3.7
    if applicant['ID'] == 'C43172':
        return 4.0
    
    if gpa > 4.0 and gpa <= 5.0:
        return gpa / 5 * 4
    if gpa > 5.0:
        if gpa >= 93.0:
            return 4.0
        elif gpa >= 90.0:
            return 3.7
        elif gpa >= 87.0:
            return 3.3
        elif gpa >= 83.0:
            return 3.0
        elif gpa >= 80.0:
            return 2.7
        elif gpa >= 77.0:
            return 2.3
        elif gpa >= 73.0:
            return 2.0
        elif gpa >= 70.0:
            return 1.7
        else:
            return 1.0
    return gpa

cleaned_gpas_df['GPA'] = pd.Series([normalize_gpa(cleaned_gpas_df.iloc[i]) for i in range(0, cleaned_gpas_df.shape[0])])
cleaned_gpas_df


# In[22]:

# 7. Fill in missing values of Last Stage with 'Applied' (default)

cleaned_last_stage_df = cleaned_gpas_df

def replace_missing_vals(last_stage):
    last_stage = str(last_stage)
    if last_stage == '-' or last_stage == '' or last_stage == 'nan' or last_stage == 'Not Listed':
        return 'Applied'
    return last_stage
    
cleaned_last_stage_df['Last Stage'] = cleaned_last_stage_df['Last Stage'].apply(lambda last_stage : replace_missing_vals(last_stage))
cleaned_last_stage_df


# In[23]:

# Export cleaned data to excel file
cleaned_applicants_df = cleaned_last_stage_df
cleaned_applicants_df.to_excel('xxx_2018-Quant_Trader_Applicant_Data_Cleaned_3.xlsx')

