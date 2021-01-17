import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random 
import time
#MAKE SURE THAT THE DATASET.CSV IS IN THE SAME DIRECTORY AS THIS PY. FILE
# Change for the directory here
course_data_path = 'dataset.csv'

# Defining the seeds that are for data samples (those particular seeds were used for our report /results)
data_10_seed = 777
data_50_seed = 678
data_100_seed = 4567

def data_importing(data_path):
    
    # Importing the data from here
    data = pd.read_csv(data_path)  
    
    # Change index to machine name (M1, M2, ... M60)
    data.index = ['M' + str(i+1) for i in range(0, len(data.index))]

    return data

data = data_importing(course_data_path)

#Start of Question 1
# Calculate the total processing time for each job over all machines
sum_processing_time = data.sum(axis=0)

total_time = sum_processing_time

pd_processing_time = pd.DataFrame(columns = ['Job', 'Time']) 

pd_processing_time['Job'] = data.columns

for i in range(0,len(data.columns)):
    pd_processing_time['Time'][i]=total_time[i]
    pd_processing_time.set_index('Job') # resetting the index  

# Observe the outliers from plots
# Change columns' names
time_df = sum_processing_time.to_frame()
time_df['job'] = time_df.index
time_df.rename(columns = {0:'total_time'}, inplace = True)

# Scatterplot before removing outliers
def scatter(x, y):

    ax=plt.axes()
    plt.scatter(x, y, alpha = 0.5)
    ax.set_title("Total Processing Time Per Job")
    
    plt.xticks(np.arange(0, 850, 100), np.arange(0, 850, 100))
    
    plt.xlabel('Job')
    plt.ylabel('Total processing time ')

    plt.show()

scatter(time_df['job'], time_df['total_time'])

# Observe the outliers from boxplot
fig = plt.figure(1,figsize=(5,5))
ax = fig.add_subplot(1,1,1)
ax.set_title("Total Processing Time Per Job")
ax.set_xlabel('Jobs')
ax.set_ylabel('Total Processing Time')
bp = ax.boxplot(sum_processing_time, flierprops=dict(markeredgecolor='red'))

# Detect outliers using 1.5*IQR rule
def outlier(df_column):
    sorted(df_column)
    Q1,Q3 = np.percentile(df_column, [25,75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range,upper_range

lower_range, upper_range = outlier(sum_processing_time)

# Outliers to be removed, according to 1.5*IQR rule
outliers = time_df[(sum_processing_time < lower_range) | (sum_processing_time > upper_range)]

# Removing the outliers  
outliers_jobs=list(outliers.index)
data.drop(outliers_jobs,axis=1,inplace=True)

# Observe the missing value with heatmap
def heatmap(data):
    ax = plt.axes()
    ax.set_title("Total Missing Values Across Machines and Jobs")
    plt.xlabel("Job")
    plt.ylabel("Machine")
    return sns.heatmap(data.isnull())
# calling the heatmap function
heatmap(data)
    
# Handling missing data
def na_check(data):
    print("Is there any missing value in data?:", data.isnull().values.any(),",", data.isnull().sum().sum(),"missing variables")
    if data.isnull().values.any():
        # Check the machines which contain missing values
        df_na_col_list = data[data.isna().any(axis = 1)].index.tolist()
        print("List of machines which contain missing values:\n ", df_na_col_list)
        
na_check(data)

# Replace the missing data with the mean of each job
data = data.fillna(data.mean())
na_check(data)

# Check the missing value with heatmap again
heatmap(data)

# Removing remaining inconsistencies
def no_inconsistencies(data):    
    #1. Convert integer to float and round the number for all values in dataset  
    data = data.astype(float).round(2)
    #2. Change all negative value to absolute value
    data = data.abs()
    
    return data

data = no_inconsistencies(data)

# End of Q1
# Start of Q2
# Creating smaller datasets
def data_n(data, n, seed=0):
    new_data = data.sample(n = n, axis = 1, random_state = seed) #random_state acts as random.seed
    return new_data 
    
data_10 = data_n(data, 10, data_10_seed)
data_50 = data_n(data, 50, data_50_seed)
data_100 = data_n(data, 100, data_100_seed)

# End of Q2

# Start of Q3 Random Search

# Makespan function in evaluation.py file 
def makespan(s, P):
    C = P[s, :]
    n, m = C.shape
    C[0, :] = np.cumsum(C[0, :])
    C[:, 0] = np.cumsum(C[:, 0])

    for i in range(1, n):
        for j in range(1, m):
            C[i, j] += np.maximum(C[i - 1, j], C[i, j - 1])
    return C[-1, -1]

# Implement random search algorithm
def random_search(data):
    start_time = time.time()
    col_count = len(data.columns)
    max_iterations = col_count*1
    # Giving a largest number 
    best_time = random.getrandbits(128)
    best_seq = []

# Iterate until finding the best sequence
    while max_iterations > 0:
        max_iterations = max_iterations - 1
        P0 = np.transpose(np.array(data))
        s0 = np.random.permutation(col_count)

        new_time = makespan(s0, P0)
        
        if new_time < best_time:
                best_time = new_time
                best_seq = s0
    
    # Calculating the time for the algorithm 
    time_cost = time.time() - start_time
    
    return best_seq, best_time, round(time_cost, 2)



# Implement descriptive analytics for random search
def descriptive_analytics_random(data):
    random_search_analytics=pd.DataFrame(columns=['seed','sequences','corresponding makespan', 'Time'])
    a=0

# Finding the best sequence for each of the 30 seeds from 0-29
    while a <30:
        np.random.seed(a)
        random_search_analytics.loc[a,'seed']=a
        (best_seq, best_time, time_cost) = random_search(data)
        best_seq=list(best_seq)
        for i, v in enumerate(best_seq):
            best_seq[i] = data.columns[v]

        random_search_analytics.loc[a,'sequences']=best_seq
        random_search_analytics.loc[a,'corresponding makespan']=best_time
        random_search_analytics.loc[a,'Time'] = time_cost
        a += 1
        best_of_all=random_search_analytics['corresponding makespan'].min()

    print("the mean is", 
          random_search_analytics['corresponding makespan'].mean(),"\n","the max is ", 
          random_search_analytics['corresponding makespan'].max(),"\n","the min is ",
          random_search_analytics['corresponding makespan'].min(),"\n","the std is ",
          random_search_analytics['corresponding makespan'].std())
    best_index=random_search_analytics[random_search_analytics['corresponding makespan']==best_of_all].index.values
    print(random_search_analytics.loc[best_index,['sequences']])

    return random_search_analytics

#These codes are gonna return a dataframe of the best makespans(random_search) for 30 seeds and print its statistics 
descriptive_analytics_random(data_10)
descriptive_analytics_random(data_50)
descriptive_analytics_random(data_100)
# End of Q3

# Start of Q4, Genetic Algorithm 
#generate population for the GA
def generate_pop(data,P):
    global Population
    # NumPy matrix of shape (number of jobs, number of machines) 
    jobs_matrix=np.transpose(np.array(data))
    #random possible sequence
    s=np.random.permutation(jobs_matrix.shape[0])
    
    # This loop permutess 30 different times and calculates the makespan for each sequence
    # Store our population in a dataframe with two columns, the sequence and its corresponding makespan
    Population = pd.DataFrame(index=range(0,P), columns=['Sequences','Makespans'])
    for i in range(0,P):    
        a=np.random.permutation(s)
        Population.iloc[i,0]=a
        Population.iloc[i,1]=makespan(a,jobs_matrix)
    
    return Population

# Creating the function that gives us the description of the population
def Population_statistics(Population):
    return Population['Makespans'].astype(float).describe()

def Parent_Selection(Population,P):
# Creating the probability distribution for choosing the first parent
    pb=[]
    for i in range(0,P):
        pb.append((2*i)/((P-1)*P))
        
# Selecting the parents by assigning them to the corrresponding index
    index_parent_1=np.random.choice(np.arange(0,P),p=pb)
    P1=Population.loc[index_parent_1,['Sequences']]
    Parent_1 = P1[0]
# Selecting the second parent using the uniform distribution
    index_parent_2=np.random.choice(np.arange(0,P))
    P2=Population.loc[index_parent_2,['Sequences']]
    Parent_2 = P2[0]
    
    return Parent_1,Parent_2

# Creating the crossover function
# Takes the two parents selected by the selection function and the crossover probability as parameters
# b is a certain random decimal between 0 and 1 that is generated to check when the crossover would be done 
def crossover(data,Parent_1,Parent_2,Pc,b):
    if b <= Pc:
        array=np.random.choice(range(len(data.iloc[0])-2),2,replace=False)
        sorted_array=np.sort(array)
        first_crossover_point = sorted_array[0]
        second_crossover_point = sorted_array[1]
        
# Divide each parent into 3 blocks so we can work on each block seperatly
# block 1 - before first crossover point
# block 2 - between two crossover points
# block 3 - after second cross over
        block1_1 = Parent_1[0:first_crossover_point+1]
        block2_1 = Parent_1[first_crossover_point+1:second_crossover_point+1]
        block3_1 = Parent_1[second_crossover_point+1:]
        block1_2 = Parent_2[0:first_crossover_point+1]
        block2_2 = Parent_2[first_crossover_point+1:second_crossover_point+1]
        block3_2 = Parent_2[second_crossover_point+1:]

# Creating an egg which will be the children before the mutaion
# for loops that assign blocks to an egg
        egg_1 = []
        egg_1.extend(block1_1)
        for _ in range(0, len(block2_1)):
            egg_1.append(np.nan)
        for i in block3_2:
            if i not in block1_1:
                egg_1.append(i)
                
# Checking for illegitimate swapping
            else:
                egg_1.append(np.nan)
        illegitimate_array = []
        for i in range(0,len(data.iloc[0])):
            if i not in egg_1:
                illegitimate_array.append(i)
                
# Shuffling the array randomly
        np.random.shuffle(illegitimate_array)

        egg_1 = np.array(egg_1)
        egg_1[np.isnan(egg_1)] = illegitimate_array
        egg_1=egg_1.astype(int)
        
# The same process is done for the second egg 
        egg_2 = []
        egg_2.extend(block1_2)
        for _ in range(0, len(block2_2)):
            egg_2.append(np.nan)
            
        for i in block3_1:
            if i not in block1_2:
                egg_2.append(i)
                
            else:
                egg_2.append(np.nan)
        illegitimate_array = []
        for i in range(0,len(data.iloc[0])):
            if i not in egg_2:
                illegitimate_array.append(i)
                
        np.random.shuffle(illegitimate_array)
        
        egg_2 = np.array(egg_2)
        egg_2[np.isnan(egg_2)] = illegitimate_array
        egg_2=egg_2.astype(int)

# If crossover doesn't happen then the eggs will become the parents and then mutation will happen
    else:
        egg_1= Parent_1 
        egg_2=Parent_2
    return egg_1,egg_2

# Creating the mutation function
# Mutation exchange mutation, simple exchange of two elements choosen at random
def mutation(data,egg_1,egg_2,Pm,D,b):
        
    array=np.random.choice(range(len(data.iloc[0])),2,replace=False)
    sorted_array=np.sort(array)
    first_index = sorted_array[0]
    second_index = sorted_array[1]

    if b<=Pm:
        egg_1[second_index],egg_1[first_index]=egg_1[first_index],egg_1[second_index]
        egg_2[second_index],egg_2[first_index]=egg_2[first_index],egg_2[second_index]

    child_1=egg_1
    child_2=egg_2
    
    return child_1,child_2

# Function replacing the old sequence (unfit member) with the children
def replace(data,Population,child_1,child_2):
    
    jobs_matrix=np.transpose(np.array(data))
    
    # Replacing the old sequence(unfit member ) with the childs if the childs have a better makespan
    if makespan(child_1,jobs_matrix)<= makespan(Population.iloc[0,0],jobs_matrix):
        Population.iloc[0,0]=child_1
        Population.iloc[0,1]=makespan(child_1,jobs_matrix)
        
        if makespan(child_2,jobs_matrix)<= makespan(Population.iloc[1,0],jobs_matrix):
            Population.iloc[1,0]=child_2
            Population.iloc[1,1]=makespan(child_2,jobs_matrix)
    
    elif makespan(child_2,jobs_matrix)<= makespan(Population.iloc[0,0],jobs_matrix):
        Population.iloc[0,0]=child_2
        Population.iloc[0,1]=makespan(child_2,jobs_matrix)

# Genetic Algorithm Function
def GA(data,seed,P,Pc,Pm,D):
    #Parameter for decreasing the mutation probability each iteration
    omega=0.99
    
    start_time=time.time()
    # Creating the population
    np.random.seed(seed)
    Population=generate_pop(data,P)
    Pm_temp=Pm
    
    # Number of iteration (changes for dataset N = 10 for data_100, N = 100 for data_50 and N=1000 for data_10)
    for i in range(1*data.shape[1]):
        
        np.random.seed(i)
        
        b=np.random.random()
        
        # Sorting makespan in descending order and replace unfit member with new children
        Population.sort_values(by='Makespans',ascending=False,inplace=True,ignore_index=True)
        
        Parent_1,Parent_2=Parent_Selection(Population,P)
        
        egg_1, egg_2 = crossover(data,Parent_1, Parent_2,Pc,b)
        
        child_1, child_2 = mutation(data,egg_1, egg_2,Pm,D,b)
        
        replace(data,Population,child_1,child_2)
        
        # Reset factor for mutation rate
        if b <= Pm:
            Pm=Pm*omega
            
        Vmin=Population['Makespans'].min()
        Vmean=Population['Makespans'].mean()
        #Resetting the probability of mutation if threshold is achieved
        if  Vmin/Vmean > D:
              Pm =Pm_temp 
              
        Population.sort_values(by='Makespans',ascending=False,inplace=True,ignore_index=True)
    

    #Finding the best sequnce 
    last_statistics=Population['Makespans'].astype(float).describe()
    best_sequence=Population.iloc[-1,0]
    best_makespan=Population.iloc[-1,1]
    finish_time=time.time()
    best_sequence=list(best_sequence)
    
    for i,v in enumerate(best_sequence):
        best_sequence[i]=data.columns[v]
        
    best_sequence=np.array(best_sequence)
    Time_of_alg=finish_time-start_time

    return best_sequence,best_makespan,last_statistics,Time_of_alg


# Function to find out descriptive statistics for the genetic algorithm
def descriptive_analytics_GA(data,P,Pc,Pm,D):
    
    data_statistics=pd.DataFrame(columns=['seed','best sequences','corresponding makespan','Time'])
    a=0
    while a <30:
        (best_sequences, best_makespan,last_statistics,Time_for_alg)=GA(data,a,P,Pc,Pm,D)
        data_statistics.loc[a,'seed']=a
        data_statistics.loc[a,'best sequences']=(np.array2string(best_sequences,separator=','))
        data_statistics.loc[a,'corresponding makespan']=best_makespan
        data_statistics.loc[a,'Time']=Time_for_alg
        a += 1
   
    print(data_statistics['corresponding makespan'].astype(float).describe())
    print(data_statistics['Time'].sum())
    print("Results for GA with", "Dataset:",len(data.columns) , "\nPopulation size: ", P , "\nCrossover probability ", Pc, "\nMutation probability ", Pm , "\nReset factor D ", D)
    return data_statistics

descriptive_analytics_GA(data=data_10, P=30, Pc=1, Pm=0.8, D=0.95)
descriptive_analytics_GA(data=data_50, P=30, Pc=1, Pm=0.8, D=0.95)
descriptive_analytics_GA(data=data_100, P=30, Pc=1, Pm=0.8, D=0.95)

#End of Q4
#Start of Q5
# Function to create excel sheets so when we split the code for each member to run individually it will  save time and recollect the data from each member(rather than doing it on one ＰＣ)
def analytic(method, data, P=30, Pc=1, Pm=0.8, D=0.95):
    
    if method == "GA":
        # We detect which parameter is a list, returning 3 dataframes of 3 different parameter values,while others are still at default values
        
        if type(P) == list:
            for i in P:
                output_data = descriptive_analytics_GA(data, i, Pc, Pm, D)
                
                # This will generate an excel file located in the same directory as this python code
                filename = "GA_{}_P_{}".format(len(data.columns), i)
                output_data.to_excel( "{}.xlsx".format(filename))
        elif type(Pc) == list:
            for i in Pc:
                output_data = descriptive_analytics_GA(data, P, i, Pm, D)
                filename = "GA_{}_Pc_{}".format(len(data.columns), i)
                output_data.to_excel( "{}.xlsx".format(filename))
        elif type(Pm) == list:
            for i in Pm:
                output_data = descriptive_analytics_GA(data, P, Pc, i, D)
                filename = "GA_{}_Pm_{}".format(len(data.columns), i)
                output_data.to_excel( "{}.xlsx".format(filename))
        else:
            # generate GA analytics with default parameters
            output_data = descriptive_analytics_GA(data, P, Pc, Pm, D)
            filename = "GA_{}_default".format(len(data.columns))
            output_data.to_excel( "{}.xlsx".format(filename))
            
    #We also added the collection of random search to draw the graph comparing it to genetic algorithm
    elif method == "RS":
        output_data = descriptive_analytics_random(data)
        output_data.to_excel("RS_{}_default.xlsx".format(len(data.columns)))
                
        
# Import back the excel results after running the code separately to visualize results
def data_collection(method, data, P=30, Pm=0.8, Pc=1):
    
    if method == "GA":
        #Creating an empty list that will be filled with the results of different paramters
        pd_list = []
        
        if type(P) == list:
            for i in P:
                file_name = "GA_{}_P_{}.xlsx".format(len(data.columns), i)
                file = pd.read_excel(file_name)
                pd_list.append(file)
                
        elif type(Pc) == list:
            for i in Pc:
                file_name = "GA_{}_Pc_{}.xlsx".format(len(data.columns), i)
                file = pd.read_excel(file_name)
                pd_list.append(file)
                
        elif type(Pm) == list:
            for i in Pm:
                file_name = "GA_{}_Pm_{}.xlsx".format(len(data.columns), i)
                file = pd.read_excel(file_name)
                pd_list.append(file)  
        else:
            file_name = "GA_{}_default.xlsx".format(len(data.columns))
            file = pd.read_excel(file_name)
            return file
                
        return pd_list
    
    else:
        # if the method is GA
        rs = pd.read_excel("RS_{}_default.xlsx".format(len(data.columns)))
        return rs

# Running the algorithm and get analytics from here
analytic(method="GA",data=data_10,Pc=[0,0.7,0.9])
analytic(method="GA",data=data_10,P=[10, 50, 100])
analytic(method="GA",data=data_10,Pm=[0.05, 0.6, 0.8])

analytic(method="GA",data=data_50,Pc=[0, 0.7, 0.9])
analytic(method="GA",data=data_50,P=[10, 50, 100])
analytic(method="GA",data=data_50,Pm=[0.05, 0.6, 0.8])

analytic(method="GA",data=data_100,Pm=[0.05, 0.6, 0.8])
analytic(method="GA",data=data_100,P=[10, 50, 100])
analytic(method="GA",data=data_100,Pc=[0, 0.7, 0.9])

# Analysis for default parameters: P=30 pm=0.8 pc=1 D=0.95

analytic(method="GA", data=data_10)
analytic(method="GA", data=data_50)
analytic(method="GA", data=data_100)

analytic(method="RS", data=data_10) 
analytic(method="RS", data=data_50)
analytic(method="RS", data=data_100)

# Get the data from excel files and stored in local variables
P_list_10 = data_collection(method="GA",data=data_10, P=[10, 50, 100])
Pc_list_10 = data_collection(method="GA",data=data_10, Pc=[0, 0.7, 0.9])
Pm_list_10 = data_collection(method="GA",data=data_10, Pm=[0.05, 0.6, 0.8])

P_list_50 = data_collection(method="GA",data=data_50,P=[10,50,100])
Pm_list_50 = data_collection(method="GA",data=data_50, Pm=[0.05, 0.6, 0.8])
Pc_list_50 = data_collection(method="GA",data=data_50, Pc=[0, 0.7, 0.9])

P_list_100 = data_collection(method="GA",data=data_100,P=[10,50,100])
Pc_list_100 = data_collection(method="GA", data=data_100, Pc=[0, 0.7, 0.9])
Pm_list_100 = data_collection(method="GA", data=data_100, Pm=[0.05, 0.6, 0.8])

GA_10 = data_collection(method="GA", data=data_10)
GA_50 = data_collection(method="GA", data=data_50)
GA_100 = data_collection(method="GA", data=data_100)

RS_10 = data_collection(method="RS", data=data_10)
RS_50 = data_collection(method="RS", data=data_50)
RS_100 = data_collection(method="RS", data=data_100)


# this function will crate different boxplot comparing the different possible scenarios.
def boxplot(change_par, P, P_list, data, yMin, yMax):
    df_comparison= pd.concat([P_list[0]['corresponding makespan'].rename('{}={}'.format(change_par,P[0])),
                            P_list[1]['corresponding makespan'].rename('{}={}'.format(change_par, P[1])),
                            P_list[2]['corresponding makespan'].rename('{}={}'.format(change_par, P[2]))],
                            axis=1)
    axes = df_comparison.boxplot(color=dict( medians='r'),showfliers=False, grid=False,return_type='axes')  
    axes.set_ylim(yMin, yMax)
    

boxplot("P", [10, 50, 100], P_list_10, data_10,4200, 4260)

boxplot("Pc", [0, 0.7, 0.9], Pc_list_10, data_10,4200, 4350)
boxplot("Pm", [0.05, 0.6, 0.8], Pm_list_10,data_10,4200, 4260)

boxplot("Pm", [0.05, 0.6, 0.8], Pm_list_50,data_50,6900, 7300)
boxplot("Pc", [0, 0.7, 0.9], Pm_list_50,data_50,6900, 7300)
boxplot("P", [10, 50, 100], Pm_list_50,data_50,6900, 7300)

boxplot("P", [10, 50, 100], Pm_list_100,data_100,10400, 10800)
boxplot("Pc", [0, 0.7, 0.9], Pm_list_100,data_100, 10400, 10800)
boxplot("Pm", [0.05, 0.6, 0.8], Pm_list_100,data_100,10400, 10800)

boxplot("P", [10, 50, 100], P_list_10,data_10,4200, 4260)
boxplot("Pc", [0, 0.7, 0.9], Pc_list_10, data_10,4200, 4350)
boxplot("Pm", [0.05, 0.6, 0.8], Pm_list_10,data_10,4200, 4260)

boxplot("Pm", [0.05, 0.6, 0.8], Pm_list_50,data_50,6900, 7300)
boxplot("Pc", [0, 0.7, 0.9], Pm_list_50,data_50,6900, 7300)
boxplot("P", [10, 50, 100], Pm_list_50,data_50,6900, 7300)

boxplot("P", [10, 50, 100], Pm_list_100,data_100,10400, 10800)
boxplot("Pc", [0, 0.7, 0.9], Pm_list_100,data_100, 10400, 10800)
boxplot("Pm", [0.05, 0.6, 0.8], Pm_list_100,data_100,10400, 10800)



df_time_10 = pd.DataFrame()
df_time_10['RS_10'] = RS_10['Time'].round(2)
df_time_10['GA_10'] = GA_10['Time'].round(2)
df_time_10.plot.line()


df_time_50 = pd.DataFrame()
df_time_50['RS_50'] = RS_50['Time'].round(2)
df_time_50['GA_50'] = GA_50['Time'].round(2)
df_time_50.plot.line()


df_time_100 = pd.DataFrame()
df_time_100['RS_100'] = RS_100['Time'].round(2)
df_time_100['GA_100'] = GA_100['Time'].round(2)
df_time_100.plot.line()

analytic("GA", data,P=100, Pc=0.95, Pm=0.05, D=0.95)




