

# Enter your code here. Read input from STDIN. Print output to STDOUT

#Data table
# F+1 columns (F features + $/sq.ft ) X H observations

#Input format
#1st line: 2 integers separated by spaces, F (features) and N (training observations)
#Following N lines: F + 1 integers space separated, F features and the house's $/sq.ft
#Next line: T, the # of entries with features but no label (testing observations)
#Following T lines: F integers space separated, F features lacking labels

#F - features
#N - observations, n lines with (F+1 floats space separated)
#T - test observations, t lines (F floats space separated)


#Output format
#Print T lines, the housing price HP for each of the T observations that need them

#Constraints:
# 1 <= F <= 10
# 5 <= N <= 100
# 1 <= T <= 100
# 0 <= HP <= 10^6
# 0 <= Factor Values? <= 1

#------------------------------------------------------------------------------------

#Data Collection
def read_input():
    # Reading the number of observed features (n) and the number of rows/houses (m)
    n, m = map(int, input().split())

    # Reading the observed features and price per square foot for each house
    observed_data = []
    for _ in range(m):
        row = list(map(float, input().split()))
        observed_data.append(row)

    # Reading the number of houses for which pricing is not known (q)
    q = int(input())

    # Reading the features of houses for which pricing is not known
    unknown_data = []
    for _ in range(q):
        row = list(map(float, input().split()))
        unknown_data.append(row)

    return n, m, observed_data, q, unknown_data

#dimensions
n, m, observed_data, q, unknown_data = read_input()
 
labels = [i[-1] for i in observed_data]
observed_data = [lst[:-1] for lst in observed_data]

#This is where data cleaning would be, replacing any missing or outlier data points

#Model Training

class LinearRegression:
    
    def __init__(self, a, e):
        #learning rate, epochs
        self.a = a
        self.e=e
    
    def fit(self,X,Y):
         #training data and labels
        self.X=X
        self.Y=Y
        self.transpose()
        
        #observations, features
        self.m = len(X)
        self.n = len(X[0])
        
        #parameters
        self.W = [0 for x in range(self.n)]
        self.b = 0
        
        for i in range(self.e):
            self.updateWeights()
    
    def updateWeights(self):
        
        dW = 0
        db = 0
        
        #error calc
        errors = []
        for i in range(self.m):
            prediction = self.predict(self.X[i])
            errors.append(self.Y[i]-prediction)
        
        
        dw = [(-2/self.m)*x for x in self.dot(self.XT, errors)]
        db = (-2/self.m)*sum(errors)
            
        for i in range(len(self.W)):
            self.W[i]=self.W[i] - self.a*(dw[i])
        
        self.b = self.b - self.a*(db)
    
    def predict(self, x):
        val = 0
        for i in range(self.n):
            val = val + x[i]*self.W[i]
        val = val + self.b
        return val
    
    def transpose(self):
       self.XT = [[self.X[j][i] for j in range(len(self.X))] for i in range(len(self.X[0]))]
    
    #Dot product, (m,n)*(n,1)=(m,1)
    def dot(self, a, b):
        if len(a[0]) != len(b):
            return None
        
        result = [0] * len(a)
        
        for i in range(len(a)):
            sum = 0
            for j in range(len(a[0])):
                sum += a[i][j]*b[j]
            result[i] = sum
        return result

            
            
        
    

model = LinearRegression(0.01,1000)

model.fit(observed_data, labels)

for i in unknown_data:
    prediction = model.predict(i)
    print(prediction)

#Accuracy: 96.27%