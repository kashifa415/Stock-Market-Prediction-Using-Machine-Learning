import tkinter 
from tkinter import Label, Button 
from tkinter import messagebox 
import numpy as np 
from sklearn import model_selection, neighbors 
from sklearn.model_selection import train_test_split 
import pandas as pd 
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC
from sklearn import metrics 
import matplotlib.pyplot as plt

class stocks_prd():
    def rules(self):
        a = """# Attribute Domain
        -----------------------------------------
        1. Date	 27/03/2000 - 08/09/2017
        2. Open	      2.00 - 120.00
        3. High     2.00 - 120.00
        4. Low	     2.00 - 120.00
        5. Close	   2.00 - 120.00
        6. Adj_Close	2.00 - 120.00
        7. Volume	   50000 - 50000000

        """
        messagebox.showinfo("Guidelines for parameters are as follows:-",a)
    def quit_window(self):
        if messagebox.askokcancel("Quit", "Are you sure! You want to quit now? "):
           root.destroy()
           
    def print_predictd(self):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        lr=LinearRegression() 
        lr.fit(X_train, y_train) 
        p= lr.predict(X_test) 
        print("p=",p)

        #lr.fit(X_train, y_train)
        #accuracy = lr.score(X_test, y_test) 
        #print("Accuracy=",accuracy)

        clf_entropy = DecisionTreeClassifier( criterion = "entropy", random_state = 100, max_depth = 3, 
        min_samples_leaf = 5)

        clf_entropy.fit(X_train, y_train) 
        e= clf_entropy.predict(X_test) 
        print("e=",e)

  #Fitting SVM to the Training set
        classifier = SVC(kernel = 'rbf', random_state = 0)
        classifier.fit(X_train, y_train) 
        pred = classifier.predict(X_test)

        print(metrics.mean_absolute_error(y_test,p))
        print(metrics.mean_squared_error(y_test,p))
        print(np.sqrt(metrics.mean_squared_error(y_test,p)))
   
        print("predict=",p) 
        print("entropy=",e) 
        print("prediction=",pred)

    def actionPerformed(self):

      plt.subplot(221) 
      plt.title("stock market")
      plt.plot(y_test,p)

      plt.subplot(222) 
      plt.plot(y_test,e)

      plt.subplot(223) 
      plt.plot(y_test,pred)

      plt.subplot(224)
      plt.plot(y_test,pred)

      plt.show()

    def __init__(self, root):

        self.label = Label(root, text='Stock Market Prediction', font=('arial', 30, 'bold'), bg='AZURE',fg='black') 
        self.label.pack()

        self.button1 = Button(root, width=16, text='SHOW DATA',bg="GREY", command=self.actionPerformed, font=('arial', 12), bd=3) 
        self.button1.place(x=30, y=400)

        self.button2 = Button(root, width=6, text='QUIT',bg="GREY", command=self.quit_window, font=('arial', 12), bd=3) 
        self.button2.place(x=30, y=500)

        predt_button = Button(root, text="CHECK PREDICTION",bg="GREY",command = self.print_predictd, font=('arial', 12),bd=3) 
        predt_button.place(x=30, y=300)
    
        acc_button = Button(root, text="RULES", bg="GREY", font=('arial', 10), command=self.rules,bd=3) 
        acc_button.place(x=30, y=100)

        root.mainloop()


df = pd.read_csv('stock.csv')
df.replace('#', -99999, inplace=True) 
print(df.head)
df= df.to_numpy() 
X= df[:,[2]] 
y= df[:,-1]
y=y.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr=LinearRegression()
lr.fit(X_train, y_train)
p= lr.predict(X_test)
print("p=",p)

lr.fit(X_train, y_train)
accuracy = lr.score(X_test, y_test)
print("Accuracy=",accuracy)

clf_entropy = DecisionTreeClassifier( criterion = "entropy", random_state = 100, max_depth = 3,
min_samples_leaf = 5)
clf_entropy.fit(X_train, y_train)
e= clf_entropy.predict(X_test)
print("e=",e)

 #Fitting SVM to the Training set
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)

print(metrics.mean_absolute_error(y_test,p))
print(metrics.mean_squared_error(y_test,p))
print(np.sqrt(metrics.mean_squared_error(y_test,p)))

print("predict=",p)
print("entropy=",e)
print("prediction=",pred)

def print_acc():

   accuracy_2 = float(accuracy)
   panel_acc = Label(root, text=float(accuracy_2),bg="AZURE", font=('arial', 10, 'bold'), width=20, height=2)
   panel_acc.place(x=180, y=550) 
   plt.figure(2)

   
def print_predictd():
   predict = float(p)
   panel_acc = Label(root, text=float(predict),bg="AZURE", font=('arial', 10, 'bold'), width=20, height=2)
   panel_acc.place(x=90, y=350)

root = tkinter.Tk() 
# root.title("BERZA")
path1 =(r'C:/Users/dell/Downloads/python-logo.png') 
root.iconbitmap(path1) 
root.minsize(600,600) 
root.config(bg="lightseagreen")

predt_button = Button(root, text="CHECK PREDICTION",bg="GREY", font=('arial', 10), command = print_predictd)
predt_button.place(x=20, y=500)

acc_button = Button(root, text="CHECK ACCURACY",bg="GREY", font=('arial', 10), command = print_acc)
acc_button.place(x=30, y=200)

m = stocks_prd(root) 
root.mainloop()
 
 
