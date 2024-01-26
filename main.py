import pandas as pd
import numpy as np
import tkinter as tk
import time
from tkinter import filedialog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


def preprocessor(X):
    # veriler normalize edilirse algoritma daha verimli calisir
    A = np.copy(X)
    scaler = StandardScaler()
    A = scaler.fit_transform(X)
    return A

class resultWindow(tk.Toplevel):
    def __init__(self, classifierName, time, report, record):
        tk.Toplevel.__init__(self)
        self.title = classifierName
        self.timeLabel = tk.Label(self, text="Execution Time:", fg="#4B0082", font=('courier', 20, 'bold'), pady=10)
        self.time = tk.Label(self, text=time, font=('courier', 20, 'bold'))
        self.recordLabel = tk.Label(self, text="Prediction for selected record:", fg="#4B0082", font=('courier', 18, 'bold'), pady=10)
        self.record = tk.Label(self, text=record, font=('courier', 18, 'bold'))
        self.resultLabel = tk.Label(self, text="Classification Report:", fg="#4B0082", font=('courier', 20, 'bold'), pady=10)
        self.result = tk.Label(self, text=report, font=16, justify="right")
        self.timeLabel.grid(row=0, column=1, sticky='ew')
        self.time.grid(row=0, column=2, sticky='ew')
        self.recordLabel.grid(row=1, column=1, sticky='ew')
        self.record.grid(row=1, column=2, sticky='ew')
        self.resultLabel.grid(row=2, column=1, columnspan=2, sticky='ew')
        self.result.grid(row=3, column=1, columnspan=2, sticky='ew')
        self.grid_columnconfigure((0, 3), weight=1)
        self.minsize(800, 600)
        self.mainloop()


def executePipe(p, X_train, y_train, X_test, y_test, classifierName, record):
    start = time.time()
    print(X_train.shape)
    p.fit(X_train, y_train)
    test_preds = p.predict(X_test)
    end = time.time()
    test_record = p.predict(record)
    ConfusionMatrix = confusion_matrix(y_test, test_preds)
    resultWindow(classifierName, end-start, classification_report(test_preds, y_test), test_record)
    print(classifierName + " algorithm results:")
    print(ConfusionMatrix)
    print(classification_report(test_preds, y_test))
    print()
    print("execute time:")
    print(end-start)


class KnnFrame(tk.LabelFrame):
    def __init__(self, parent):
        tk.LabelFrame.__init__(self, parent, text="KNN config", font=('courier', 16, 'bold'))
        self.distance = tk.StringVar()
        self.distance.set("euclidean")
        self.distances = tk.OptionMenu(self, self.distance, "euclidean", "manhattan", "cosine")
        self.distances.grid(row=1, column=1,columnspan=2, pady=20)
        self.distancesLabel = tk.Label(self, text="Distance Algorithm:", font=14)
        self.distancesLabel.grid(row=1, column=0)
        self.slideLabel = tk.Label(self, text="Neighbors Count:", font=14)
        self.slideLabel.grid(row=0, column=0,padx=100, sticky="S")
        self.neighborCount = tk.Scale(self, from_=0, to=200,length=300, orient=tk.HORIZONTAL)
        self.neighborCount.grid(row=0, column=1,columnspan=2)

class RfcFrame(tk.LabelFrame):
    def __init__(self, parent):
        tk.LabelFrame.__init__(self, parent, text="Random Forest config", font=('courier', 16, 'bold'))
        self.feature = tk.StringVar()
        self.feature.set("sqrt")
        self.features = tk.OptionMenu(self, self.feature, "sqrt", "log2")
        self.features.grid(row=1, column=1,columnspan=2, pady=20)
        self.featureLabel = tk.Label(self, text="max_features:", font=14)
        self.featureLabel.grid(row=1, column=0)
        self.slideLabel = tk.Label(self, text="The number of trees:", font=14)
        self.slideLabel.grid(row=0, column=0,padx=100, sticky="S")
        self.treeCount = tk.Scale(self, from_=0, to=300,length=300, orient=tk.HORIZONTAL)
        self.treeCount.set(100)
        self.treeCount.grid(row=0, column=1,columnspan=2)




class MainApplication(tk.LabelFrame):
    def __init__(self, parent):
        tk.LabelFrame.__init__(self, parent, width=550, height=200, text="Main Settings",font=('courier',20,'bold'))
        self.r = tk.IntVar()
        self.filename =""
        self.chooseFileButton = tk.Button(self, text="Choose File", font=14, command=lambda: self.readFilename())
        self.filenameLabel = tk.Label(self, text="no file chosen", font=14)
        self.recordLabel = tk.Label(self, text="Choose Specific Record", font=14)
        self.record = tk.StringVar()
        self.record.set("risky 1")
        self.records = tk.OptionMenu(self, self.record, "risky 1", "risky2", "healthy 1", "healthy 2")
        self.knnRadio = tk.Radiobutton(self, text='KNeighbors Classifier', font=14, variable=self.r, value=1)
        self.RfcRadio = tk.Radiobutton(self, text='RandomForest Classifier', font=14, variable=self.r, value=2)
        self.NbRadio = tk.Radiobutton(self, text='GaussianNB Classifier', font=14, variable=self.r, value=3)
        self.testSlider = tk.Scale(self, from_=0, to=1,resolution=0.05, length=300, orient=tk.HORIZONTAL)
        self.testSlider.set(0.2)
        self.testLabel = tk.Label(self, text="Test/Train Ratio:", font=14)
        self.ready = tk.Button(self, text="Start", font=14, fg='orange', bg='black', command=lambda: self.runnable())
        self.knnSettings = KnnFrame(parent)
        self.rfcSettings = RfcFrame(parent)
        self.locateWidgets(parent)

    def locateWidgets(self, parent):
        self.chooseFileButton.grid(row=0,column=0, pady=20)
        self.filenameLabel.grid(row=0,column=1, columnspan=2)
        self.knnRadio.grid(row=1,column=0, sticky='ew')
        self.RfcRadio.grid(row=1,column=1, sticky='ew')
        self.NbRadio.grid(row=1,column=2, sticky='ew')
        self.testSlider.grid(row=2,column=1, columnspan=2)
        self.testLabel.grid(row=2, column=0)
        self.recordLabel.grid(row=3, column=0)
        self.records.grid(row=3, column=1, columnspan=2)
        self.ready.grid(row=4, column=2, pady=10)
        self.grid(row=0, column=1, sticky='ew')
        self.knnSettings.grid(row=1, column=1, pady=50, sticky='ew')
        self.rfcSettings.grid(row=2, column=1, pady=10, sticky='ew')

    def readFilename(self):
        self.filename = filedialog.askopenfilename(initialdir="C:/<whatever>", title="Select a file")
        self.filenameLabel.config(text=self.filename)
        print(self.knnSettings.distance.get())
        print(type(self.knnSettings.distance.get()))

    def runnable(self):
        # Data seti okuma
        dataset = pd.read_csv(self.filenameLabel.cget("text"))
        # Pipeline icin FunctionTransformer
        preprocess_transformer = FunctionTransformer(preprocessor)
        # Blood Presure Low ve High Kismi birlikte verilmis
        # bu degerleri kullanabilmak icin iki ayri colluma bolunmeli
        dataset[['BP_Systolic', 'BP_Diastolic']] = dataset['Blood Pressure'].str.split('/', expand=True)
        dataset['BP_Systolic'] = pd.to_numeric(dataset['BP_Systolic'])
        dataset['BP_Diastolic'] = pd.to_numeric(dataset['BP_Diastolic'])
        # Eski Coll silinir
        dataset = dataset.drop("Blood Pressure", axis=1)
        # Country gibi string elemanlari encode ederek sayiya donusturulmeli
        encoder = LabelEncoder()
        for col_name in dataset.columns:
            if dataset[col_name].dtype == "object":
                # print(dataset[[col_name]].shape)
                dataset[col_name] = encoder.fit_transform(dataset[[col_name]])
        # X <- veriler y<- labellar
        X = dataset.drop(columns=['Patient ID', 'Heart Attack Risk'])
        y = dataset['Heart Attack Risk']
        # Veri setini train ve test olarak bolme
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.testSlider.get(), random_state=0)
        KNNpipe = Pipeline([('Scaler', preprocess_transformer),
                            ('KNeighbors Classifier',
                             KNeighborsClassifier(n_neighbors=self.knnSettings.neighborCount.get(),
                                                  metric=self.knnSettings.distance.get()))])

        RandomForestPipe = Pipeline([('Scaler', preprocess_transformer),
                                     ('RandomForest Classifier',
                                      RandomForestClassifier(n_estimators=self.rfcSettings.treeCount.get(),
                                                             max_features=self.rfcSettings.feature.get()))])
        NaivePipe = Pipeline([('Scaler', preprocess_transformer),
                              ('Naive Bayesian Classifier', GaussianNB())])
        selectedRecord = pd.read_csv('riskli1.csv', index_col=0)
        print(selectedRecord.head)
        if self.recordLabel == "risky 1":
            selectedRecord = pd.read_csv('riskli1.csv', index_col=0)
        elif self.recordLabel == "risky2":
            selectedRecord = pd.read_csv('riskli2.csv', index_col=0)
        elif self.recordLabel == "healthy 1":
            selectedRecord = pd.read_csv('risksiz1.csv', index_col=0)
        elif self.recordLabel == "healthy 2":
            selectedRecord = pd.read_csv('risksiz2.csv', index_col=0)

        if self.r.get() == 1:
            executePipe(KNNpipe, X_train, y_train, X_test, y_test, "KNN", selectedRecord)
        elif self.r.get() == 2:
            print(X_train.head)
            executePipe(RandomForestPipe, X_train, y_train, X_test, y_test, "Random Forest", selectedRecord)
        elif self.r.get() == 3:
            executePipe(NaivePipe, X_train, y_train, X_test, y_test, "Naive Bayesian", selectedRecord)

def main():
    root = tk.Tk()
    root.title('Heart Attack Risk Classification')
    root.minsize(900, 700)
    root.grid_columnconfigure((0, 2), weight=1)
    app = MainApplication(root)
    root.mainloop()


if __name__ == '__main__':
    main()

