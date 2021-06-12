from tkinter import filedialog, messagebox
from tkinter import *
import pandas as pd
import os
from preprocessing import PreProcessing
from NaiveBayes import NaiveBayes
from sklearn.metrics import accuracy_score


class GUI:
    def __init__(self):
        self.master = Tk()
        self.master.title("Naïve Bayes Classifier")
        self.master.geometry("500x250")

        self.dir = ''
        self.text_entry = StringVar()
        # path entry
        self.text = Entry(self.master, textvariable=self.text_entry, width=50)
        self.dir_label = Label(self.master, text="Directory Path")
        self.good_dir = False

        self.browse_button = Button(self.master, text='Browse', command=lambda: self.browse())

        self.num_of_bins = 0
        self.bins_label = Label(self.master, text="Discretization Bins:")
        vcmd = self.master.register(self.validate)  # we have to wrap the command
        self.bins_entry = Entry(self.master, validate="key", validatecommand=(vcmd, '%P'))
        self.good_bins = False

        self.build_button = Button(self.master, text='Build', state=DISABLED, command=lambda: self.build(), width=25)
        self.classify_button = Button(self.master, text='Classify', state=DISABLED, command=lambda: self.classify(), width=25)

        # layout
        self.browse_button.grid(row=0, column=2, padx=20, pady=2)
        self.text.grid(row=0, column=1, pady=2)
        self.dir_label.grid(row=0, column=0, sticky=W, pady=2)
        self.bins_label.grid(row=1, column=0, sticky=W, pady=2)
        self.bins_entry.grid(row=1, column=1, sticky=W)
        self.build_button.grid(row=3, column=1, pady=15)
        self.classify_button.grid(row=4, column=1)

        self.master.mainloop()

    def browse(self):
        self.dir = filedialog.askdirectory()
        if self.check_files():
            self.text_entry.set(self.dir)
            self.good_dir = True
        else:
            self.dir = ''
            self.good_dir = False

        self.check_switch()

    def validate(self, new_text):
        if not new_text:  # the field is being cleared
            self.num_of_bins = 0
            self.good_bins = False
            self.check_switch()
            return True

        try:
            self.num_of_bins = int(new_text)
            if self.num_of_bins >= 2:
                self.good_bins = True
            else:
                self.good_bins = False
            self.check_switch()
            return True
        except ValueError:
            # print(self.num_of_bins)
            return False

    def build(self):
        try:
            self.struct = self.read_struct()
            self.preprocessing_obj = PreProcessing(self.num_of_bins, self.struct)
            train_df = pd.read_csv(os.path.join(self.dir, 'train.csv'))
            train_df = self.preprocessing_obj.fill_missing(train_df)
            train_df = self.preprocessing_obj.discretize(train_df)

            self.model = NaiveBayes(train_df)
            messagebox.showinfo(title="Naïve Bayes Classifier", message="Building classifier using train-set is done!")
            self.classify_button['state'] = 'normal'
        except Exception as e:
            messagebox.showerror(title="Naïve Bayes Classifier", message=e)

    def check_files(self):
        try:
            df_train = pd.read_csv(os.path.join(self.dir, 'train.csv'))
            df_test = pd.read_csv(os.path.join(self.dir, 'test.csv'))
            if not os.path.isfile(os.path.join(self.dir, 'Structure.txt')):
                raise Exception()
            return True
        except:
            messagebox.showerror(title="Naïve Bayes Classifier", message='Missing files in folder')
            return False

    def classify(self):
        try:
            df_test = pd.read_csv(os.path.join(self.dir, 'test.csv'))
            df_test = self.preprocessing_obj.fill_missing(df_test)
            df_test = self.preprocessing_obj.discretize(df_test)
            predictions = self.model.predict(df_test)
            predictions = self.preprocessing_obj.inverse_pred(predictions)
            self.write_output(predictions)
            messagebox.showinfo(title="Naïve Bayes Classifier", message="Classification done successfully!")
            self.master.destroy()
        except Exception as e:
            messagebox.showerror(title="Naïve Bayes Classifier", message=e)

    def read_struct(self):
        path = os.path.join(self.dir, 'Structure.txt')
        f = open(path, "r")
        struct = {}
        for line in f:
            splited = line.split(' ', 2)
            if 'NUMERIC' in splited[2]:
                struct[splited[1]] = 'NUMERIC'
            else:
                struct[splited[1]] = splited[2][splited[2].find('{') + 1:splited[2].find('}')].split(',')

        return struct

    def write_output(self, predictions):
        output_dir = os.path.join(self.dir, 'output.txt')
        with open(output_dir, 'w') as f:
            for idx, entry in enumerate(predictions):
                f.write(f'{idx + 1} {entry}\n')

    def check_switch(self):
        if self.good_dir and self.good_bins:
            self.build_button["state"] = "normal"
        else:
            self.build_button["state"] = "disabled"
