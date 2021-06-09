from tkinter import filedialog, messagebox
from tkinter import *
import pandas as pd
import os
import preprocessing
from NaiveBayes import NaiveBayes


class GUI:
    def __init__(self):
        self.master = Tk()
        self.master.title("GUI")
        self.master.geometry("500x250")

        self.dir = ''
        self.text_entry = StringVar()
        self.text = Entry(self.master, textvariable=self.text_entry, width=20)

        self.browse_button = Button(self.master, text='Browse', command=lambda: self.browse())

        self.num_of_bins = 0
        self.bins_label = Label(self.master, text="Number of bins:")
        vcmd = self.master.register(self.validate)  # we have to wrap the command
        self.bins_entry = Entry(self.master, validate="key", validatecommand=(vcmd, '%P'))

        self.build_button = Button(self.master, text='Build', command=lambda: self.build())
        self.classify_button = Button(self.master, text='Classify', command=lambda: self.classify())

        # layout
        self.browse_button.grid(row=0)
        self.text.grid(row=0, column=1)
        self.bins_label.grid(row=4)
        self.bins_entry.grid(row=4, column=1)
        self.build_button.grid()
        self.classify_button.grid()

        self.master.mainloop()

    def browse(self):
        self.dir = filedialog.askdirectory()
        self.text_entry.set(self.dir)

    def validate(self, new_text):
        if not new_text:  # the field is being cleared
            self.num_of_bins = 0
            return True

        try:
            self.num_of_bins = int(new_text)
            print("number of bins correct")
            print(self.num_of_bins)
            return True
        except ValueError:
            print("number of bins is not a number")
            print(self.num_of_bins)
            return False

    def build(self):
        if self.check_files():
            try:
                self.struct = self.read_struct()
                train_df = pd.read_csv(os.path.join(self.dir, 'train.csv'))
                train_df = preprocessing.fill_missing(train_df, self.struct)
                train_df = preprocessing.discretize(train_df, self.num_of_bins, self.struct)

                self.model = NaiveBayes(train_df)
                messagebox.showinfo(title="Build", message="Model was Built successfuly!")
            except Exception as e:
                messagebox.showerror(title="Error", message=e)

    def check_files(self):
        print('trying to read files from given folder')
        try:
            df_train = pd.read_csv(os.path.join(self.dir, 'train.csv'))
            df_test = pd.read_csv(os.path.join(self.dir, 'test.csv'))
            if not os.path.isfile(os.path.join(self.dir, 'Structure.txt')):
                raise Exception()
            print("all files exist")
            return True
        except:
            messagebox.showerror(title="Error", message='Missing files in folder')
            return False

    def classify(self):
        if self.check_files():
            df_test = pd.read_csv(os.path.join(self.dir, 'test.csv'))
            print(df_test.head)

    def read_struct(self):
        path = os.path.join(self.dir,'Structure.txt')
        f = open(path, "r")
        struct = {}
        for line in f:
            splited = line.split(' ', 2)
            if 'NUMERIC' in splited[2]:
                struct[splited[1]] = 'NUMERIC'
            else:
                struct[splited[1]] = splited[2][splited[2].find('{')+1:splited[2].find('}')].split(',')


        return struct