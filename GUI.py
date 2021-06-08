from tkinter import filedialog, messagebox
from tkinter import *
import pandas as pd
import os


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

        # layout
        self.browse_button.grid(row=0)
        self.text.grid(row=0, column=1)
        self.bins_label.grid(row=4)
        self.bins_entry.grid(row=4, column=1)
        self.build_button.grid()

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

        self.check_files()

        self.build_model()

    def check_files(self):
        print('trying to read files from given folder')
        try:
            df_train = pd.read_csv(os.path.join(self.dir, 'train.csv'))
            df_test = pd.read_csv(os.path.join(self.dir, 'test.csv'))
            if not os.path.isfile(os.path.join(self.dir, 'Structure.txt')):
                raise Exception()
            print("all files exist")
        except:
            messagebox.showerror(title="Error", message='Missing files in folder')

    def build_model(self):
        pass
