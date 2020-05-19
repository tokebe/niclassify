import tkinter as tk
import pandas as pd
from tkinter import ttk
from tkinter import filedialog
from .elements import DataPanel, TrainPanel, PredictPanel


class MainApp(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        # pre-defined variables to store important stuff
        self.data_file = None
        self.sheets = None
        self.raw_data = None
        self.column_names = None

        parent.title("Random Forest Classifier Tool")

        self.data_section = DataPanel(
            parent,
            self,
            text="Data",
            labelanchor=tk.N)
        self.data_section.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.operate_section = tk.Frame(
            parent)
        self.operate_section.pack(side=tk.RIGHT, anchor=tk.N)

        self.train_section = TrainPanel(
            self.operate_section,
            self,
            text="Train",
            labelanchor=tk.N)
        self.train_section.pack(fill=tk.X)

        self.predict_section = PredictPanel(
            self.operate_section,
            self,
            text="Predict",
            labelanchor=tk.N)
        self.predict_section.pack(fill=tk.X)

    def get_data(self):
        # self.parent.get_data()
        data_file = filedialog.askopenfilename(initialdir="data/")

        if len(data_file) > 0:
            self.data_file = data_file
            if self.data_file.split("/")[-1][-5:] == ".xlsx":
                self.data_section.excel_sheet_input.config(state=tk.ACTIVE)
                self.sheets = list(pd.read_excel(self.data_file, None).keys())
                self.data_section.excel_sheet_input["values"] = self.sheets
            else:
                self.data_section.excel_sheet_input.config(state=tk.DISABLED)
                self.column_names = pd.read_csv(
                    self.data_file,
                    na_values=self.parent.nans(),
                    keep_default_na=True,
                    sep=None,
                    engine="python"
                ).columns.values.tolist()
                self.data_section.col_select_panel.update_contents(
                    self.column_names)
                # TODO write update_contents to get new column names


def main():

    root = tk.Tk()
    root.style = ttk.Style()
    root.style.theme_use("winnative")
    # print(root.style.theme_names())
    MainApp(root)
    root.update()
    root.minsize(root.winfo_width(), root.winfo_height())
    root.mainloop()


if __name__ == "__main__":
    main()
