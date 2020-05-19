import tkinter as tk
from tkinter import ttk
import core
import tkui


class MainRoot(tk.Tk):

    def nans(self):
        return core.NANS

    # def get_data(self):
    #     print("welp")


def main():

    core.assure_path()

    root = MainRoot()
    root.style = ttk.Style()
    root.style.theme_use("winnative")
    # print(root.style.theme_names())
    tkui.MainApp(root)
    root.update()
    root.minsize(root.winfo_width(), root.winfo_height())
    root.mainloop()


if __name__ == "__main__":
    main()
