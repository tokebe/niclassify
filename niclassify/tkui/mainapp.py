


# TODO replace a lot of "stored" values with properties that get data from file
# TODO replace any temporary files with tempfile, including read data files




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
