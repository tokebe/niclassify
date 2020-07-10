import sys
import threading

import tkinter as tk


class TextRedirector():
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, out):
        self.widget.configure(state="normal")
        self.widget.insert("end", out, (self.tag,))
        self.widget.configure(state="disabled")

    def fileno(self):
        return 1


class OutputConsole(tk.Toplevel):
    def __init__(self, parent, app, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.app = app

        self.process = None

        self.text = tk.Text(self)
        self.text.pack(expand=True, fill=tk.BOTH, side=tk.LEFT)

        self.scroll = tk.Scrollbar(self, command=self.text.yview)
        self.scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.text.config(yscrollcommand=self.scroll.set)
        self.text.tag_configure("stderr", foreground="red")

    #     self.old_stdout = sys.stdout
    #     self.old_stderr = sys.stderr

    #     sys.stdout = TextRedirector(self.text, "stdout")
    #     sys.sterr = TextRedirector(self.text, "stderr")

    #     self.protocol("WM_DELETE_WINDOW", self.wm_exit)

    # def wm_exit(self):
    #     sys.stdout = self.old_stdout
    #     sys.stderr = self.old_stderr
    #     self.destroy()

    def _read(self):
        # for line in self.process.stdout:
        #     self.text.insert(tk.END, line)

        self.text.insert(tk.END, "kill me")
        print("thread started.")

        while True:
            print("attempting to read line...")
            line = self.process.stdout.read(1)
            if not line:
                print("end of line.")
                break
            else:
                print("got line : {}".format(line))
                self.text.config(state=tk.NORMAL)
                self.text.insert(tk.END, "fuckin hell")
                self.text.yview(tk.END)
                self.text.config(state=tk.DISABLED)

    def read(self, process):
        self.process = process
        thread = threading.Thread(target=self._read)
        thread.start()
