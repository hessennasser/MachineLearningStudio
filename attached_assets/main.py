import tkinter as tk
from gui.welcome_page import WelcomePage

class MLAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ML Analyzer")
        self.geometry("800x600")
        self.resizable(False, False)

        self.container = tk.Frame(self)
        self.container.pack(fill="both", expand=True)

        self.frames = {}
        self.show_frame(WelcomePage)

    def show_frame(self, page_class):
        frame = self.frames.get(page_class)
        if frame is None:
            frame = page_class(parent=self.container, controller=self)
            self.frames[page_class] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        frame.tkraise()

if __name__ == "__main__":
    app = MLAnalyzerApp()
    app.mainloop()
