# gui/welcome_page.py

import tkinter as tk
from tkinter import ttk

class WelcomePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.configure(bg="#f5f5f5")

        title = tk.Label(self, text="ML Analyzer: Predict & Classify Any Dataset",
                        font=("Helvetica", 20, "bold"), bg="#f5f5f5", fg="#333")
        title.pack(pady=150)

        start_button = ttk.Button(self, text="Start", command=self.start_app)
        start_button.pack()

    def start_app(self):
        from gui.dataset_page import DatasetPage
        self.controller.show_frame(DatasetPage)
