
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd

from logic.data_handler import DataHandler

class DatasetPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.data_handler = DataHandler()

        self.df = None
        self.task = tk.StringVar()
        self.selected_algo = tk.StringVar()

        self.create_widgets()

    def create_widgets(self):
        upload_btn = ttk.Button(self, text="Upload CSV", command=self.upload_csv)
        upload_btn.grid(row=0, column=0, padx=10, pady=10)

        self.columns_label = tk.Label(self, text="Columns will appear here")
        self.columns_label.grid(row=1, column=0, sticky="w", padx=10)

        task_label = tk.Label(self, text="Select Task:")
        task_label.grid(row=2, column=0, sticky="w", padx=10)

        task_frame = tk.Frame(self)
        task_frame.grid(row=3, column=0, sticky="w", padx=10)

        tk.Radiobutton(task_frame, text="Classification", variable=self.task, value="classification",
                      command=self.update_algorithms).pack(side="left")
        tk.Radiobutton(task_frame, text="Regression", variable=self.task, value="regression",
                      command=self.update_algorithms).pack(side="left")

        algo_label = tk.Label(self, text="Select Algorithm:")
        algo_label.grid(row=4, column=0, sticky="w", padx=10)

        self.algo_combo = ttk.Combobox(self, textvariable=self.selected_algo, state="readonly")
        self.algo_combo.grid(row=5, column=0, padx=10, sticky="w")

        preview_label = tk.Label(self, text="Data Preview:")
        preview_label.grid(row=6, column=0, sticky="w", padx=10)

        self.preview_text = tk.Text(self, height=10, width=100)
        self.preview_text.grid(row=7, column=0, padx=10, pady=5)

        continue_btn = ttk.Button(self, text="Continue", command=self.go_to_model_page)
        continue_btn.grid(row=8, column=0, pady=10)

    def upload_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        try:
            self.df = pd.read_csv(file_path)
            self.data_handler.set_dataframe(self.df)

            # عرض الأعمدة
            self.columns_label.config(text=", ".join(self.df.columns.tolist()))

            # عرض أول 5 صفوف
            self.preview_text.delete("1.0", tk.END)
            self.preview_text.insert(tk.END, self.df.head().to_string())

            # تجهيز البيانات تلقائيًا
            self.data_handler.preprocess()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV:\n{str(e)}")

    def update_algorithms(self):
        task = self.task.get()
        if task == "classification":
            self.algo_combo["values"] = ["KNN", "SVM", "Decision Tree"]
        elif task == "regression":
            self.algo_combo["values"] = ["Linear Regression"]

    def go_to_model_page(self):
        if self.df is None or not self.selected_algo.get():
            messagebox.showwarning("Incomplete", "Please upload data and select algorithm")
            return

        from gui.model_page import ModelPage
        self.controller.show_frame(ModelPage)
