import tkinter as tk
from tkinter import ttk, messagebox
from logic.model_trainer import ModelTrainer
from logic.data_handler import DataHandler
from logic.evaluation import Evaluator

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ModelPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.model = None
        self.trainer = ModelTrainer()
        self.evaluator = Evaluator()
        self.entries = []

        self.create_widgets()

    def create_widgets(self):
        self.train_button = ttk.Button(self, text="Train", command=self.train_model)
        self.train_button.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.pred_frame = tk.LabelFrame(self, text="Manual Input for Prediction")
        self.pred_frame.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        self.predict_button = ttk.Button(self, text="Predict", command=self.predict_manual)
        self.predict_button.grid(row=2, column=0, padx=10, pady=10, sticky="w")

        self.output_text = tk.Text(self, height=15, width=100)
        self.output_text.grid(row=3, column=0, padx=10, pady=10)

    def train_model(self):
        try:
            handler = self.trainer.data_handler
            df = handler.get_processed_data()

            # be sure the dataframe is not empty
            target_col = df.columns[-1]
            handler.set_target_column(target_col)

            self.trainer.train()

            # set the model to the trainer 
            self.model = self.trainer.model
            self.display_input_fields(handler.X.columns)

            self.output_text.insert(tk.END, "Model trained successfully.\n")

            metrics, plot = self.evaluator.evaluate_model(
                model=self.model,
                X_test=self.trainer.X_test,
                y_test=self.trainer.y_test,
                task=self.trainer.task_type
            )

            self.output_text.insert(tk.END, "\n".join(metrics) + "\n")

            if plot:
                canvas = FigureCanvasTkAgg(plot, master=self)
                canvas.draw()
                canvas.get_tk_widget().grid(row=4, column=0)

        except Exception as e:
            messagebox.showerror("Training Error", str(e))

    def display_input_fields(self, columns):
        for widget in self.pred_frame.winfo_children():
            widget.destroy()

        self.entries = []
        for col in columns:
            row = tk.Frame(self.pred_frame)
            row.pack(anchor="w")
            tk.Label(row, text=col + ": ").pack(side="left")
            entry = ttk.Entry(row)
            entry.pack(side="left")
            self.entries.append(entry)

    def predict_manual(self):
        if not self.model:
            messagebox.showerror("Error", "Train the model first.")
            return

        try:
            input_data = [float(entry.get()) for entry in self.entries]
            result = self.model.predict([input_data])[0]
            self.output_text.insert(tk.END, f"\nPrediction Result: {result}\n")

        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
