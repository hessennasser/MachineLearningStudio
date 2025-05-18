import tkinter as tk
from tkinter import ttk
import os
import sys
from gui.welcome_page import WelcomePage
from gui.dataset_page import DatasetPage
from gui.model_page import ModelPage
from gui.theme_manager import ThemeManager
from config.settings import APP_TITLE, DEFAULT_GEOMETRY, APP_VERSION

class MLAnalyzerApp(tk.Tk):
    """
    Main application class for the ML Analyzer.
    Manages the overall UI container and frame navigation.
    """
    def __init__(self):
        super().__init__()
        
        # Configure the main window
        self.title(f"{APP_TITLE} v{APP_VERSION}")
        self.geometry(DEFAULT_GEOMETRY)
        self.minsize(800, 600)
        
        # Initialize theme manager
        self.theme_manager = ThemeManager(self)
        
        # Create main container
        self.container = ttk.Frame(self)
        self.container.pack(fill="both", expand=True, padx=10, pady=10)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        
        # Setup status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self, textvariable=self.status_var, 
                                    relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create menu
        self.create_menu()
        
        # Initialize frame dictionary
        self.frames = {}
        
        # Register all pages
        self.available_frames = {
            "welcome": WelcomePage,
            "dataset": DatasetPage,
            "model": ModelPage
        }
        
        # Application state
        self.app_data = {
            "dataset": None,
            "dataset_path": None,
            "task_type": None,
            "algorithm": None,
            "target_column": None,
            "model": None,
            "evaluation_results": None,
            "preprocessing_params": None
        }
        
        # Start with welcome page
        self.show_frame("welcome")
    
    def create_menu(self):
        """Create the application menu bar"""
        menubar = tk.Menu(self)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New Session", command=self.reset_session)
        file_menu.add_command(label="Save Model", command=self.save_model)
        file_menu.add_command(label="Load Model", command=self.load_model)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Light Theme", command=lambda: self.theme_manager.set_theme("light"))
        view_menu.add_command(label="Dark Theme", command=lambda: self.theme_manager.set_theme("dark"))
        menubar.add_cascade(label="View", menu=view_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Documentation")
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.config(menu=menubar)
    
    def show_frame(self, page_name):
        """
        Raise the specified frame to the top
        
        Args:
            page_name: The name of the page to display
        """
        # If frame doesn't exist yet, create it
        if page_name not in self.frames:
            frame_class = self.available_frames[page_name]
            frame = frame_class(self.container, self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        
        # Raise the frame to the top
        self.frames[page_name].tkraise()
        
        # If frame has an on_show method, call it
        if hasattr(self.frames[page_name], "on_show"):
            self.frames[page_name].on_show()
    
    def update_status(self, message):
        """Update the status bar with a message"""
        self.status_var.set(message)
        self.update_idletasks()
    
    def reset_session(self):
        """Reset the current session data"""
        # Clear application data
        for key in self.app_data:
            self.app_data[key] = None
        
        # Delete and recreate frames to reset their state
        for frame in self.frames.values():
            frame.destroy()
        self.frames = {}
        
        # Show welcome page
        self.show_frame("welcome")
        self.update_status("New session started")
    
    def save_model(self):
        """Save the current model if one exists"""
        if self.app_data["model"] is None:
            self.update_status("No model to save")
            return
        
        # This will be implemented in the model page's save_model method
        if "model" in self.frames:
            self.frames["model"].save_model()
    
    def load_model(self):
        """Load a saved model"""
        # This will be implemented in the model page's load_model method
        if "model" in self.frames:
            self.frames["model"].load_model()
        else:
            # Create model page if it doesn't exist
            self.show_frame("model")
            self.frames["model"].load_model()
    
    def show_about(self):
        """Show about dialog"""
        about_window = tk.Toplevel(self)
        about_window.title("About ML Analyzer")
        about_window.geometry("400x300")
        about_window.resizable(False, False)
        about_window.transient(self)
        about_window.grab_set()
        
        # Center the window
        about_window.update_idletasks()
        width = about_window.winfo_width()
        height = about_window.winfo_height()
        x = (self.winfo_width() // 2) - (width // 2) + self.winfo_x()
        y = (self.winfo_height() // 2) - (height // 2) + self.winfo_y()
        about_window.geometry(f"+{x}+{y}")
        
        # Content
        ttk.Label(about_window, text=f"ML Analyzer v{APP_VERSION}", 
                 font=("Helvetica", 16, "bold")).pack(pady=20)
        ttk.Label(about_window, text="A desktop application for machine learning analysis",
                 wraplength=350).pack(pady=10)
        ttk.Label(about_window, text="© 2023 ML Analyzer Team").pack(pady=20)
        ttk.Button(about_window, text="Close", command=about_window.destroy).pack(pady=10)

if __name__ == "__main__":
    app = MLAnalyzerApp()
    app.mainloop()
