"""
Welcome page for the ML Analyzer application
"""
import tkinter as tk
from tkinter import ttk
import webbrowser
from assets.styles import HEADING_FONT, SUBHEADING_FONT, NORMAL_FONT
from assets.images import WELCOME_BANNER
from assets.icons import (
    DATA_ICON, MODEL_ICON, CHART_ICON,
    NEXT_ICON, INFO_ICON
)
from gui.components import SvgImage, ScrollableFrame, TooltipManager

class WelcomePage(ttk.Frame):
    """
    Welcome page displayed when the application starts
    """
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.tooltip_manager = TooltipManager()
        
        # Create a scrollable main frame
        self.scrollable = ScrollableFrame(self)
        self.scrollable.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Main content frame
        self.content_frame = ttk.Frame(self.scrollable.scrollable_frame)
        self.content_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create and pack the welcome banner
        self.create_banner()
        
        # Create and pack the features section
        self.create_features_section()
        
        # Create and pack the get started button
        self.create_action_buttons()
    
    def create_banner(self):
        """Create the welcome banner with logo and text"""
        banner_frame = ttk.Frame(self.content_frame)
        banner_frame.pack(fill="x", padx=20, pady=20)
        
        # Add ML Analyzer banner image
        try:
            banner_image = SvgImage(WELCOME_BANNER, width=600, height=200)
            banner_label = ttk.Label(banner_frame, image=banner_image.get())
            banner_label.image = banner_image.get()  # Keep a reference
            banner_label.pack(padx=20, pady=20)
        except Exception as e:
            # Fallback to text if image can't be loaded
            print(f"Error loading banner: {e}")
            title_label = ttk.Label(
                banner_frame, 
                text="ML Analyzer", 
                font=("Helvetica", 36, "bold")
            )
            title_label.pack(padx=20, pady=20)
        
        # Add description
        description = (
            "ML Analyzer is a desktop application that helps you analyze "
            "datasets, train machine learning models, and evaluate their "
            "performance - all with an intuitive user interface."
        )
        
        desc_label = ttk.Label(
            banner_frame, 
            text=description, 
            font=NORMAL_FONT,
            wraplength=600, 
            justify=tk.CENTER
        )
        desc_label.pack(padx=20, pady=10)
    
    def create_features_section(self):
        """Create the features section with cards"""
        features_frame = ttk.Frame(self.content_frame)
        features_frame.pack(fill="x", padx=20, pady=10)
        
        # Section title
        section_title = ttk.Label(
            features_frame, 
            text="Key Features", 
            font=SUBHEADING_FONT
        )
        section_title.pack(pady=(0, 10), anchor="w")
        
        # Create a frame for the feature cards
        cards_frame = ttk.Frame(features_frame)
        cards_frame.pack(fill="x", expand=True)
        
        # Configure a 3-column grid
        for i in range(3):
            cards_frame.columnconfigure(i, weight=1)
        
        # Feature 1: Data Analysis
        self.create_feature_card(
            cards_frame, 
            0, 0,
            DATA_ICON,
            "Data Analysis", 
            "Upload CSV datasets and analyze their structure. "
            "Automatic preprocessing handles missing values and "
            "categorical data.",
            "Upload and analyze your CSV data"
        )
        
        # Feature 2: Model Training
        self.create_feature_card(
            cards_frame, 
            0, 1,
            MODEL_ICON,
            "Model Training", 
            "Train various machine learning models for classification "
            "and regression tasks. Select algorithm and parameters "
            "based on your needs.",
            "Train ML models on your data"
        )
        
        # Feature 3: Evaluation
        self.create_feature_card(
            cards_frame, 
            0, 2,
            CHART_ICON,
            "Performance Evaluation", 
            "Evaluate model performance with metrics like accuracy, "
            "precision, recall for classification; MAE, RMSE, R² "
            "for regression.",
            "Evaluate model performance with key metrics"
        )
    
    def create_feature_card(self, parent, row, col, icon_svg, title, description, tooltip=None):
        """Create a feature card with icon, title and description"""
        # Card container
        style = ttk.Style()
        current_theme = style.theme_use()
        
        card = ttk.Frame(parent, padding=10)
        card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        
        # Icon
        try:
            icon_image = SvgImage(icon_svg, width=40, height=40)
            icon_label = ttk.Label(card, image=icon_image.get())
            icon_label.image = icon_image.get()  # Keep a reference
            icon_label.pack(pady=(0, 10))
        except Exception as e:
            print(f"Error loading icon: {e}")
        
        # Title
        title_label = ttk.Label(card, text=title, font=SUBHEADING_FONT)
        title_label.pack(pady=(0, 5))
        
        # Description
        desc_label = ttk.Label(
            card, 
            text=description, 
            wraplength=180, 
            justify=tk.CENTER
        )
        desc_label.pack(pady=5, fill="x", expand=True)
        
        # Add tooltip if provided
        if tooltip:
            self.tooltip_manager.add_tooltip(card, tooltip)
    
    def create_action_buttons(self):
        """Create action buttons section"""
        buttons_frame = ttk.Frame(self.content_frame)
        buttons_frame.pack(fill="x", padx=20, pady=40)
        
        # Get started button
        self.start_button = ttk.Button(
            buttons_frame,
            text="Get Started",
            command=lambda: self.controller.show_frame("dataset"),
            style="Accent.TButton",
            padding=(20, 10)
        )
        self.start_button.pack(pady=10)
        
        # Footer links frame
        footer_frame = ttk.Frame(buttons_frame)
        footer_frame.pack(fill="x", pady=10)
        
        # Left-aligned credits
        credits_label = ttk.Label(
            footer_frame, 
            text="© 2023 ML Analyzer Team", 
            font=("Helvetica", 8)
        )
        credits_label.pack(side=tk.LEFT)
        
        # Right-aligned help link
        help_button = ttk.Button(
            footer_frame,
            text="Documentation",
            command=self.open_documentation,
            style="TButton",
            padding=(5, 2)
        )
        help_button.pack(side=tk.RIGHT)
    
    def open_documentation(self):
        """Open documentation in the default web browser"""
        # In a real app, this would point to actual documentation
        webbrowser.open("https://github.com/ml-analyzer/docs")

    def on_show(self):
        """Called when this frame is shown"""
        # Update any dynamic content here if needed
        self.controller.update_status("Welcome to ML Analyzer")
