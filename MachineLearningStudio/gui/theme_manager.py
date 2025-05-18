"""
Theme manager for handling application appearance
"""
import tkinter as tk
from tkinter import ttk
import os
import json
from assets.styles import (
    LIGHT_THEME, DARK_THEME, configure_treeview_style,
    LIGHT_BG, DARK_BG
)

class ThemeManager:
    """
    Manages application theming including light/dark modes and style customization.
    """
    def __init__(self, root):
        """
        Initialize the theme manager.
        
        Args:
            root: The root Tk instance
        """
        self.root = root
        self.style = ttk.Style()
        
        # Store current theme
        self.current_theme = "light"
        
        # Create config directory if it doesn't exist
        os.makedirs(os.path.expanduser("~/.ml_analyzer"), exist_ok=True)
        self.config_file = os.path.expanduser("~/.ml_analyzer/theme_config.json")
        
        # Load saved theme if available
        self.load_theme_preference()
        
        # Apply the theme
        self.apply_theme(self.current_theme)
    
    def set_theme(self, theme_name):
        """
        Set the application theme.
        
        Args:
            theme_name: 'light' or 'dark'
        """
        if theme_name in ["light", "dark"]:
            self.current_theme = theme_name
            self.apply_theme(theme_name)
            self.save_theme_preference()
    
    def toggle_theme(self):
        """Toggle between light and dark themes"""
        new_theme = "dark" if self.current_theme == "light" else "light"
        self.set_theme(new_theme)
    
    def apply_theme(self, theme_name):
        """
        Apply the specified theme to all widgets.
        
        Args:
            theme_name: 'light' or 'dark'
        """
        theme_dict = LIGHT_THEME if theme_name == "light" else DARK_THEME
        bg_color = LIGHT_BG if theme_name == "light" else DARK_BG
        
        # Set background color for main window
        self.root.configure(background=bg_color)
        
        # Apply ttk styles
        for style_name, style_dict in theme_dict.items():
            if "configure" in style_dict:
                self.style.configure(style_name, **style_dict["configure"])
            
            if "map" in style_dict:
                self.style.map(style_name, **style_dict["map"])
        
        # Configure treeview (special case)
        configure_treeview_style(self.style, theme_name)
    
    def save_theme_preference(self):
        """Save the current theme preference to a file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump({"theme": self.current_theme}, f)
        except Exception as e:
            print(f"Error saving theme preference: {e}")
    
    def load_theme_preference(self):
        """Load theme preference from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    if "theme" in config and config["theme"] in ["light", "dark"]:
                        self.current_theme = config["theme"]
        except Exception as e:
            print(f"Error loading theme preference: {e}")
