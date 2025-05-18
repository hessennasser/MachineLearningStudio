"""
Style definitions for the ML Analyzer application
"""

# Common style constants
HEADING_FONT = ("Helvetica", 16, "bold")
SUBHEADING_FONT = ("Helvetica", 12, "bold")
NORMAL_FONT = ("Helvetica", 10)
SMALL_FONT = ("Helvetica", 8)

# Light theme colors
LIGHT_BG = "#f5f5f5"
LIGHT_FG = "#333333"
LIGHT_ACCENT = "#4a86e8"
LIGHT_SECONDARY = "#9fc5e8"
LIGHT_SUCCESS = "#6aa84f"
LIGHT_WARNING = "#e69138"
LIGHT_ERROR = "#cc0000"
LIGHT_FRAME_BG = "#ffffff"
LIGHT_BUTTON_BG = "#e7e7e7"
LIGHT_BUTTON_ACTIVE = "#d6d6d6"

# Dark theme colors
DARK_BG = "#2d2d2d"
DARK_FG = "#f0f0f0"
DARK_ACCENT = "#5c85d6"
DARK_SECONDARY = "#3d5a80"
DARK_SUCCESS = "#78a75a"
DARK_WARNING = "#e6a03f"
DARK_ERROR = "#d93030"
DARK_FRAME_BG = "#3d3d3d"
DARK_BUTTON_BG = "#444444"
DARK_BUTTON_ACTIVE = "#555555"

# TTK style configuration for light theme
LIGHT_THEME = {
    "TFrame": {
        "configure": {"background": LIGHT_FRAME_BG}
    },
    "TLabel": {
        "configure": {"background": LIGHT_FRAME_BG, "foreground": LIGHT_FG}
    },
    "TButton": {
        "configure": {"padding": (5, 2), "background": LIGHT_BUTTON_BG},
        "map": {
            "background": [("active", LIGHT_BUTTON_ACTIVE), 
                           ("disabled", LIGHT_SECONDARY)]
        }
    },
    "Accent.TButton": {
        "configure": {"padding": (5, 2), "background": LIGHT_ACCENT, 
                     "foreground": "white"},
        "map": {
            "background": [("active", LIGHT_SECONDARY), 
                          ("disabled", "#a0a0a0")]
        }
    },
    "Success.TButton": {
        "configure": {"padding": (5, 2), "background": LIGHT_SUCCESS, 
                     "foreground": "white"},
        "map": {
            "background": [("active", "#7ab85f"), 
                          ("disabled", "#a0a0a0")]
        }
    },
    "TEntry": {
        "configure": {"padding": 5}
    },
    "TCombobox": {
        "configure": {"padding": 5}
    },
    "TNotebook": {
        "configure": {"tabmargins": [2, 5, 2, 0]}
    },
    "TNotebook.Tab": {
        "configure": {"padding": [10, 4], "background": LIGHT_BUTTON_BG},
        "map": {
            "background": [("selected", LIGHT_FRAME_BG), 
                          ("active", LIGHT_BUTTON_ACTIVE)]
        }
    },
    "Horizontal.TProgressbar": {
        "configure": {"background": LIGHT_ACCENT}
    }
}

# TTK style configuration for dark theme
DARK_THEME = {
    "TFrame": {
        "configure": {"background": DARK_FRAME_BG}
    },
    "TLabel": {
        "configure": {"background": DARK_FRAME_BG, "foreground": DARK_FG}
    },
    "TButton": {
        "configure": {"padding": (5, 2), "background": DARK_BUTTON_BG, 
                     "foreground": DARK_FG},
        "map": {
            "background": [("active", DARK_BUTTON_ACTIVE), 
                          ("disabled", "#666666")],
            "foreground": [("disabled", "#aaaaaa")]
        }
    },
    "Accent.TButton": {
        "configure": {"padding": (5, 2), "background": DARK_ACCENT, 
                     "foreground": "white"},
        "map": {
            "background": [("active", "#6a93e8"), 
                          ("disabled", "#555555")]
        }
    },
    "Success.TButton": {
        "configure": {"padding": (5, 2), "background": DARK_SUCCESS, 
                     "foreground": "white"},
        "map": {
            "background": [("active", "#88bb6a"), 
                          ("disabled", "#555555")]
        }
    },
    "TEntry": {
        "configure": {"padding": 5, "fieldbackground": "#444444", 
                     "foreground": DARK_FG}
    },
    "TCombobox": {
        "configure": {"padding": 5, "fieldbackground": "#444444", 
                     "foreground": DARK_FG}
    },
    "TNotebook": {
        "configure": {"tabmargins": [2, 5, 2, 0], "background": DARK_BG}
    },
    "TNotebook.Tab": {
        "configure": {"padding": [10, 4], "background": DARK_BUTTON_BG, 
                     "foreground": DARK_FG},
        "map": {
            "background": [("selected", DARK_FRAME_BG), 
                          ("active", DARK_BUTTON_ACTIVE)]
        }
    },
    "Horizontal.TProgressbar": {
        "configure": {"background": DARK_ACCENT}
    }
}

# Custom treeview styles
def configure_treeview_style(style, theme_type="light"):
    if theme_type == "light":
        style.configure("Treeview", 
                        background=LIGHT_FRAME_BG,
                        foreground=LIGHT_FG, 
                        fieldbackground=LIGHT_FRAME_BG)
        style.map('Treeview', 
                 background=[('selected', LIGHT_ACCENT)])
    else:
        style.configure("Treeview", 
                        background=DARK_FRAME_BG,
                        foreground=DARK_FG, 
                        fieldbackground=DARK_FRAME_BG)
        style.map('Treeview', 
                 background=[('selected', DARK_ACCENT)])
