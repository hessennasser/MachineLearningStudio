"""
Reusable UI components for the ML Analyzer application
"""
import tkinter as tk
from tkinter import ttk
import io
import base64
from PIL import Image, ImageTk

class ScrollableFrame(ttk.Frame):
    """
    A scrollable frame container
    """
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        
        # Create a canvas and a scrollbar
        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configure the canvas
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack the scrollbar and canvas
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Make sure the frame expands to fill the canvas
        self.canvas.bind('<Configure>', self.frame_width)
        
        # Add mouse wheel scrolling
        self.scrollable_frame.bind('<Enter>', self.bind_mousewheel)
        self.scrollable_frame.bind('<Leave>', self.unbind_mousewheel)
    
    def frame_width(self, event):
        """Adjust the width of the frame to match the canvas"""
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_frame, width=canvas_width)
    
    def bind_mousewheel(self, event):
        """Bind the mousewheel to scrolling"""
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)
        
    def unbind_mousewheel(self, event):
        """Unbind the mousewheel when leaving the frame"""
        self.canvas.unbind_all("<MouseWheel>")
        
    def on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")


class StatusProgressBar(ttk.Frame):
    """
    A combined status label and progress bar component
    """
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self, textvariable=self.status_var)
        self.progress_bar = ttk.Progressbar(self, orient="horizontal", length=200, mode="determinate")
        
        self.status_label.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        self.progress_bar.pack(side="right", padx=5, pady=5)
    
    def set_status(self, message, progress=None):
        """
        Update the status message and progress bar
        
        Args:
            message: Status message to display
            progress: Progress value (0-100)
        """
        self.status_var.set(message)
        
        if progress is not None:
            self.progress_bar["value"] = progress
        
        self.update_idletasks()
    
    def start_indeterminate(self, message="Processing..."):
        """Start indeterminate progress mode with a message"""
        self.status_var.set(message)
        self.progress_bar.config(mode="indeterminate")
        self.progress_bar.start()
        self.update_idletasks()
    
    def stop_indeterminate(self, message="Complete"):
        """Stop indeterminate progress mode"""
        self.progress_bar.stop()
        self.progress_bar.config(mode="determinate")
        self.status_var.set(message)
        self.update_idletasks()


class SvgImage:
    """
    Helper class to render SVG images in Tkinter
    """
    def __init__(self, svg_data, width=None, height=None):
        """
        Create an image from SVG data
        
        Args:
            svg_data: The SVG XML data as a string
            width: Optional width to resize the image
            height: Optional height to resize the image
        """
        # Convert SVG to PNG using Pillow with cairosvg
        try:
            # Try to use cairosvg if available
            import cairosvg
            png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
            self.image = Image.open(io.BytesIO(png_data))
        except ImportError:
            # Fallback to a simple placeholder if cairosvg is not available
            self.image = Image.new('RGBA', (width or 100, height or 100), (240, 240, 240, 255))
        
        # Resize if dimensions are provided
        if width and height:
            self.image = self.image.resize((width, height), Image.LANCZOS)
        
        # Convert to PhotoImage for Tkinter
        self.tk_image = ImageTk.PhotoImage(self.image)
    
    def get(self):
        """Get the Tkinter PhotoImage object"""
        return self.tk_image


class TooltipManager:
    """
    Manager for tooltips on widgets
    """
    def __init__(self, delay=0.5):
        """
        Initialize the tooltip manager
        
        Args:
            delay: Delay in seconds before showing tooltip
        """
        self.delay = delay  # Seconds before the tooltip appears
        self.tip_window = None
        self.widget = None
        self.id = None
        self.x = self.y = 0
    
    def add_tooltip(self, widget, text):
        """
        Add a tooltip to a widget
        
        Args:
            widget: The widget to add the tooltip to
            text: The tooltip text
        """
        self.widget = widget
        def enter(event):
            self.schedule_tooltip(text)
        def leave(event):
            self.unschedule_tooltip()
            self.hide_tooltip()
        
        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)
        widget.bind('<ButtonPress>', self.hide_tooltip)
    
    def schedule_tooltip(self, text):
        """Schedule the tooltip to appear after a delay"""
        self.unschedule_tooltip()
        self.id = self.widget.after(int(self.delay * 1000), lambda: self.show_tooltip(text))
    
    def unschedule_tooltip(self):
        """Cancel the scheduled tooltip"""
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None
    
    def show_tooltip(self, text):
        """Show the tooltip"""
        if self.tip_window or not text:
            return
        
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 1
        
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        
        label = ttk.Label(tw, text=text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1)
        label.pack(ipadx=4, ipady=2)
    
    def hide_tooltip(self, event=None):
        """Hide the tooltip"""
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class DataTable(ttk.Frame):
    """
    A component for displaying tabular data
    """
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        
        # Create a frame with a scrollbar
        self.frame = ttk.Frame(self)
        self.frame.pack(fill="both", expand=True)
        
        # Scrollbars
        self.y_scrollbar = ttk.Scrollbar(self.frame)
        self.y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.x_scrollbar = ttk.Scrollbar(self.frame, orient=tk.HORIZONTAL)
        self.x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Treeview for the table
        self.tree = ttk.Treeview(self.frame, 
                                yscrollcommand=self.y_scrollbar.set,
                                xscrollcommand=self.x_scrollbar.set)
        
        self.y_scrollbar.config(command=self.tree.yview)
        self.x_scrollbar.config(command=self.tree.xview)
        
        self.tree.pack(side=tk.LEFT, fill="both", expand=True)
    
    def set_data(self, headers, data, max_rows=None):
        """
        Set the data for the table
        
        Args:
            headers: List of column headers
            data: List of rows, each row is a list of values
            max_rows: Maximum number of rows to display
        """
        # Clear existing data
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Configure columns
        self.tree["columns"] = headers
        self.tree["show"] = "headings"  # Hide the first empty column
        
        # Set column headings
        for col in headers:
            self.tree.heading(col, text=col)
            max_width = max(len(str(col)), 
                           max([len(str(row[headers.index(col)])) for row in data[:max_rows or len(data)]]) if data else 0)
            self.tree.column(col, width=min(max_width * 10, 200), minwidth=50)
        
        # Insert data
        for i, row in enumerate(data[:max_rows]):
            values = []
            for col_idx, col in enumerate(headers):
                if col_idx < len(row):
                    values.append(str(row[col_idx]))
                else:
                    values.append("")
            
            self.tree.insert("", tk.END, text=f"Row {i+1}", values=values)
