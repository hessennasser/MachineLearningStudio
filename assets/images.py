"""
SVG images and placeholders for the ML Analyzer application
"""

# Welcome page banner SVG
WELCOME_BANNER = """
<svg width="600" height="200" xmlns="http://www.w3.org/2000/svg">
    <style>
        .title { font: bold 40px sans-serif; fill: #4a86e8; }
        .subtitle { font: 20px sans-serif; fill: #666; }
        .path { stroke: #4a86e8; stroke-width: 2; fill: none; }
        .node { fill: #4a86e8; }
        .path-alt { stroke: #9fc5e8; stroke-width: 2; stroke-dasharray: 5,5; fill: none; }
    </style>
    
    <!-- Title -->
    <text x="50" y="50" class="title">ML Analyzer</text>
    <text x="50" y="80" class="subtitle">Analyze, Train, Predict</text>
    
    <!-- Neural network visualization -->
    <line x1="400" y1="40" x2="450" y2="80" class="path" />
    <line x1="400" y1="40" x2="450" y2="120" class="path" />
    <line x1="400" y1="40" x2="450" y2="160" class="path-alt" />
    
    <line x1="400" y1="80" x2="450" y2="80" class="path-alt" />
    <line x1="400" y1="80" x2="450" y2="120" class="path" />
    <line x1="400" y1="80" x2="450" y2="160" class="path" />
    
    <line x1="400" y1="120" x2="450" y2="80" class="path" />
    <line x1="400" y1="120" x2="450" y2="120" class="path-alt" />
    <line x1="400" y1="120" x2="450" y2="160" class="path" />
    
    <line x1="400" y1="160" x2="450" y2="80" class="path-alt" />
    <line x1="400" y1="160" x2="450" y2="120" class="path" />
    <line x1="400" y1="160" x2="450" y2="160" class="path" />
    
    <line x1="450" y1="80" x2="500" y2="100" class="path" />
    <line x1="450" y1="120" x2="500" y2="100" class="path" />
    <line x1="450" y1="160" x2="500" y2="100" class="path-alt" />
    
    <line x1="350" y1="100" x2="400" y2="40" class="path" />
    <line x1="350" y1="100" x2="400" y2="80" class="path-alt" />
    <line x1="350" y1="100" x2="400" y2="120" class="path" />
    <line x1="350" y1="100" x2="400" y2="160" class="path" />
    
    <circle cx="350" cy="100" r="5" class="node" />
    
    <circle cx="400" cy="40" r="5" class="node" />
    <circle cx="400" cy="80" r="5" class="node" />
    <circle cx="400" cy="120" r="5" class="node" />
    <circle cx="400" cy="160" r="5" class="node" />
    
    <circle cx="450" cy="80" r="5" class="node" />
    <circle cx="450" cy="120" r="5" class="node" />
    <circle cx="450" cy="160" r="5" class="node" />
    
    <circle cx="500" cy="100" r="5" class="node" />
    
    <!-- Data flow visualization -->
    <path d="M 50 120 C 70 110, 90 130, 110 120 C 130 110, 150 130, 170 120 C 190 110, 210 130, 230 120" class="path" />
    <rect x="60" y="140" width="160" height="20" rx="5" ry="5" fill="#9fc5e8" opacity="0.6" />
    <text x="85" y="155" font-size="12" fill="#333">Dataset Analysis</text>
</svg>
"""

# Dataset page visualization
DATASET_VISUALIZATION = """
<svg width="600" height="200" xmlns="http://www.w3.org/2000/svg">
    <style>
        .title { font: bold 24px sans-serif; fill: #4a86e8; }
        .box { stroke: #ccc; stroke-width: 1; fill: white; }
        .box-highlight { stroke: #4a86e8; stroke-width: 2; fill: white; }
        .arrow { stroke: #666; stroke-width: 1.5; fill: none; marker-end: url(#arrowhead); }
        .text { font: 12px sans-serif; fill: #333; }
    </style>
    
    <defs>
        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#666" />
        </marker>
    </defs>
    
    <!-- CSV File -->
    <rect x="50" y="50" width="100" height="60" rx="5" ry="5" class="box" />
    <text x="75" y="85" class="text" text-anchor="middle">CSV Data</text>
    
    <!-- Arrow -->
    <path d="M 150 80 L 200 80" class="arrow" />
    
    <!-- Preprocessing -->
    <rect x="200" y="50" width="100" height="60" rx="5" ry="5" class="box-highlight" />
    <text x="250" y="85" class="text" text-anchor="middle">Preprocessing</text>
    
    <!-- Arrow -->
    <path d="M 300 80 L 350 80" class="arrow" />
    
    <!-- Feature Engineering -->
    <rect x="350" y="50" width="100" height="60" rx="5" ry="5" class="box" />
    <text x="400" y="85" class="text" text-anchor="middle">Feature Selection</text>
    
    <!-- Arrow -->
    <path d="M 450 80 L 500 80" class="arrow" />
    
    <!-- Model Training -->
    <rect x="500" y="50" width="100" height="60" rx="5" ry="5" class="box" />
    <text x="550" y="85" class="text" text-anchor="middle">Model Ready</text>
    
    <text x="50" y="150" class="title">Data Preprocessing Pipeline</text>
</svg>
"""

# Model page visualization
MODEL_VISUALIZATION = """
<svg width="600" height="200" xmlns="http://www.w3.org/2000/svg">
    <style>
        .title { font: bold 24px sans-serif; fill: #4a86e8; }
        .box { stroke: #ccc; stroke-width: 1; fill: white; }
        .box-highlight { stroke: #4a86e8; stroke-width: 2; fill: white; }
        .arrow { stroke: #666; stroke-width: 1.5; fill: none; marker-end: url(#arrowhead); }
        .text { font: 12px sans-serif; fill: #333; }
        .metrics { font: 11px monospace; fill: #333; }
    </style>
    
    <defs>
        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#666" />
        </marker>
    </defs>
    
    <!-- Processed Data -->
    <rect x="50" y="50" width="100" height="60" rx="5" ry="5" class="box" />
    <text x="100" y="85" class="text" text-anchor="middle">Processed Data</text>
    
    <!-- Arrow -->
    <path d="M 150 80 L 200 80" class="arrow" />
    
    <!-- Model Training -->
    <rect x="200" y="50" width="100" height="60" rx="5" ry="5" class="box-highlight" />
    <text x="250" y="70" class="text" text-anchor="middle">Model</text>
    <text x="250" y="90" class="text" text-anchor="middle">Training</text>
    
    <!-- Arrow -->
    <path d="M 300 80 L 350 80" class="arrow" />
    
    <!-- Evaluation -->
    <rect x="350" y="50" width="100" height="60" rx="5" ry="5" class="box" />
    <text x="400" y="85" class="text" text-anchor="middle">Evaluation</text>
    
    <!-- Arrow -->
    <path d="M 450 80 L 500 80" class="arrow" />
    
    <!-- Prediction -->
    <rect x="500" y="50" width="100" height="60" rx="5" ry="5" class="box" />
    <text x="550" y="85" class="text" text-anchor="middle">Prediction</text>
    
    <text x="50" y="150" class="title">Model Training Pipeline</text>
    
    <!-- Sample Metrics -->
    <rect x="50" y="170" width="550" height="20" rx="3" ry="3" fill="#f5f5f5" stroke="#ddd" />
    <text x="300" y="185" class="metrics" text-anchor="middle">Accuracy: 0.92 | Precision: 0.91 | Recall: 0.94 | F1: 0.92</text>
</svg>
"""

# EDA visualization placeholder
EDA_VISUALIZATION = """
<svg width="600" height="300" xmlns="http://www.w3.org/2000/svg">
    <style>
        .title { font: bold 24px sans-serif; fill: #4a86e8; }
        .axis { stroke: #666; stroke-width: 1; }
        .bar { fill: #4a86e8; opacity: 0.7; }
        .bar-alt { fill: #9fc5e8; opacity: 0.7; }
        .axis-label { font: 12px sans-serif; fill: #333; }
        .grid { stroke: #eee; stroke-width: 1; }
    </style>
    
    <text x="50" y="30" class="title">Data Exploration</text>
    
    <!-- Coordinate system -->
    <line x1="50" y1="250" x2="550" y2="250" class="axis" />
    <line x1="50" y1="250" x2="50" y2="50" class="axis" />
    
    <!-- Grid lines -->
    <line x1="50" y1="200" x2="550" y2="200" class="grid" />
    <line x1="50" y1="150" x2="550" y2="150" class="grid" />
    <line x1="50" y1="100" x2="550" y2="100" class="grid" />
    
    <!-- Bars representing data distribution -->
    <rect x="75" y="150" width="30" height="100" class="bar" />
    <rect x="125" y="100" width="30" height="150" class="bar" />
    <rect x="175" y="170" width="30" height="80" class="bar" />
    <rect x="225" y="80" width="30" height="170" class="bar" />
    <rect x="275" y="120" width="30" height="130" class="bar" />
    <rect x="325" y="180" width="30" height="70" class="bar" />
    <rect x="375" y="200" width="30" height="50" class="bar" />
    <rect x="425" y="130" width="30" height="120" class="bar" />
    <rect x="475" y="90" width="30" height="160" class="bar" />
    
    <!-- Scatter plot points -->
    <circle cx="90" cy="120" r="3" fill="#ff6b6b" />
    <circle cx="140" cy="90" r="3" fill="#ff6b6b" />
    <circle cx="190" cy="150" r="3" fill="#ff6b6b" />
    <circle cx="240" cy="70" r="3" fill="#ff6b6b" />
    <circle cx="290" cy="100" r="3" fill="#ff6b6b" />
    <circle cx="340" cy="160" r="3" fill="#ff6b6b" />
    <circle cx="390" cy="180" r="3" fill="#ff6b6b" />
    <circle cx="440" cy="110" r="3" fill="#ff6b6b" />
    <circle cx="490" cy="80" r="3" fill="#ff6b6b" />
    
    <!-- Axis labels -->
    <text x="300" y="280" class="axis-label" text-anchor="middle">Features</text>
    <text x="30" y="150" class="axis-label" text-anchor="middle" transform="rotate(-90,30,150)">Values</text>
</svg>
"""

# Confusion matrix visualization template
CONFUSION_MATRIX_TEMPLATE = """
<svg width="300" height="300" xmlns="http://www.w3.org/2000/svg">
    <style>
        .title { font: bold 16px sans-serif; fill: #333; }
        .label { font: 12px sans-serif; fill: #333; }
        .value { font: bold 14px sans-serif; fill: white; }
        .tp { fill: #4a86e8; }
        .tn { fill: #6aa84f; }
        .fp { fill: #e69138; }
        .fn { fill: #cc0000; }
        .border { fill: none; stroke: #ccc; stroke-width: 1; }
    </style>
    
    <text x="150" y="30" class="title" text-anchor="middle">Confusion Matrix</text>
    
    <!-- Headers -->
    <text x="180" y="60" class="label" text-anchor="middle">Predicted</text>
    <text x="60" y="180" class="label" text-anchor="middle" transform="rotate(-90,60,180)">Actual</text>
    
    <text x="130" y="80" class="label" text-anchor="middle">Negative</text>
    <text x="230" y="80" class="label" text-anchor="middle">Positive</text>
    
    <text x="80" y="130" class="label" text-anchor="middle">Negative</text>
    <text x="80" y="230" class="label" text-anchor="middle">Positive</text>
    
    <!-- Cells -->
    <rect x="100" y="100" width="60" height="60" class="tn" rx="5" ry="5" />
    <rect x="200" y="100" width="60" height="60" class="fp" rx="5" ry="5" />
    <rect x="100" y="200" width="60" height="60" class="fn" rx="5" ry="5" />
    <rect x="200" y="200" width="60" height="60" class="tp" rx="5" ry="5" />
    
    <!-- Values (to be replaced programmatically) -->
    <text x="130" y="140" class="value" text-anchor="middle">TN</text>
    <text x="230" y="140" class="value" text-anchor="middle">FP</text>
    <text x="130" y="240" class="value" text-anchor="middle">FN</text>
    <text x="230" y="240" class="value" text-anchor="middle">TP</text>
    
    <!-- Border -->
    <rect x="100" y="100" width="160" height="160" class="border" />
    <line x1="100" y1="160" x2="260" y2="160" class="border" />
    <line x1="160" y1="100" x2="160" y2="260" class="border" />
</svg>
"""
