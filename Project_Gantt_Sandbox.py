import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
from datetime import datetime, date, timedelta
import random
import json
import math
import os
import re

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import cm  # for colormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path
import matplotlib.patches as mpatches

from tkcalendar import DateEntry  # calendar widget

DATE_FORMAT = "%Y-%m-%d"  # expected format for dates

# Help files directory - can be customized
HELP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "help")


# ----------------------------------------------------------------------
# LINE INTERSECTION AND JUMP HELPER FUNCTIONS
# ----------------------------------------------------------------------

def line_segment_intersection(p1, p2, p3, p4):
    """
    Find intersection point of two line segments (p1-p2) and (p3-p4).
    Returns (x, y) if they intersect, None otherwise.

    Uses parametric form to find intersection.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None  # Parallel or coincident

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    # Check if intersection is within both segments (with small margin)
    margin = 0.01
    if margin < t < (1 - margin) and margin < u < (1 - margin):
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)

    return None


def get_segments_from_path(start, waypoints, end):
    """
    Extract line segments from a connector path.
    Returns list of ((x1,y1), (x2,y2)) tuples.
    """
    segments = []
    points = [start] + [tuple(wp) for wp in waypoints] + [end]
    for i in range(len(points) - 1):
        segments.append((points[i], points[i + 1]))
    return segments


def is_horizontal_segment(seg):
    """Check if a segment is horizontal (or nearly so)."""
    (x1, y1), (x2, y2) = seg
    return abs(y2 - y1) < abs(x2 - x1)


def build_path_with_jumps(start, waypoints, end, intersections, jump_width, jump_height):
    """
    Build a matplotlib Path that includes arc jumps at intersection points.

    intersections: list of (x, y) points where this path crosses others
    jump_width: horizontal extent of the jump arc
    jump_height: vertical extent of the jump arc

    Returns: (vertices, codes) for matplotlib Path
    """
    points = [start] + [tuple(wp) for wp in waypoints] + [end]

    # Collect all segments with their intersections
    segment_intersections = []
    for i in range(len(points) - 1):
        seg_start = points[i]
        seg_end = points[i + 1]

        # Find intersections on this segment
        seg_ints = []
        for ix, iy in intersections:
            # Check if intersection is on this segment
            if is_point_on_segment(seg_start, seg_end, (ix, iy)):
                # Calculate parameter t along segment
                dx = seg_end[0] - seg_start[0]
                dy = seg_end[1] - seg_start[1]
                if abs(dx) > abs(dy):
                    t = (ix - seg_start[0]) / dx if abs(dx) > 1e-10 else 0
                else:
                    t = (iy - seg_start[1]) / dy if abs(dy) > 1e-10 else 0
                seg_ints.append((t, ix, iy))

        # Sort intersections by parameter t
        seg_ints.sort(key=lambda x: x[0])
        segment_intersections.append((seg_start, seg_end, seg_ints))

    # Build path with jumps
    verts = [start]
    codes = [Path.MOVETO]

    for seg_start, seg_end, seg_ints in segment_intersections:
        if not seg_ints:
            # No intersections - straight line to end
            verts.append(seg_end)
            codes.append(Path.LINETO)
        else:
            # Determine if segment is horizontal or vertical
            dx = seg_end[0] - seg_start[0]
            dy = seg_end[1] - seg_start[1]
            is_horiz = abs(dx) > abs(dy)

            current = seg_start
            for t, ix, iy in seg_ints:
                if is_horiz:
                    # Horizontal segment - NO jump, just continue straight through
                    # (the vertical segment will do the jumping)
                    pass
                else:
                    # Vertical segment - jump goes right
                    # Determine direction of travel
                    going_down = dy > 0
                    if going_down:
                        before_jump = (ix, iy - jump_height)
                        after_jump = (ix, iy + jump_height)
                    else:
                        before_jump = (ix, iy + jump_height)
                        after_jump = (ix, iy - jump_height)
                    # Arc control points (semicircle going right)
                    arc_ctrl1 = (ix + jump_width, iy - jump_height * 0.55 * (1 if going_down else -1))
                    arc_ctrl2 = (ix + jump_width, iy + jump_height * 0.55 * (1 if going_down else -1))

                    # Line to before jump
                    verts.append(before_jump)
                    codes.append(Path.LINETO)

                    # Bezier curve for the jump arc
                    verts.append(arc_ctrl1)
                    codes.append(Path.CURVE4)
                    verts.append(arc_ctrl2)
                    codes.append(Path.CURVE4)
                    verts.append(after_jump)
                    codes.append(Path.CURVE4)

                    current = after_jump

            # Line to segment end
            verts.append(seg_end)
            codes.append(Path.LINETO)

    return verts, codes


def is_point_on_segment(seg_start, seg_end, point, tolerance=0.01):
    """Check if a point lies on a line segment within tolerance."""
    x1, y1 = seg_start
    x2, y2 = seg_end
    px, py = point

    # Check if point is within bounding box
    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)

    # Expand bounding box by tolerance
    if not (min_x - tolerance <= px <= max_x + tolerance and
            min_y - tolerance <= py <= max_y + tolerance):
        return False

    # Check distance from line
    seg_len = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if seg_len < 1e-10:
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2) < tolerance

    # Distance from point to line
    dist = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / seg_len
    return dist < tolerance


# ----------------------------------------------------------------------
# DISTRIBUTION SAMPLING FUNCTIONS
# ----------------------------------------------------------------------

def sample_triangular(min_val, mode, max_val):
    """Sample from a triangular distribution."""
    return random.triangular(min_val, max_val, mode)


def sample_betapert(min_val, mode, max_val, lambd=4.0):
    """
    Sample from a BetaPERT distribution.

    BetaPERT is a modified beta distribution commonly used in schedule risk analysis.
    The lambda parameter controls the weight given to the mode (default=4).
    Higher lambda = tighter distribution around the mode.
    """
    if max_val <= min_val:
        return mode

    # Calculate mean using PERT formula
    mu = (min_val + lambd * mode + max_val) / (lambd + 2)

    # Ensure mu is within bounds
    if mu <= min_val:
        mu = min_val + 0.001 * (max_val - min_val)
    if mu >= max_val:
        mu = max_val - 0.001 * (max_val - min_val)

    # Calculate alpha and beta parameters
    if mode == mu:
        # Special case: symmetric
        alpha = lambd / 2 + 1
        beta = alpha
    else:
        alpha = ((mu - min_val) * (2 * mode - min_val - max_val)) / \
                ((mode - mu) * (max_val - min_val))
        beta = alpha * (max_val - mu) / (mu - min_val)

    # Ensure valid parameters
    if alpha <= 0 or beta <= 0:
        # Fallback to triangular if parameters are invalid
        return sample_triangular(min_val, mode, max_val)

    # Sample from beta and transform to [min, max]
    x = random.betavariate(alpha, beta)
    return min_val + x * (max_val - min_val)


def compute_distribution_params(base_duration, prob, impact, fail_mode="late"):
    """
    Compute triangular/BetaPERT distribution parameters from I×P scores.

    Calibrated Rubric:
    - Probability (P) → base spread (how uncertain/volatile)
      P=0.0 → base=0.10 (rare slip, ±5%)
      P=0.25 → base=0.20 (low likelihood, ±10%)
      P=0.50 → base=0.30 (moderate uncertainty, ±20%)
      P=0.75 → base=0.50 (significant uncertainty, ±30-40%)
      P=1.0 → base=0.80 (volatile/unstable, fat tail)

    - Impact (I) → scale factor on spread
      I=0.0 → scale=0.50 (minimal disruption)
      I=0.25 → scale=0.75 (low impact)
      I=0.50 → scale=1.00 (moderate impact)
      I=0.75 → scale=1.25 (high impact)
      I=1.0 → scale=1.50 (severe impact)

    - Effective spread r = base(P) × scale(I)

    Early-fail vs Late-fail:
    - Early-fail: More symmetric, smaller right tail
    - Late-fail: Right-skewed, larger right tail (fat-tailed)

    Returns: (min_duration, mode_duration, max_duration, distribution_type)
    """
    # Map probability to base spread (linear interpolation)
    # P=0 → 0.10, P=1 → 0.80
    base_spread = 0.10 + prob * 0.70

    # Map impact to scale factor
    # I=0 → 0.50, I=1 → 1.50
    scale_factor = 0.50 + impact * 1.00

    # Effective spread
    r = base_spread * scale_factor

    # Calculate distribution parameters
    mode = base_duration

    if fail_mode == "early":
        # Early-fail: More symmetric, uncertainty front-loaded
        # Smaller right tail, larger left potential
        min_dur = mode * (1 - 0.4 * r)  # Can be somewhat faster
        max_dur = mode * (1 + 0.6 * r)  # Moderate extension
        dist_type = "triangular"
    else:
        # Late-fail: Right-skewed, fat right tail
        # Problems emerge late, causing large extensions
        min_dur = mode * (1 - 0.2 * r)  # Small chance of being faster
        max_dur = mode * (1 + 1.2 * r)  # Can extend significantly
        dist_type = "betapert"

    # Ensure valid bounds
    min_dur = max(0.1, min_dur)  # At least 0.1 days
    if max_dur <= min_dur:
        max_dur = min_dur + 0.1
    if mode < min_dur:
        mode = min_dur
    if mode > max_dur:
        mode = max_dur

    return min_dur, mode, max_dur, dist_type


def sample_task_duration(base_duration, prob, impact, fail_mode="late"):
    """
    Sample a task duration from the appropriate distribution.

    This is the core function that implements the I×P → distribution mapping.
    """
    min_dur, mode, max_dur, dist_type = compute_distribution_params(
        base_duration, prob, impact, fail_mode
    )

    if dist_type == "betapert":
        # Use higher lambda for low probability (tighter around mode)
        # Use lower lambda for high probability (wider spread)
        lambd = 6.0 - 4.0 * prob  # lambd ranges from 6 (tight) to 2 (wide)
        return sample_betapert(min_dur, mode, max_dur, lambd)
    else:
        return sample_triangular(min_dur, mode, max_dur)


def compute_correlation(x_values, y_values):
    """
    Compute Pearson correlation coefficient between two lists.

    Used to calculate Schedule Sensitivity Index (SSI).
    """
    n = len(x_values)
    if n < 2:
        return 0.0

    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n

    # Compute covariance and standard deviations
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values)) / n
    var_x = sum((x - mean_x) ** 2 for x in x_values) / n
    var_y = sum((y - mean_y) ** 2 for y in y_values) / n

    if var_x < 1e-10 or var_y < 1e-10:
        return 0.0

    return cov_xy / (math.sqrt(var_x) * math.sqrt(var_y))


class GanttApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project Gantt Chart with Dependencies & Risk")

        # Data: list of dicts:
        # {"name", "start", "end", "deps", "prob", "impact", "fail_mode"}
        self.tasks = []

        # Monte Carlo results cache
        # {"criticality": {task_name: 0..1}, "ssi": {task_name: 0..1}, "finish_stats": {...}}
        self.mc_results = None

        # Index of selected task in self.tasks (for editing)
        self.selected_task_index = None

        # Current file path for save operations
        self.current_file = None

        # Task ordering mode: True = use list order, False = auto-sort by category
        self.use_manual_order = False

        # Connector customization state
        # Key: (from_task_name, to_task_name), Value: {"ctrl1": (x, y), "ctrl2": (x, y)}
        self.connector_waypoints = {}

        # Interactive connector editing state
        self.connector_edit_mode = False
        self.snap_to_grid = True  # Snap waypoints to grid when dragging
        self.dragging_point = None  # (connector_key, point_name, artist)
        self.control_point_artists = []  # List of control point circle artists
        self.connector_artists = {}  # Key: connector_key, Value: path artist

        # Mouse event connection IDs
        self._cid_press = None
        self._cid_release = None
        self._cid_motion = None

        # Build the menu bar
        self._build_menu()

        # Build the UI
        self._build_ui()

        # Populate with initial sample tasks
        self.load_sample_data()
        # Initial MC summary text
        self.update_mc_summary(
            "No Monte Carlo results yet.\nClick 'Run Monte Carlo' to analyze risk."
        )

    # ------------------------------------------------------------------
    # MENU BAR
    # ------------------------------------------------------------------
    def _build_menu(self):
        """Create the application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Project", command=self.new_project, accelerator="Ctrl+N")
        file_menu.add_command(label="Open...", command=self.open_project, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Save", command=self.save_project, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As...", command=self.save_project_as, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Load Sample Data", command=self.load_sample_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.exit_app, accelerator="Alt+F4")

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Add Task", command=self.add_task)
        edit_menu.add_command(label="Update Selected Task", command=self.update_task)
        edit_menu.add_command(label="Delete Selected Task", command=self.delete_task, accelerator="Del")
        edit_menu.add_separator()
        edit_menu.add_command(label="Clear Form", command=self.clear_form)
        edit_menu.add_command(label="Clear All Tasks", command=self.clear_all_tasks)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Refresh Gantt Chart", command=self.plot_gantt)
        view_menu.add_command(label="Run Monte Carlo...", command=self.run_monte_carlo_ui)
        view_menu.add_separator()
        view_menu.add_command(label="Toggle Connector Edit Mode", command=self.toggle_connector_edit_mode,
                              accelerator="Ctrl+E")
        view_menu.add_command(label="Toggle Snap to Grid", command=self.toggle_snap_to_grid, accelerator="Ctrl+G")
        view_menu.add_command(label="Reset All Connectors", command=self.reset_connectors)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Risk Assessment Guide", command=self.show_risk_help, accelerator="F1")
        help_menu.add_command(label="Distribution Guide", command=self.show_distribution_help)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about)

        # Keyboard shortcuts
        self.root.bind("<Control-n>", lambda e: self.new_project())
        self.root.bind("<Control-o>", lambda e: self.open_project())
        self.root.bind("<Control-s>", lambda e: self.save_project())
        self.root.bind("<Control-S>", lambda e: self.save_project_as())
        self.root.bind("<Control-e>", lambda e: self.toggle_connector_edit_mode())
        self.root.bind("<Control-E>", lambda e: self.toggle_connector_edit_mode())
        self.root.bind("<Control-g>", lambda e: self.toggle_snap_to_grid())
        self.root.bind("<Control-G>", lambda e: self.toggle_snap_to_grid())
        self.root.bind("<F1>", lambda e: self.show_risk_help())
        self.root.bind("<Delete>", lambda e: self.delete_task())

        # Task reordering shortcuts
        self.root.bind("<Control-Up>", lambda e: self.move_task_up())
        self.root.bind("<Control-Down>", lambda e: self.move_task_down())

    # ------------------------------------------------------------------
    # FILE OPERATIONS
    # ------------------------------------------------------------------
    def new_project(self):
        """Create a new empty project."""
        if self.tasks:
            response = messagebox.askyesnocancel(
                "New Project",
                "Do you want to save the current project before creating a new one?"
            )
            if response is None:  # Cancel
                return
            if response:  # Yes, save first
                self.save_project()

        self.tasks.clear()
        self.mc_results = None
        self.current_file = None
        self.selected_task_index = None
        self.connector_waypoints = {}  # Clear custom connector positions
        self.connector_edit_mode = False  # Reset edit mode
        self._disconnect_mouse_events()  # Disconnect any event handlers
        self.root.title("Project Gantt Chart with Dependencies & Risk - New Project")
        self.refresh_deps_listbox()
        self.refresh_task_tree()
        self.clear_form()
        self.update_mc_summary(
            "No Monte Carlo results yet.\nClick 'Run Monte Carlo' to analyze risk."
        )
        # Clear the plot
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("#F8FAFC")
        self.figure.patch.set_facecolor("#F8FAFC")
        self.ax.text(
            0.5, 0.5,
            "Add tasks and click 'Plot / Refresh Gantt'",
            ha="center", va="center", transform=self.ax.transAxes
        )
        self.ax.axis("off")
        self.canvas.draw()

    def open_project(self):
        """Open a project from a JSON file."""
        filepath = filedialog.askopenfilename(
            title="Open Project",
            filetypes=[("Gantt Project Files", "*.gantt"), ("JSON Files", "*.json"), ("All Files", "*.*")],
            defaultextension=".gantt"
        )
        if not filepath:
            return

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Parse tasks from JSON
            self.tasks = []
            for t in data.get("tasks", []):
                self.tasks.append({
                    "name": t["name"],
                    "start": datetime.strptime(t["start"], DATE_FORMAT),
                    "end": datetime.strptime(t["end"], DATE_FORMAT),
                    "deps": t.get("deps", []),
                    "prob": t.get("prob", 0.0),
                    "impact": t.get("impact", 0.0),
                    "fail_mode": t.get("fail_mode", "late"),
                })

            # Load connector waypoints if present
            self.connector_waypoints = {}
            waypoints_json = data.get("connector_waypoints", {})
            for key, points in waypoints_json.items():
                # Convert string key back to tuple
                parts = key.split("|")
                if len(parts) == 2:
                    connector_key = (parts[0], parts[1])
                    self.connector_waypoints[connector_key] = points

            # Load manual order setting
            self.use_manual_order = data.get("use_manual_order", False)
            self.manual_order_var.set(self.use_manual_order)

            self.current_file = filepath
            self.mc_results = None
            self.selected_task_index = None
            self.connector_edit_mode = False  # Reset edit mode on load
            self.root.title(f"Project Gantt Chart - {filepath}")
            self.refresh_deps_listbox()
            self.refresh_task_tree()
            self.clear_form()
            self.update_mc_summary(
                "Project loaded.\nClick 'Run Monte Carlo' to analyze risk."
            )
            if self.tasks:
                self.plot_gantt()

        except Exception as e:
            messagebox.showerror("Error Opening File", f"Could not open file:\n{e}")

    def save_project(self):
        """Save the project to the current file, or prompt for location."""
        if self.current_file:
            self._save_to_file(self.current_file)
        else:
            self.save_project_as()

    def save_project_as(self):
        """Save the project to a new file location."""
        filepath = filedialog.asksaveasfilename(
            title="Save Project As",
            filetypes=[("Gantt Project Files", "*.gantt"), ("JSON Files", "*.json"), ("All Files", "*.*")],
            defaultextension=".gantt"
        )
        if not filepath:
            return

        self._save_to_file(filepath)
        self.current_file = filepath
        self.root.title(f"Project Gantt Chart - {filepath}")

    def _save_to_file(self, filepath):
        """Internal method to save tasks to a JSON file."""
        try:
            # Convert connector waypoints keys from tuples to strings for JSON
            waypoints_json = {}
            for (from_task, to_task), points in self.connector_waypoints.items():
                key = f"{from_task}|{to_task}"
                waypoints_json[key] = points

            data = {
                "tasks": [
                    {
                        "name": t["name"],
                        "start": t["start"].strftime(DATE_FORMAT),
                        "end": t["end"].strftime(DATE_FORMAT),
                        "deps": t["deps"],
                        "prob": t["prob"],
                        "impact": t["impact"],
                        "fail_mode": t.get("fail_mode", "late"),
                    }
                    for t in self.tasks
                ],
                "connector_waypoints": waypoints_json,
                "use_manual_order": self.use_manual_order
            }
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            messagebox.showinfo("Saved", f"Project saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error Saving File", f"Could not save file:\n{e}")

    def clear_all_tasks(self):
        """Clear all tasks after confirmation."""
        if not self.tasks:
            messagebox.showinfo("Clear All", "No tasks to clear.")
            return

        response = messagebox.askyesno(
            "Clear All Tasks",
            "Are you sure you want to delete all tasks?\nThis cannot be undone."
        )
        if response:
            self.tasks.clear()
            self.mc_results = None
            self.selected_task_index = None
            self.refresh_deps_listbox()
            self.refresh_task_tree()
            self.clear_form()
            self.update_mc_summary(
                "All tasks cleared.\nAdd tasks and run Monte Carlo."
            )

    def exit_app(self):
        """Exit the application with confirmation if unsaved changes."""
        if self.tasks and not self.current_file:
            response = messagebox.askyesnocancel(
                "Exit",
                "You have unsaved changes. Save before exiting?"
            )
            if response is None:  # Cancel
                return
            if response:  # Yes, save first
                self.save_project()
        self.root.quit()

    def _load_help_file(self, filename):
        """Load help text from an external file."""
        filepath = os.path.join(HELP_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return None
        except Exception as e:
            print(f"Error loading help file {filename}: {e}")
            return None

    def _load_help_json(self, filename):
        """Load help data from a JSON file."""
        filepath = os.path.join(HELP_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except Exception as e:
            print(f"Error loading help JSON {filename}: {e}")
            return None

    def show_about(self):
        """Show the About dialog."""
        # Try to load from external file
        about_text = self._load_help_file("about.txt")

        if about_text is None:
            # Fallback to embedded text
            about_text = (
                "Project Gantt Chart with Dependencies & Risk\n"
                "Version 2.0 - Risk-Driven Timeline Architecture\n\n"
                "A project scheduling tool featuring:\n"
                "• Gantt chart visualization\n"
                "• Task dependencies\n"
                "• Calibrated I×P → Distribution mapping\n"
                "• Early-fail vs Late-fail modeling\n"
                "• Monte Carlo risk simulation\n"
                "• Criticality Index (CI)\n"
                "• Schedule Sensitivity Index (SSI)\n\n"
                "Use probability and impact values to model\n"
                "schedule risk and identify critical tasks."
            )

        messagebox.showinfo("About", about_text)

    def show_distribution_help(self):
        """Show help dialog explaining the distribution methodology."""
        # Try to load from external file
        help_text = self._load_help_file("distribution_guide.txt")

        if help_text is None:
            # Fallback to embedded text
            help_text = """Distribution Methodology Guide

This tool uses a calibrated I×P → Distribution rubric to convert 
risk scores into realistic duration distributions.

PROBABILITY (0-1) → Distribution Shape
• 0.0-0.2: Narrow spread (±5-10%), rare slippage
• 0.2-0.4: Low spread (±10-20%), unlikely delays
• 0.4-0.6: Moderate spread (±20-30%), possible delays
• 0.6-0.8: Wide spread (±30-50%), likely delays
• 0.8-1.0: Fat-tailed (±50-80%+), volatile/unstable

IMPACT (0-1) → Distribution Scale
• 0.0-0.2: Minimal (0.5× base spread)
• 0.2-0.4: Low (0.75× base spread)
• 0.4-0.6: Moderate (1.0× base spread)
• 0.6-0.8: High (1.25× base spread)
• 0.8-1.0: Severe (1.5× base spread)

FAIL MODE (Early vs Late)
Early-Fail:
• Problems surface quickly at task start
• More symmetric distribution
• Smaller right tail
• Examples: Material procurement, internal design

Late-Fail:
• Work appears on-track until near completion
• Right-skewed (BetaPERT) distribution
• Large right tail - can extend significantly
• Examples: Supplier qualification, system integration,
  regulatory testing, validation

METRICS COMPUTED
• Criticality Index (CI): % of simulations where task 
  is on the critical path
• Schedule Sensitivity Index (SSI): Correlation between
  task duration variation and project completion variation
"""

        help_window = tk.Toplevel(self.root)
        help_window.title("Distribution Methodology Guide")
        help_window.geometry("600x650")
        help_window.configure(bg="#E2E8F0")

        # Add scrollbar for longer content
        frame = tk.Frame(help_window, bg="#E2E8F0")
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text_widget = tk.Text(frame, wrap=tk.WORD, padx=10, pady=10,
                              bg="#E2E8F0", fg="black", font=("Consolas", 10),
                              yscrollcommand=scrollbar.set)
        text_widget.insert("1.0", help_text)
        text_widget.config(state="disabled")
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=text_widget.yview)

        close_btn = tk.Button(help_window, text="Close",
                              command=help_window.destroy,
                              bg="#E2E8F0", padx=20, pady=5)
        close_btn.pack(pady=10)

    # ------------------------------------------------------------------
    # SAMPLE DATA
    # ------------------------------------------------------------------
    def load_sample_data(self):
        """
        Populate the app with a realistic product development scenario.

        Three interacting critical paths:
        - Technical Path (D): D1 Concept → D2 Detailed Design → D3 Design Verification
        - Supplier Path (S): S1 Quote → S2 Sample Parts → S3 PPAP Qualification
        - Manufacturing Path (M): M1 Process Design → M2 Assessment Builds → M3 Production Ramp

        Cross-path dependencies create a complex network:
        - S1 depends on D1 (need concept specs for supplier quotes)
        - S2 depends on D2, S1 (need detailed design and quotes for tooling)
        - S3 depends on D3, S2 (need DVT complete and sample parts for PPAP)
        - M1 depends on D2 (need detailed design for process engineering)
        - M2 depends on M1, S2 (need process design and sample parts)
        - M3 depends on M2, S3 (need assessment builds and PPAP complete)

        I×P Scale Mapping (1-5 to 0-1):
        P: 1→0.10, 2→0.25, 3→0.50, 4→0.75, 5→0.90
        I: 1→0.10, 2→0.25, 3→0.50, 4→0.75, 5→1.00
        """
        self.tasks.clear()

        today = date.today()

        def dt(offset_days):
            return datetime.combine(
                today + timedelta(days=offset_days), datetime.min.time()
            )

        # =================================================================
        # TECHNICAL CRITICAL PATH (D)
        # =================================================================
        # D1: Concept Development - 20 days, P=1, I=2
        # Routine internal kickoff, early-fail (narrow triangular)
        d1_start = 0
        d1_end = d1_start + 20

        # D2: Detailed Design - 40 days, P=2, I=3
        # Internal iteration, early-fail (triangular)
        d2_start = d1_end
        d2_end = d2_start + 40

        # D3: Design Verification Test - 30 days, P=3, I=4
        # May uncover issues requiring design changes, late-fail (BetaPERT)
        d3_start = d2_end
        d3_end = d3_start + 30

        # =================================================================
        # SUPPLIER CRITICAL PATH (S)
        # =================================================================
        # S1: Supplier Quote - 15 days, P=2, I=2
        # Routine supplier response, early-fail (triangular)
        # Depends on D1 (need concept specs)
        s1_start = d1_end
        s1_end = s1_start + 15

        # S2: Sample Parts - 45 days, P=4, I=4
        # Tooling fabrication, late-fail (skewed BetaPERT)
        # Depends on S1 and D2
        s2_start = max(s1_end, d2_end)
        s2_end = s2_start + 45

        # S3: PPAP Qualification - 30 days, P=5, I=5
        # First-time qualification, severe late-fail (heavy tail)
        # Depends on S2 and D3
        s3_start = max(s2_end, d3_end)
        s3_end = s3_start + 30

        # =================================================================
        # MANUFACTURING CRITICAL PATH (M)
        # =================================================================
        # M1: Process Design - 25 days, P=2, I=3
        # Internal manufacturing engineering, early-fail (triangular)
        # Depends on D2
        m1_start = d2_end
        m1_end = m1_start + 25

        # M2: Assessment Builds - 20 days, P=4, I=4
        # Depends on part quality, late-fail (skewed BetaPERT)
        # Depends on M1 and S2
        m2_start = max(m1_end, s2_end)
        m2_end = m2_start + 20

        # M3: Production Ramp - 30 days, P=3, I=3
        # Learning curve, moderate uncertainty, late-fail (BetaPERT)
        # Depends on M2 and S3 (final gate)
        m3_start = max(m2_end, s3_end)
        m3_end = m3_start + 30

        self.tasks = [
            # --- TECHNICAL PATH ---
            {
                "name": "D1_Concept",
                "start": dt(d1_start),
                "end": dt(d1_end),
                "deps": [],
                "prob": 0.10,  # P=1: Rare slip
                "impact": 0.25,  # I=2: Minor impact
                "fail_mode": "early",
            },
            {
                "name": "D2_DetailDesign",
                "start": dt(d2_start),
                "end": dt(d2_end),
                "deps": ["D1_Concept"],
                "prob": 0.25,  # P=2: Low likelihood
                "impact": 0.50,  # I=3: Moderate impact
                "fail_mode": "early",
            },
            {
                "name": "D3_DVT",
                "start": dt(d3_start),
                "end": dt(d3_end),
                "deps": ["D2_DetailDesign"],
                "prob": 0.50,  # P=3: Possible
                "impact": 0.75,  # I=4: High impact
                "fail_mode": "late",  # Testing may uncover issues late
            },
            # --- SUPPLIER PATH ---
            {
                "name": "S1_Quote",
                "start": dt(s1_start),
                "end": dt(s1_end),
                "deps": ["D1_Concept"],
                "prob": 0.25,  # P=2: Low likelihood
                "impact": 0.25,  # I=2: Minor impact
                "fail_mode": "early",
            },
            {
                "name": "S2_SampleParts",
                "start": dt(s2_start),
                "end": dt(s2_end),
                "deps": ["S1_Quote", "D2_DetailDesign"],
                "prob": 0.75,  # P=4: Likely to slip
                "impact": 0.75,  # I=4: High impact
                "fail_mode": "late",  # Tooling issues emerge late
            },
            {
                "name": "S3_PPAP",
                "start": dt(s3_start),
                "end": dt(s3_end),
                "deps": ["S2_SampleParts", "D3_DVT"],
                "prob": 0.90,  # P=5: Almost certain issues
                "impact": 1.00,  # I=5: Severe impact
                "fail_mode": "late",  # First-time qual = severe late-fail
            },
            # --- MANUFACTURING PATH ---
            {
                "name": "M1_ProcessDesign",
                "start": dt(m1_start),
                "end": dt(m1_end),
                "deps": ["D2_DetailDesign"],
                "prob": 0.25,  # P=2: Low likelihood
                "impact": 0.50,  # I=3: Moderate impact
                "fail_mode": "early",
            },
            {
                "name": "M2_AssessBuild",
                "start": dt(m2_start),
                "end": dt(m2_end),
                "deps": ["M1_ProcessDesign", "S2_SampleParts"],
                "prob": 0.75,  # P=4: Likely to slip
                "impact": 0.75,  # I=4: High impact
                "fail_mode": "late",  # Part quality issues emerge late
            },
            {
                "name": "M3_ProdRamp",
                "start": dt(m3_start),
                "end": dt(m3_end),
                "deps": ["M2_AssessBuild", "S3_PPAP"],
                "prob": 0.50,  # P=3: Possible
                "impact": 0.50,  # I=3: Moderate impact
                "fail_mode": "late",  # Learning curve uncertainty
            },
        ]

        self.current_file = None
        self.mc_results = None
        self.root.title("Project Gantt Chart - Product Development Scenario")
        self.refresh_deps_listbox()
        self.refresh_task_tree()
        self.plot_gantt()

    # ------------------------------------------------------------------
    # UI BUILDING
    # ------------------------------------------------------------------
    def _build_ui(self):
        # Color scheme - refined palette
        SLATE_GREY = "#64748B"  # Overall background (lighter, modern)
        LIGHT_GRAY = "#E2E8F0"  # Panel/frame backgrounds (blue-tinted)
        DARK_SLATE = "#475569"  # Entry field backgrounds (darker for contrast)
        CHART_BG = "#F8FAFC"  # Chart background (near white)

        # Store for use elsewhere
        self.colors = {
            'slate_grey': SLATE_GREY,
            'light_gray': LIGHT_GRAY,
            'dark_slate': DARK_SLATE,
            'chart_bg': CHART_BG,
        }

        # Configure root window background
        self.root.configure(bg=SLATE_GREY)

        # Configure ttk styles
        style = ttk.Style()
        style.configure("TFrame", background=SLATE_GREY)
        style.configure("TLabelframe", background=LIGHT_GRAY)
        style.configure("TLabelframe.Label", background=LIGHT_GRAY, foreground="black")
        style.configure("TLabel", background=LIGHT_GRAY, foreground="black")
        style.configure("TButton", background=LIGHT_GRAY)
        style.configure("TPanedwindow", background=SLATE_GREY)

        # Configure Treeview style
        style.configure("Treeview",
                        background=LIGHT_GRAY,
                        fieldbackground=LIGHT_GRAY,
                        foreground="black")
        style.configure("Treeview.Heading", background=LIGHT_GRAY, foreground="black")

        # Main paned window: left = controls & list, right = chart
        main_pane = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(main_pane, padding=5)
        right_frame = ttk.Frame(main_pane, padding=5)
        main_pane.add(left_frame, weight=1)
        main_pane.add(right_frame, weight=3)

        # ----------------- LEFT SIDE: FORM + BUTTONS + TASK LIST -----------------
        form_frame = ttk.LabelFrame(left_frame, text="Add / Edit Task")
        form_frame.pack(fill=tk.X, padx=5, pady=5)

        # Task Name
        ttk.Label(form_frame, text="Task Name:").grid(
            row=0, column=0, sticky="e", padx=2, pady=2
        )
        self.name_entry = tk.Entry(form_frame, width=25, bg="white", fg="black", insertbackground="black")
        self.name_entry.grid(
            row=0, column=1, columnspan=2, sticky="we", padx=2, pady=2
        )

        # Start Date (with calendar)
        ttk.Label(form_frame, text="Start Date:").grid(
            row=1, column=0, sticky="e", padx=2, pady=2
        )
        self.start_entry = DateEntry(
            form_frame,
            width=12,
            date_pattern="yyyy-mm-dd",
            background=DARK_SLATE,
            fieldbackground=DARK_SLATE,
            foreground="white",
        )
        self.start_entry.grid(row=1, column=1, sticky="w", padx=2, pady=2)

        # End Date (with calendar)
        ttk.Label(form_frame, text="End Date:").grid(
            row=2, column=0, sticky="e", padx=2, pady=2
        )
        self.end_entry = DateEntry(
            form_frame,
            width=12,
            date_pattern="yyyy-mm-dd",
            background=DARK_SLATE,
            fieldbackground=DARK_SLATE,
            foreground="white",
        )
        self.end_entry.grid(row=2, column=1, sticky="w", padx=2, pady=2)

        # Probability
        ttk.Label(form_frame, text="Probability (0–1):").grid(
            row=3, column=0, sticky="e", padx=2, pady=2
        )
        self.prob_entry = tk.Entry(form_frame, width=8, bg="white", fg="black", insertbackground="black")
        self.prob_entry.insert(0, "0.0")
        self.prob_entry.grid(row=3, column=1, sticky="w", padx=2, pady=2)

        # Impact
        ttk.Label(form_frame, text="Impact (0–1):").grid(
            row=4, column=0, sticky="e", padx=2, pady=2
        )
        self.impact_entry = tk.Entry(form_frame, width=8, bg="white", fg="black", insertbackground="black")
        self.impact_entry.insert(0, "0.0")
        self.impact_entry.grid(row=4, column=1, sticky="w", padx=2, pady=2)

        # Fail Mode (Early vs Late)
        ttk.Label(form_frame, text="Fail Mode:").grid(
            row=5, column=0, sticky="e", padx=2, pady=2
        )
        self.fail_mode_var = tk.StringVar(value="late")
        fail_mode_frame = tk.Frame(form_frame, bg=LIGHT_GRAY)
        fail_mode_frame.grid(row=5, column=1, columnspan=2, sticky="w", padx=2, pady=2)

        tk.Radiobutton(fail_mode_frame, text="Early", variable=self.fail_mode_var,
                       value="early", bg=LIGHT_GRAY, fg="black",
                       activebackground=LIGHT_GRAY).pack(side=tk.LEFT)
        tk.Radiobutton(fail_mode_frame, text="Late", variable=self.fail_mode_var,
                       value="late", bg=LIGHT_GRAY, fg="black",
                       activebackground=LIGHT_GRAY).pack(side=tk.LEFT)

        # Dependencies (multi-select listbox with existing tasks)
        deps_frame = ttk.LabelFrame(
            form_frame, text="Dependencies (tasks this depends on)"
        )
        deps_frame.grid(row=6, column=0, columnspan=3, sticky="nsew", padx=2, pady=4)

        self.deps_listbox = tk.Listbox(
            deps_frame, selectmode=tk.MULTIPLE, height=5, exportselection=False,
            bg=LIGHT_GRAY, fg="black", selectbackground=DARK_SLATE, selectforeground="white"
        )
        self.deps_listbox.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Resize behavior inside form
        form_frame.columnconfigure(1, weight=1)

        # Buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        add_button = ttk.Button(button_frame, text="Add Task", command=self.add_task)
        add_button.pack(side=tk.LEFT, padx=2)

        update_button = ttk.Button(
            button_frame, text="Update Selected", command=self.update_task
        )
        update_button.pack(side=tk.LEFT, padx=2)

        delete_button = ttk.Button(
            button_frame, text="Delete Selected", command=self.delete_task
        )
        delete_button.pack(side=tk.LEFT, padx=2)

        plot_button = ttk.Button(
            button_frame, text="Plot / Refresh Gantt", command=self.plot_gantt
        )
        plot_button.pack(side=tk.LEFT, padx=2)

        mc_button = ttk.Button(
            button_frame, text="Run Monte Carlo", command=self.run_monte_carlo_ui
        )
        mc_button.pack(side=tk.LEFT, padx=2)

        # Task list (Treeview) with reorder buttons
        # Use a PanedWindow so user can resize between task list and MC results
        list_mc_pane = ttk.Panedwindow(left_frame, orient=tk.VERTICAL)
        list_mc_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        list_frame = ttk.LabelFrame(list_mc_pane, text="Tasks")
        list_mc_pane.add(list_frame, weight=2)

        # Create a frame to hold treeview and buttons side by side
        list_inner_frame = tk.Frame(list_frame, bg=LIGHT_GRAY)
        list_inner_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        columns = ("name", "start", "end", "prob", "impact", "mode", "deps")
        self.tree = ttk.Treeview(
            list_inner_frame, columns=columns, show="headings", selectmode="browse", height=8
        )
        self.tree.heading("name", text="Task")
        self.tree.heading("start", text="Start")
        self.tree.heading("end", text="End")
        self.tree.heading("prob", text="Prob")
        self.tree.heading("impact", text="Impact")
        self.tree.heading("mode", text="Mode")
        self.tree.heading("deps", text="Dependencies")

        self.tree.column("name", width=100, anchor="w")
        self.tree.column("start", width=85, anchor="center")
        self.tree.column("end", width=85, anchor="center")
        self.tree.column("prob", width=50, anchor="center")
        self.tree.column("impact", width=50, anchor="center")
        self.tree.column("mode", width=50, anchor="center")
        self.tree.column("deps", width=120, anchor="w")

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Up/Down buttons frame
        reorder_frame = tk.Frame(list_inner_frame, bg=LIGHT_GRAY)
        reorder_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=2)

        # Up button (▲)
        up_btn = tk.Button(reorder_frame, text="▲", width=3, command=self.move_task_up,
                           bg=LIGHT_GRAY, font=("Arial", 10))
        up_btn.pack(pady=(20, 5))

        # Down button (▼)
        down_btn = tk.Button(reorder_frame, text="▼", width=3, command=self.move_task_down,
                             bg=LIGHT_GRAY, font=("Arial", 10))
        down_btn.pack(pady=5)

        # Manual order checkbox
        self.manual_order_var = tk.BooleanVar(value=False)
        manual_cb = tk.Checkbutton(reorder_frame, text="Manual\nOrder",
                                   variable=self.manual_order_var,
                                   command=self.toggle_manual_order,
                                   bg=LIGHT_GRAY, font=("Arial", 7),
                                   justify=tk.CENTER)
        manual_cb.pack(pady=(10, 5))

        # Bind selection to load task into form
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        # Monte Carlo results panel (in the same PanedWindow for resizing)
        mc_frame = ttk.LabelFrame(list_mc_pane, text="Monte Carlo Results (CI & SSI)")
        list_mc_pane.add(mc_frame, weight=1)

        self.mc_text = tk.Text(mc_frame, height=8, wrap="word", state="disabled",
                               bg=LIGHT_GRAY, fg="black", font=("Courier", 9))
        self.mc_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # ----------------- RIGHT SIDE: MATPLOTLIB GANTT CHART -----------------
        chart_frame = ttk.LabelFrame(
            right_frame, text="Gantt Chart (Committed Schedule)"
        )
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.figure, self.ax = plt.subplots(figsize=(7, 5))
        self.figure.patch.set_facecolor(CHART_BG)
        self.ax.set_facecolor(CHART_BG)
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Initial message in the plot
        self.ax.text(
            0.5,
            0.5,
            "Add tasks and click 'Plot / Refresh Gantt'",
            ha="center",
            va="center",
            transform=self.ax.transAxes,
        )
        self.ax.axis("off")
        self.canvas.draw()

    # ------------------------------------------------------------------
    # HELPER METHODS
    # ------------------------------------------------------------------
    def parse_date(self, text):
        try:
            return datetime.strptime(text.strip(), DATE_FORMAT)
        except ValueError:
            raise ValueError(f"Date '{text}' is not in expected format {DATE_FORMAT}")

    def refresh_deps_listbox(self):
        """Refresh the list of possible dependency tasks with all current task names."""
        self.deps_listbox.delete(0, tk.END)
        for t in self.tasks:
            self.deps_listbox.insert(tk.END, t["name"])

    def refresh_task_tree(self):
        """Refresh the visible list of tasks in the treeview."""
        for item in self.tree.get_children():
            self.tree.delete(item)

        for task in self.tasks:
            deps_str = ", ".join(task["deps"])
            p = task["prob"]
            i = task["impact"]
            mode = task.get("fail_mode", "late")[:1].upper()  # 'E' or 'L'
            self.tree.insert(
                "",
                tk.END,
                values=(
                    task["name"],
                    task["start"].strftime(DATE_FORMAT),
                    task["end"].strftime(DATE_FORMAT),
                    f"{p:.2f}",
                    f"{i:.2f}",
                    mode,
                    deps_str,
                ),
            )

    def clear_form(self):
        self.name_entry.delete(0, tk.END)
        today = date.today()
        self.start_entry.set_date(today)
        self.end_entry.set_date(today)
        self.prob_entry.delete(0, tk.END)
        self.prob_entry.insert(0, "0.0")
        self.impact_entry.delete(0, tk.END)
        self.impact_entry.insert(0, "0.0")
        self.fail_mode_var.set("late")
        self.deps_listbox.selection_clear(0, tk.END)
        self.selected_task_index = None

    def update_mc_summary(self, text_or_lines):
        """Update the Monte Carlo results text panel."""
        if isinstance(text_or_lines, list):
            text = "\n".join(text_or_lines)
        else:
            text = str(text_or_lines)

        self.mc_text.config(state="normal")
        self.mc_text.delete("1.0", tk.END)
        self.mc_text.insert(tk.END, text)
        self.mc_text.config(state="disabled")

    def show_risk_help(self):
        """Display a help dialog with probability and impact rubrics."""
        # Try to load from external JSON file
        help_data = self._load_help_json("risk_assessment.json")

        # Fallback data if file not found
        if help_data is None:
            help_data = {
                "title": "Risk Assessment Guide",
                "subtitle": "Use this guide to assign probability and impact values.\nThese values drive the duration distribution for Monte Carlo simulation.",
                "probability_scale": {
                    "title": "Probability Scale → Distribution Shape",
                    "columns": ["value", "level", "spread", "dist"],
                    "headings": ["Value", "Level", "Base Spread", "Distribution"],
                    "widths": [60, 120, 150, 200],
                    "data": [
                        ["0.10", "Rare", "±5% (P10-P90)", "Narrow triangular"],
                        ["0.25", "Unlikely", "±10%", "Triangular"],
                        ["0.50", "Possible", "±20%", "BetaPERT"],
                        ["0.75", "Likely", "±30-40%", "Skewed BetaPERT"],
                        ["0.90", "Almost Certain", "±50-80%", "Fat-tailed"],
                    ]
                },
                "impact_scale": {
                    "title": "Impact Scale → Distribution Scale",
                    "columns": ["value", "level", "effect", "mult"],
                    "headings": ["Value", "Level", "Duration Effect", "Multiplier"],
                    "widths": [60, 120, 200, 150],
                    "data": [
                        ["0.10", "Minimal", "+0-2 days", "0.5× spread"],
                        ["0.25", "Minor", "+3-7 days", "0.75× spread"],
                        ["0.50", "Moderate", "+1-2 weeks", "1.0× spread"],
                        ["0.75", "Major", "+3-5 weeks", "1.25× spread"],
                        ["1.00", "Severe", "+6+ weeks", "1.5× spread"],
                    ]
                },
                "fail_mode": {
                    "title": "Fail Mode: Early vs Late",
                    "text": "Early-Fail: Problems surface quickly at task start.\n  • More symmetric distribution, smaller right tail\n  • Examples: Material procurement, internal design, mature processes\n\nLate-Fail: Work appears on-track until near completion.\n  • Right-skewed BetaPERT, large right tail (fat-tailed)\n  • Examples: Supplier qualification, system integration, testing"
                },
                "metrics": {
                    "title": "Monte Carlo Metrics",
                    "text": "Criticality Index (CI):\n  • % of simulations where task lies on the critical path\n  • High CI = structural bottleneck (always/often constrains finish)\n\nSchedule Sensitivity Index (SSI):\n  • Correlation between task duration and project finish\n  • High SSI = statistical risk driver (variation impacts finish)\n\nTasks with high CI AND high SSI are the dominant schedule risks."
                }
            }

        help_window = tk.Toplevel(self.root)
        help_window.title("Risk Assessment Guide")
        help_window.geometry("750x700")
        help_window.resizable(False, False)
        help_window.configure(bg="#E2E8F0")

        style = ttk.Style()
        style.configure(
            "Help.Treeview",
            background="#E2E8F0",
            fieldbackground="#E2E8F0",
            foreground="black",
            rowheight=25
        )
        style.configure(
            "Help.Treeview.Heading",
            background="#E2E8F0",
            foreground="black",
            font=("Arial", 10, "bold")
        )
        style.layout("Help.Treeview", [
            ('Help.Treeview.treearea', {'sticky': 'nswe'})
        ])

        main_frame = tk.Frame(help_window, bg="#E2E8F0", padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(main_frame, bg="#E2E8F0", highlightthickness=0)
        scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#E2E8F0")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", on_mousewheel)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        title_font = ("Arial", 14, "bold")
        section_font = ("Arial", 11, "bold")
        normal_font = ("Arial", 10)

        def make_label(parent, text, font, **kwargs):
            return tk.Label(parent, text=text, font=font, bg="#E2E8F0", fg="black", **kwargs)

        def create_table(parent, columns, headings, widths, data, height=5):
            table_frame = tk.Frame(parent, bg="black", padx=1, pady=1)

            tree = ttk.Treeview(
                table_frame,
                columns=columns,
                show="headings",
                height=height,
                style="Help.Treeview"
            )

            for col, heading, width in zip(columns, headings, widths):
                tree.heading(col, text=heading, anchor="w")
                tree.column(col, width=width, anchor="w" if col != columns[0] else "center")

            for item in data:
                tree.insert("", tk.END, values=item)

            tree.pack(fill=tk.X)
            return table_frame

        # Title
        title_label = make_label(scrollable_frame, text=help_data["title"], font=title_font)
        title_label.pack(anchor="w", pady=(0, 5))

        subtitle_label = make_label(scrollable_frame, text=help_data["subtitle"], font=normal_font)
        subtitle_label.pack(anchor="w", pady=(0, 15))

        # Probability Scale
        prob = help_data["probability_scale"]
        prob_section = make_label(scrollable_frame, text=prob["title"], font=section_font)
        prob_section.pack(anchor="w", pady=(10, 5))
        prob_table = create_table(scrollable_frame, prob["columns"], prob["headings"], prob["widths"], prob["data"])
        prob_table.pack(fill=tk.X, pady=(0, 10))

        # Impact Scale
        impact = help_data["impact_scale"]
        impact_section = make_label(scrollable_frame, text=impact["title"], font=section_font)
        impact_section.pack(anchor="w", pady=(10, 5))
        impact_table = create_table(scrollable_frame, impact["columns"], impact["headings"], impact["widths"],
                                    impact["data"])
        impact_table.pack(fill=tk.X, pady=(0, 10))

        # Fail Mode Section
        fail = help_data["fail_mode"]
        mode_section = make_label(scrollable_frame, text=fail["title"], font=section_font)
        mode_section.pack(anchor="w", pady=(10, 5))
        mode_label = make_label(scrollable_frame, text=fail["text"], font=normal_font, justify=tk.LEFT)
        mode_label.pack(anchor="w", padx=10, pady=(0, 15))

        # Metrics Section
        metrics = help_data["metrics"]
        metrics_section = make_label(scrollable_frame, text=metrics["title"], font=section_font)
        metrics_section.pack(anchor="w", pady=(10, 5))
        metrics_label = make_label(scrollable_frame, text=metrics["text"], font=normal_font, justify=tk.LEFT)
        metrics_label.pack(anchor="w", padx=10, pady=(0, 15))

        # Quadrant Colors Section (if present in help data)
        if "quadrant_colors" in help_data:
            quadrant = help_data["quadrant_colors"]
            quadrant_section = make_label(scrollable_frame, text=quadrant["title"], font=section_font)
            quadrant_section.pack(anchor="w", pady=(10, 5))
            quadrant_label = make_label(scrollable_frame, text=quadrant["text"], font=normal_font, justify=tk.LEFT)
            quadrant_label.pack(anchor="w", padx=10, pady=(0, 15))

        close_button = tk.Button(
            scrollable_frame,
            text="Close",
            command=lambda: [canvas.unbind_all("<MouseWheel>"), help_window.destroy()],
            bg="#E2E8F0",
            relief=tk.RAISED,
            padx=20,
            pady=5
        )
        close_button.pack(pady=15)

        help_window.update_idletasks()
        x = (help_window.winfo_screenwidth() // 2) - (750 // 2)
        y = (help_window.winfo_screenheight() // 2) - (700 // 2)
        help_window.geometry(f"750x700+{x}+{y}")

    # ------------------------------------------------------------------
    # TREE SELECTION HANDLER (LOAD TASK INTO FORM)
    # ------------------------------------------------------------------
    def on_tree_select(self, event):
        selected = self.tree.selection()
        if not selected:
            return
        item = selected[0]
        row_index = self.tree.index(item)
        if row_index < 0 or row_index >= len(self.tasks):
            return

        task = self.tasks[row_index]
        self.selected_task_index = row_index

        # Populate form with selected task data
        self.name_entry.delete(0, tk.END)
        self.name_entry.insert(0, task["name"])

        self.start_entry.set_date(task["start"].date())
        self.end_entry.set_date(task["end"].date())

        self.prob_entry.delete(0, tk.END)
        self.prob_entry.insert(0, f"{task['prob']:.2f}")
        self.impact_entry.delete(0, tk.END)
        self.impact_entry.insert(0, f"{task['impact']:.2f}")

        self.fail_mode_var.set(task.get("fail_mode", "late"))

        # Dependencies: select items in listbox
        self.deps_listbox.selection_clear(0, tk.END)
        all_names = [t["name"] for t in self.tasks]
        for dep_name in task["deps"]:
            if dep_name in all_names:
                idx = all_names.index(dep_name)
                self.deps_listbox.selection_set(idx)

    # ------------------------------------------------------------------
    # BUTTON CALLBACKS
    # ------------------------------------------------------------------
    def add_task(self):
        name = self.name_entry.get().strip()
        start_text = self.start_entry.get().strip()
        end_text = self.end_entry.get().strip()
        prob_text = self.prob_entry.get().strip()
        impact_text = self.impact_entry.get().strip()

        if not name or not start_text or not end_text:
            messagebox.showerror(
                "Missing Data", "Please fill in task name, start date, and end date."
            )
            return

        try:
            start_date = self.parse_date(start_text)
            end_date = self.parse_date(end_text)
        except ValueError as e:
            messagebox.showerror("Invalid Date", str(e))
            return

        if end_date <= start_date:
            messagebox.showerror(
                "Invalid Range", "End date must be after start date."
            )
            return

        try:
            prob = float(prob_text) if prob_text else 0.0
            impact = float(impact_text) if impact_text else 0.0
        except ValueError:
            messagebox.showerror(
                "Invalid Input", "Probability and Impact must be numbers between 0 and 1."
            )
            return

        if not (0.0 <= prob <= 1.0) or not (0.0 <= impact <= 1.0):
            messagebox.showerror(
                "Invalid Range", "Probability and Impact must be between 0 and 1."
            )
            return

        deps_indices = self.deps_listbox.curselection()
        deps = [self.deps_listbox.get(i) for i in deps_indices]

        if any(t["name"] == name for t in self.tasks):
            if not messagebox.askyesno(
                    "Duplicate Name",
                    f"A task named '{name}' already exists.\nAdd another with the same name?",
            ):
                return

        task = {
            "name": name,
            "start": start_date,
            "end": end_date,
            "deps": deps,
            "prob": prob,
            "impact": impact,
            "fail_mode": self.fail_mode_var.get(),
        }
        self.tasks.append(task)

        self.mc_results = None
        self.update_mc_summary(
            "Monte Carlo results cleared.\nRun again to refresh."
        )

        self.refresh_deps_listbox()
        self.refresh_task_tree()
        self.clear_form()

    def update_task(self):
        """Update the currently selected task with the values from the form."""
        if self.selected_task_index is None:
            messagebox.showinfo(
                "No Selection", "Select a task from the list to update."
            )
            return

        idx = self.selected_task_index
        if idx < 0 or idx >= len(self.tasks):
            messagebox.showerror("Error", "Selected task index is invalid.")
            return

        name = self.name_entry.get().strip()
        start_text = self.start_entry.get().strip()
        end_text = self.end_entry.get().strip()
        prob_text = self.prob_entry.get().strip()
        impact_text = self.impact_entry.get().strip()

        if not name or not start_text or not end_text:
            messagebox.showerror(
                "Missing Data", "Please fill in task name, start date, and end date."
            )
            return

        try:
            start_date = self.parse_date(start_text)
            end_date = self.parse_date(end_text)
        except ValueError as e:
            messagebox.showerror("Invalid Date", str(e))
            return

        if end_date <= start_date:
            messagebox.showerror(
                "Invalid Range", "End date must be after start date."
            )
            return

        try:
            prob = float(prob_text) if prob_text else 0.0
            impact = float(impact_text) if impact_text else 0.0
        except ValueError:
            messagebox.showerror(
                "Invalid Input", "Probability and Impact must be numbers between 0 and 1."
            )
            return

        if not (0.0 <= prob <= 1.0) or not (0.0 <= impact <= 1.0):
            messagebox.showerror(
                "Invalid Range", "Probability and Impact must be between 0 and 1."
            )
            return

        deps_indices = self.deps_listbox.curselection()
        deps = [self.deps_listbox.get(i) for i in deps_indices]

        old_name = self.tasks[idx]["name"]
        self.tasks[idx]["name"] = name
        self.tasks[idx]["start"] = start_date
        self.tasks[idx]["end"] = end_date
        self.tasks[idx]["prob"] = prob
        self.tasks[idx]["impact"] = impact
        self.tasks[idx]["deps"] = deps
        self.tasks[idx]["fail_mode"] = self.fail_mode_var.get()

        if name != old_name:
            for t in self.tasks:
                t["deps"] = [name if d == old_name else d for d in t["deps"]]

        self.mc_results = None
        self.update_mc_summary(
            "Monte Carlo results cleared.\nRun again to refresh."
        )

        self.refresh_deps_listbox()
        self.refresh_task_tree()

        children = self.tree.get_children()
        if 0 <= idx < len(children):
            self.tree.selection_set(children[idx])

    def delete_task(self):
        """Delete the selected task and remove it from dependencies of other tasks."""
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo(
                "No Selection", "Please select a task to delete."
            )
            return

        item = selected[0]
        row_index = self.tree.index(item)
        if row_index < 0 or row_index >= len(self.tasks):
            messagebox.showerror("Error", "Selected task index is invalid.")
            return

        name_to_delete = self.tasks[row_index]["name"]

        new_tasks = []
        for i, t in enumerate(self.tasks):
            if i != row_index:
                new_tasks.append(t)
        self.tasks = new_tasks

        for t in self.tasks:
            t["deps"] = [d for d in t["deps"] if d != name_to_delete]

        self.selected_task_index = None
        self.mc_results = None
        self.update_mc_summary(
            "Monte Carlo results cleared.\nRun again to refresh."
        )

        self.refresh_deps_listbox()
        self.refresh_task_tree()
        self.plot_gantt()

    def move_task_up(self):
        """Move the selected task up one position in the list."""
        selected = self.tree.selection()
        if not selected:
            return

        item = selected[0]
        row_index = self.tree.index(item)

        if row_index <= 0:
            return  # Already at top

        # Swap with the task above
        self.tasks[row_index], self.tasks[row_index - 1] = \
            self.tasks[row_index - 1], self.tasks[row_index]

        # Enable manual order mode automatically when reordering
        if not self.use_manual_order:
            self.use_manual_order = True
            self.manual_order_var.set(True)

        # Refresh display
        self.refresh_deps_listbox()
        self.refresh_task_tree()

        # Reselect the moved task
        children = self.tree.get_children()
        if 0 <= row_index - 1 < len(children):
            self.tree.selection_set(children[row_index - 1])
            self.selected_task_index = row_index - 1

        # Update Gantt if visible
        if self.tasks:
            self.plot_gantt()

    def move_task_down(self):
        """Move the selected task down one position in the list."""
        selected = self.tree.selection()
        if not selected:
            return

        item = selected[0]
        row_index = self.tree.index(item)

        if row_index >= len(self.tasks) - 1:
            return  # Already at bottom

        # Swap with the task below
        self.tasks[row_index], self.tasks[row_index + 1] = \
            self.tasks[row_index + 1], self.tasks[row_index]

        # Enable manual order mode automatically when reordering
        if not self.use_manual_order:
            self.use_manual_order = True
            self.manual_order_var.set(True)

        # Refresh display
        self.refresh_deps_listbox()
        self.refresh_task_tree()

        # Reselect the moved task
        children = self.tree.get_children()
        if 0 <= row_index + 1 < len(children):
            self.tree.selection_set(children[row_index + 1])
            self.selected_task_index = row_index + 1

        # Update Gantt if visible
        if self.tasks:
            self.plot_gantt()

    def toggle_manual_order(self):
        """Toggle between manual order and auto-sort modes."""
        self.use_manual_order = self.manual_order_var.get()

        if self.tasks:
            self.plot_gantt()

    def _get_quadrant_color(self, ci, ssi, threshold=0.5):
        """
        Determine task bar color based on CI×SSI risk quadrant.

        Quadrants (default threshold = 50%):
        - Low CI + Low SSI  = Green  (#16A34A) - Safe task
        - Low CI + High SSI = Amber  (#F59E0B) - Hidden risk (watch closely)
        - High CI + Low SSI = Orange (#EA580C) - Structural bottleneck
        - High CI + High SSI = Red   (#DC2626) - Dominant schedule risk

        Colors smoothly interpolate within each quadrant for nuance.
        """
        # Normalize to 0-1 within each quadrant for smooth gradients
        ci_high = ci >= threshold
        ssi_high = ssi >= threshold

        if not ci_high and not ssi_high:
            # Low CI, Low SSI - GREEN (safe) - base #16A34A
            intensity = max(ci, ssi) / threshold  # 0 to 1 within quadrant
            r = 0.09 + 0.10 * (1 - intensity)
            g = 0.64 - 0.08 * (1 - intensity)
            b = 0.29 + 0.10 * (1 - intensity)
            return (r, g, b)

        elif not ci_high and ssi_high:
            # Low CI, High SSI - AMBER (hidden risk) - base #F59E0B
            # Shows tasks that aren't always critical but drive variance when they are
            intensity = (ssi - threshold) / (1 - threshold)  # 0 to 1 within quadrant
            r = 0.96
            g = 0.62 - 0.08 * intensity
            b = 0.04 + 0.10 * (1 - intensity)
            return (r, g, b)

        elif ci_high and not ssi_high:
            # High CI, Low SSI - ORANGE (structural bottleneck) - base #EA580C
            # Always on critical path but doesn't drive much variance
            intensity = (ci - threshold) / (1 - threshold)
            r = 0.92
            g = 0.35 - 0.08 * intensity
            b = 0.05
            return (r, g, b)

        else:
            # High CI, High SSI - RED (dominant risk) - base #DC2626
            # The true schedule killers - always critical AND high variance
            ci_intensity = (ci - threshold) / (1 - threshold)
            ssi_intensity = (ssi - threshold) / (1 - threshold)
            intensity = (ci_intensity + ssi_intensity) / 2
            r = 0.86 - 0.10 * (1 - intensity)
            g = 0.15 - 0.08 * intensity
            b = 0.15 - 0.08 * intensity
            return (r, g, b)

    def _get_task_sort_key(self, task):
        """
        Generate sort key for task ordering on Gantt chart.

        Sorts by:
        1. Category prefix with custom order (D=Design, S=Supplier, M=Manufacturing)
        2. Numeric suffix (if present) - orders D1, D2, D3 correctly
        3. Start date (as fallback)

        Examples:
          "D1" -> (0, 1, start_date)  - Design tasks first
          "S1" -> (1, 1, start_date)  - Supplier tasks second
          "M1" -> (2, 1, start_date)  - Manufacturing tasks third
          "Task_1" -> (99, 1, start_date)  - Other tasks last
        """
        name = task["name"]

        # Custom category order (modify this dict to change grouping order)
        category_order = {
            'D': 0,  # Design / Technical path
            'S': 1,  # Supplier path
            'M': 2,  # Manufacturing path
            'T': 3,  # Test (if used)
            'R': 4,  # Regulatory (if used)
        }

        # Extract prefix (letters/underscores) and numeric suffix
        match = re.match(r'^([A-Za-z_]+)(\d*)$', name)
        if match:
            prefix = match.group(1)
            num_str = match.group(2)
            num = int(num_str) if num_str else 0
        else:
            # Fallback for names that don't match pattern
            prefix = name
            num = 0

        # Get category order (default to 99 for unknown categories)
        cat_order = category_order.get(prefix, 99)

        return (cat_order, prefix, num, task["start"])

    def plot_gantt(self):
        if not self.tasks:
            messagebox.showinfo("No Tasks", "Add some tasks first.")
            return

        # Use manual order (list order) or auto-sort by category
        if self.use_manual_order:
            sorted_tasks = list(self.tasks)  # Use current list order
        else:
            sorted_tasks = sorted(self.tasks, key=self._get_task_sort_key)

        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

        CHART_BG = "#F8FAFC"  # Near white chart background
        self.figure.patch.set_facecolor(CHART_BG)
        self.ax.set_facecolor(CHART_BG)

        name_to_y = {}
        for idx, task in enumerate(sorted_tasks):
            name_to_y[task["name"]] = idx

        crit_map = None
        ssi_map = None
        if self.mc_results:
            if "criticality" in self.mc_results:
                crit_map = self.mc_results["criticality"]
            if "ssi" in self.mc_results:
                ssi_map = self.mc_results["ssi"]

        bar_height = 0.5

        for idx, task in enumerate(sorted_tasks):
            start_num = mdates.date2num(task["start"])
            end_num = mdates.date2num(task["end"])
            duration = end_num - start_num

            # Color by CI×SSI quadrant (risk categorization)
            if crit_map and ssi_map and task["name"] in crit_map:
                ci = crit_map[task["name"]]
                ssi = ssi_map.get(task["name"], 0)
                color = self._get_quadrant_color(ci, ssi)
            elif crit_map and task["name"] in crit_map:
                # Fallback to CI-only coloring if SSI not available
                crit = crit_map[task["name"]]
                color = cm.Reds(0.2 + 0.8 * crit)
            else:
                color = "#5C7999"  # Steel blue-grey for unanalyzed tasks

            rounded_box = FancyBboxPatch(
                (start_num, idx - bar_height / 2),
                duration,
                bar_height,
                boxstyle="round,pad=0,rounding_size=0.2",
                facecolor=color,
                edgecolor="black",
                linewidth=1,
            )
            self.ax.add_patch(rounded_box)

            # Task label with CI and SSI if available
            label = task["name"]
            if crit_map and ssi_map and task["name"] in crit_map:
                ci = crit_map[task["name"]] * 100
                ssi = ssi_map.get(task["name"], 0) * 100
                label = f"{task['name']}\nCI:{ci:.0f}% SSI:{ssi:.0f}%"

            self.ax.text(
                start_num + duration / 2,
                idx,
                label,
                va="center",
                ha="center",
                fontsize=10,
                fontweight="bold",
                color="white",
            )

        # Dependency arrows
        # Pre-compute x range for connector offset calculations
        all_starts = [mdates.date2num(t["start"]) for t in sorted_tasks]
        all_ends = [mdates.date2num(t["end"]) for t in sorted_tasks]
        x_range = max(all_ends) - min(all_starts)
        min_offset = x_range * 0.02  # 2% of chart width for minimum connector visibility

        # Jump dimensions for connector crossings (proportional to chart width)
        # jump_width controls the horizontal gap where the arc occurs
        # jump_height controls how high/far the arc extends
        jump_width = x_range * 0.008  # Horizontal extent of the jump
        jump_height = x_range * 0.0008  # Vertical extent of the jump

        # Initialize connector data storage for edit mode
        self._connector_data = {}
        self.connector_artists = {}

        # ===== FIRST PASS: Collect all connector data =====
        connector_list = []  # List of (connector_key, start, end, waypoints, edge_color, line_style)

        for task in sorted_tasks:
            y_task = name_to_y[task["name"]]
            for dep_name in task["deps"]:
                if dep_name not in name_to_y:
                    continue
                y_dep = name_to_y[dep_name]

                dep_task = next(
                    (t for t in sorted_tasks if t["name"] == dep_name), None
                )
                if dep_task is None:
                    continue

                x_start = mdates.date2num(dep_task["end"])
                x_end = mdates.date2num(task["start"])

                # Connector key for this dependency
                connector_key = (dep_name, task["name"])

                # Calculate default bend point (middle x position for vertical segment)
                if x_end > x_start:
                    # Normal gap - bend at midpoint
                    default_bend_x = x_start + (x_end - x_start) * 0.5
                    edge_color = '#4A5568'  # Softer charcoal
                    line_style = '-'
                elif x_end == x_start:
                    # Back-to-back - bend slightly to the right
                    default_bend_x = x_start + min_offset
                    edge_color = '#4A5568'  # Softer charcoal
                    line_style = '-'
                else:
                    # Overlapping - bend to the right of predecessor end
                    default_bend_x = x_start + min_offset * 2
                    edge_color = '#718096'  # Medium grey for overlap
                    line_style = '--'

                if y_dep == y_task:
                    # Same row - just draw a horizontal line (no waypoints)
                    connector_list.append({
                        'key': connector_key,
                        'start': (x_start, y_dep),
                        'end': (x_end, y_task),
                        'waypoints': [],
                        'edge_color': edge_color,
                        'line_style': line_style,
                        'same_row': True,
                    })
                    continue

                # Default waypoints: simple two-corner path
                default_waypoints = [
                    [default_bend_x, y_dep],  # First corner (horizontal then down/up)
                    [default_bend_x, y_task],  # Second corner (vertical then horizontal)
                ]

                # Check for custom waypoints
                if connector_key in self.connector_waypoints:
                    stored = self.connector_waypoints[connector_key]
                    if 'points' in stored:
                        # New multi-waypoint format
                        waypoints = stored['points']
                    elif 'bend_x' in stored:
                        # Legacy single bend_x format - convert
                        bend_x = stored['bend_x']
                        waypoints = [[bend_x, y_dep], [bend_x, y_task]]
                    else:
                        waypoints = default_waypoints
                else:
                    waypoints = default_waypoints

                # Store connector data for edit mode
                self._connector_data[connector_key] = {
                    'start': (x_start, y_dep),
                    'end': (x_end, y_task),
                    'waypoints': waypoints,
                    'default_waypoints': default_waypoints,
                    'edge_color': edge_color,
                    'line_style': line_style,
                    'min_offset': min_offset,
                }

                connector_list.append({
                    'key': connector_key,
                    'start': (x_start, y_dep),
                    'end': (x_end, y_task),
                    'waypoints': waypoints,
                    'edge_color': edge_color,
                    'line_style': line_style,
                    'same_row': False,
                })

        # ===== SECOND PASS: Find all intersections between connectors =====
        # Build segment lists for each connector
        connector_segments = {}
        for conn in connector_list:
            if conn['same_row']:
                segments = [(conn['start'], conn['end'])]
            else:
                segments = get_segments_from_path(conn['start'], conn['waypoints'], conn['end'])
            connector_segments[conn['key']] = segments

        # Find intersections for each connector with all other connectors
        connector_intersections = {conn['key']: [] for conn in connector_list}

        connector_keys = list(connector_segments.keys())
        for i, key1 in enumerate(connector_keys):
            for key2 in connector_keys[i + 1:]:
                # Check all segment pairs between these two connectors
                for seg1 in connector_segments[key1]:
                    for seg2 in connector_segments[key2]:
                        intersection = line_segment_intersection(seg1[0], seg1[1], seg2[0], seg2[1])
                        if intersection:
                            # Add intersection to both connectors
                            connector_intersections[key1].append(intersection)
                            connector_intersections[key2].append(intersection)

        # ===== THIRD PASS: Draw connectors with jumps =====
        for conn in connector_list:
            connector_key = conn['key']
            x_start, y_start = conn['start']
            x_end, y_end = conn['end']
            edge_color = conn['edge_color']
            line_style = conn['line_style']

            # Draw connector dot at predecessor end
            self.ax.plot(x_start, y_start, 'o', color='#4A5568', markersize=4, zorder=5)

            if conn['same_row']:
                # Same row - just draw a horizontal line
                self.ax.plot([x_start, x_end], [y_start, y_end],
                             color=edge_color, linewidth=1.5, linestyle=line_style, zorder=3)
                continue

            waypoints = conn['waypoints']
            intersections = connector_intersections[connector_key]

            if intersections:
                # Build path with jumps
                verts, codes = build_path_with_jumps(
                    conn['start'], waypoints, conn['end'],
                    intersections, jump_width, jump_height
                )
                path_with_jumps = Path(verts, codes)

                patch = mpatches.PathPatch(
                    path_with_jumps,
                    facecolor='none',
                    edgecolor=edge_color,
                    linewidth=1.5,
                    linestyle=line_style,
                    zorder=3,
                )
            else:
                # No intersections - draw normal path
                path_data = [(Path.MOVETO, (x_start, y_start))]
                for wp in waypoints:
                    path_data.append((Path.LINETO, (wp[0], wp[1])))
                path_data.append((Path.LINETO, (x_end, y_end)))

                codes, verts = zip(*path_data)
                orthogonal_path = Path(verts, codes)

                patch = mpatches.PathPatch(
                    orthogonal_path,
                    facecolor='none',
                    edgecolor=edge_color,
                    linewidth=1.5,
                    linestyle=line_style,
                    zorder=3,
                )

            self.ax.add_patch(patch)

            # Store the patch artist for later updates
            self.connector_artists[connector_key] = patch

            # Arrow at the end - pointing left into the task
            arrow_offset = min_offset * 0.5
            self.ax.annotate(
                "",
                xy=(x_end, y_end),
                xytext=(x_end + arrow_offset, y_end),
                arrowprops=dict(
                    arrowstyle="->",
                    color=edge_color,
                    linewidth=1.5,
                    mutation_scale=12,
                ),
                zorder=4,
            )

        self.ax.set_yticks(range(len(sorted_tasks)))
        self.ax.set_yticklabels([t["name"] for t in sorted_tasks])
        self.ax.invert_yaxis()

        # all_starts and all_ends already computed above for connector calculations
        x_min = min(all_starts)
        x_max = max(all_ends)
        x_padding = (x_max - x_min) * 0.05
        self.ax.set_xlim(x_min - x_padding, x_max + x_padding)

        self.ax.set_ylim(len(sorted_tasks) - 0.5, -0.5)

        self.ax.xaxis_date()
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        self.figure.autofmt_xdate()

        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Tasks")
        title = "Project Gantt Chart"
        if self.use_manual_order:
            title += " (Manual Order)"
        else:
            title += " (Auto-Sorted)"
        if self.mc_results:
            title += " — Risk Quadrant Coloring"
        self.ax.set_title(title)

        # Add grid lines - both horizontal and vertical
        self.ax.grid(True, axis="both", linestyle="--", alpha=0.3, zorder=0)
        # Make horizontal lines slightly more visible to separate task rows
        self.ax.set_axisbelow(True)  # Ensure grid is behind other elements

        # Add 2x2 quadrant legend if Monte Carlo results exist
        if self.mc_results:
            self._draw_quadrant_legend()

        self.canvas.draw_idle()

        # If in edit mode, draw control points and set up event handlers
        if self.connector_edit_mode:
            self._draw_control_points()
            self._connect_mouse_events()

    def _draw_quadrant_legend(self):
        """
        Draw a 2x2 quadrant legend showing CI×SSI risk categories.

        Format:
                         Low SSI        High SSI
                       ┌───────────┬───────────┐
          High CI      │  ORANGE   │    RED    │
                       │ Bottleneck│Dom. Risk  │
                       ├───────────┼───────────┤
          Low CI       │  GREEN    │  YELLOW   │
                       │   Safe    │Hidden Risk│
                       └───────────┴───────────┘
        """
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        # Create inset axes for the legend (upper right corner)
        # Width and height are in inches - increased size
        legend_ax = inset_axes(self.ax, width=2.2, height=1.5, loc='upper right',
                               borderpad=0.3)

        # Turn off the axis and make background transparent
        legend_ax.set_xlim(0, 4)
        legend_ax.set_ylim(0, 3)
        legend_ax.axis('off')
        legend_ax.patch.set_alpha(0)  # Transparent background

        # Colors for each quadrant - refined palette
        colors = {
            'orange': '#EA580C',  # High CI, Low SSI - Bottleneck
            'red': '#DC2626',  # High CI, High SSI - Dominant Risk
            'green': '#16A34A',  # Low CI, Low SSI - Safe
            'amber': '#F59E0B',  # Low CI, High SSI - Hidden Risk (amber instead of harsh yellow)
        }

        # Draw the 2x2 grid cells - larger cells
        cell_w, cell_h = 1.5, 1.0
        x_offset, y_offset = 1.0, 0.2

        # High CI row (top)
        # Orange - Bottleneck (High CI, Low SSI)
        legend_ax.add_patch(plt.Rectangle((x_offset, y_offset + cell_h), cell_w, cell_h,
                                          facecolor=colors['orange'], edgecolor='black', linewidth=1.5))
        legend_ax.text(x_offset + cell_w / 2, y_offset + cell_h + cell_h / 2 + 0.15, 'ORANGE',
                       ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        legend_ax.text(x_offset + cell_w / 2, y_offset + cell_h + cell_h / 2 - 0.22, 'Bottleneck',
                       ha='center', va='center', fontsize=7, color='white')

        # Red - Dominant Risk (High CI, High SSI)
        legend_ax.add_patch(plt.Rectangle((x_offset + cell_w, y_offset + cell_h), cell_w, cell_h,
                                          facecolor=colors['red'], edgecolor='black', linewidth=1.5))
        legend_ax.text(x_offset + cell_w + cell_w / 2, y_offset + cell_h + cell_h / 2 + 0.15, 'RED',
                       ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        legend_ax.text(x_offset + cell_w + cell_w / 2, y_offset + cell_h + cell_h / 2 - 0.22, 'Dom. Risk',
                       ha='center', va='center', fontsize=7, color='white')

        # Low CI row (bottom)
        # Green - Safe (Low CI, Low SSI)
        legend_ax.add_patch(plt.Rectangle((x_offset, y_offset), cell_w, cell_h,
                                          facecolor=colors['green'], edgecolor='black', linewidth=1.5))
        legend_ax.text(x_offset + cell_w / 2, y_offset + cell_h / 2 + 0.15, 'GREEN',
                       ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        legend_ax.text(x_offset + cell_w / 2, y_offset + cell_h / 2 - 0.22, 'Safe',
                       ha='center', va='center', fontsize=7, color='white')

        # Amber - Hidden Risk (Low CI, High SSI)
        legend_ax.add_patch(plt.Rectangle((x_offset + cell_w, y_offset), cell_w, cell_h,
                                          facecolor=colors['amber'], edgecolor='black', linewidth=1.5))
        legend_ax.text(x_offset + cell_w + cell_w / 2, y_offset + cell_h / 2 + 0.15, 'AMBER',
                       ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        legend_ax.text(x_offset + cell_w + cell_w / 2, y_offset + cell_h / 2 - 0.22, 'Hidden Risk',
                       ha='center', va='center', fontsize=7, color='white')

        # Column headers (SSI)
        legend_ax.text(x_offset + cell_w / 2, y_offset + 2 * cell_h + 0.18, 'Low SSI',
                       ha='center', va='bottom', fontsize=7, fontweight='bold')
        legend_ax.text(x_offset + cell_w + cell_w / 2, y_offset + 2 * cell_h + 0.18, 'High SSI',
                       ha='center', va='bottom', fontsize=7, fontweight='bold')

        # Row headers (CI)
        legend_ax.text(x_offset - 0.12, y_offset + cell_h + cell_h / 2, 'High\nCI',
                       ha='right', va='center', fontsize=7, fontweight='bold')
        legend_ax.text(x_offset - 0.12, y_offset + cell_h / 2, 'Low\nCI',
                       ha='right', va='center', fontsize=7, fontweight='bold')

    # ------------------------------------------------------------------
    # CONNECTOR EDITING
    # ------------------------------------------------------------------
    def toggle_connector_edit_mode(self):
        """Toggle the connector edit mode on/off."""
        self.connector_edit_mode = not self.connector_edit_mode

        if self.connector_edit_mode:
            snap_status = "ON" if self.snap_to_grid else "OFF"
            messagebox.showinfo(
                "Connector Edit Mode",
                f"Connector edit mode ENABLED.\n\n"
                f"• Orange/Blue squares show waypoints on each connector\n"
                f"• LEFT-CLICK + DRAG: Move a waypoint\n"
                f"• DOUBLE-CLICK on connector line: Add a new waypoint\n"
                f"• RIGHT-CLICK on waypoint: Remove it (min 2 required)\n"
                f"• Snap to Grid: {snap_status} (toggle with Ctrl+G)\n\n"
                f"• Press Ctrl+E again to exit edit mode\n"
                f"• Use 'Reset All Connectors' to restore defaults"
            )
        else:
            # Disconnect mouse events
            self._disconnect_mouse_events()

        # Redraw to show/hide control points
        self.plot_gantt()

    def reset_connectors(self):
        """Reset all connector waypoints to default positions."""
        self.connector_waypoints.clear()
        self.plot_gantt()
        messagebox.showinfo("Connectors Reset", "All connectors have been reset to default positions.")

    def toggle_snap_to_grid(self):
        """Toggle snap-to-grid on/off for waypoint editing."""
        self.snap_to_grid = not self.snap_to_grid
        state = "ENABLED" if self.snap_to_grid else "DISABLED"
        messagebox.showinfo("Snap to Grid", f"Snap to grid is now {state}.\n\n"
                                            "When enabled, waypoints will align to:\n"
                                            "• Vertical (X): Day boundaries\n"
                                            "• Horizontal (Y): 1/8 steps between task rows\n"
                                            "  (allows fine positioning between lanes)")

    def _snap_coordinates(self, x, y):
        """Snap coordinates to the nearest grid intersection."""
        if not self.snap_to_grid:
            return x, y

        # Snap Y to nearest 1/8 step between task rows
        # This allows positioning at 0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1, etc.
        snapped_y = round(y * 8) / 8

        # Snap X to nearest day (since x-axis is in matplotlib date numbers)
        # Each integer represents one day
        snapped_x = round(x)

        return snapped_x, snapped_y

    def _draw_control_points(self):
        """Draw draggable control points for orthogonal connectors."""
        self.control_point_artists = []

        if not hasattr(self, '_connector_data'):
            return

        for connector_key, data in self._connector_data.items():
            waypoints = data['waypoints']
            edge_color = data['edge_color']

            # Draw a control point for each waypoint
            for i, wp in enumerate(waypoints):
                # Alternate colors for clarity
                if i % 2 == 0:
                    color = '#EA580C'  # Orange for even waypoints
                else:
                    color = '#3182CE'  # Blue for odd waypoints

                cp = self.ax.plot(wp[0], wp[1], 's',  # Square marker
                                  color=color, markersize=9,
                                  markeredgecolor='white', markeredgewidth=2,
                                  picker=5, zorder=10)[0]
                cp.set_gid(f"{connector_key[0]}|{connector_key[1]}|wp_{i}")
                self.control_point_artists.append(cp)

            # Also highlight the connector path for easier clicking to add points
            # Draw a slightly wider invisible path for click detection
            x_start, y_start = data['start']
            x_end, y_end = data['end']

            # Build path points for hover detection
            path_points = [(x_start, y_start)] + waypoints + [(x_end, y_end)]

            # Draw segments with wider hit area (invisible)
            for j in range(len(path_points) - 1):
                p1, p2 = path_points[j], path_points[j + 1]
                line, = self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                                     color='none', linewidth=15, picker=5, zorder=2)
                line.set_gid(f"{connector_key[0]}|{connector_key[1]}|seg_{j}")
                self.control_point_artists.append(line)

        self.canvas.draw_idle()

    def _connect_mouse_events(self):
        """Connect mouse event handlers for connector editing."""
        self._disconnect_mouse_events()  # Clear any existing connections

        self._cid_press = self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self._cid_release = self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self._cid_motion = self.canvas.mpl_connect('motion_notify_event', self._on_mouse_motion)

    def _disconnect_mouse_events(self):
        """Disconnect mouse event handlers."""
        if self._cid_press is not None:
            self.canvas.mpl_disconnect(self._cid_press)
            self._cid_press = None
        if self._cid_release is not None:
            self.canvas.mpl_disconnect(self._cid_release)
            self._cid_release = None
        if self._cid_motion is not None:
            self.canvas.mpl_disconnect(self._cid_motion)
            self._cid_motion = None

    def _on_mouse_press(self, event):
        """Handle mouse button press for connector editing."""
        if event.inaxes != self.ax:
            return

        # Right-click: Remove waypoint
        if event.button == 3:
            self._handle_remove_waypoint(event)
            return

        # Double-click: Add waypoint
        if event.button == 1 and event.dblclick:
            self._handle_add_waypoint(event)
            return

        # Single left-click: Start dragging
        if event.button == 1:
            for artist in self.control_point_artists:
                contains, _ = artist.contains(event)
                if contains:
                    gid = artist.get_gid()
                    if gid and '|wp_' in gid:
                        parts = gid.split('|')
                        if len(parts) == 3:
                            from_task, to_task = parts[0], parts[1]
                            wp_idx = int(parts[2].replace('wp_', ''))
                            self.dragging_point = ((from_task, to_task), wp_idx, artist)
                            return

    def _handle_add_waypoint(self, event):
        """Add a new waypoint to a connector at the click location."""
        click_x, click_y = event.xdata, event.ydata

        # Find which connector segment was clicked
        for artist in self.control_point_artists:
            gid = artist.get_gid()
            if gid and '|seg_' in gid:
                contains, _ = artist.contains(event)
                if contains:
                    parts = gid.split('|')
                    from_task, to_task = parts[0], parts[1]
                    seg_idx = int(parts[2].replace('seg_', ''))
                    connector_key = (from_task, to_task)

                    # Get current waypoints
                    if connector_key not in self.connector_waypoints:
                        data = self._connector_data[connector_key]
                        self.connector_waypoints[connector_key] = {
                            'points': [list(wp) for wp in data['waypoints']]
                        }

                    waypoints = self.connector_waypoints[connector_key]['points']

                    # Insert two new waypoints to create a detour
                    # This creates a "bump" in the path
                    data = self._connector_data[connector_key]
                    min_offset = data.get('min_offset', 0.02 * 100)

                    # Determine the segment endpoints
                    all_points = [list(data['start'])] + waypoints + [list(data['end'])]
                    p1 = all_points[seg_idx]
                    p2 = all_points[seg_idx + 1]

                    # Create new waypoint(s) at click location
                    # For orthogonal routing, we add two points to maintain right angles
                    if abs(p1[0] - p2[0]) < 0.001:  # Vertical segment
                        # Add horizontal detour
                        new_wp1 = [click_x, p1[1] + (click_y - p1[1]) * 0.4]
                        new_wp2 = [click_x, p1[1] + (click_y - p1[1]) * 0.6]
                    else:  # Horizontal segment
                        # Add vertical detour
                        new_wp1 = [p1[0] + (click_x - p1[0]) * 0.4, click_y]
                        new_wp2 = [p1[0] + (click_x - p1[0]) * 0.6, click_y]

                    # Insert new waypoints
                    waypoints.insert(seg_idx, new_wp2)
                    waypoints.insert(seg_idx, new_wp1)

                    # Redraw
                    self.plot_gantt()
                    return

    def _handle_remove_waypoint(self, event):
        """Remove a waypoint from a connector."""
        for artist in self.control_point_artists:
            gid = artist.get_gid()
            if gid and '|wp_' in gid:
                contains, _ = artist.contains(event)
                if contains:
                    parts = gid.split('|')
                    from_task, to_task = parts[0], parts[1]
                    wp_idx = int(parts[2].replace('wp_', ''))
                    connector_key = (from_task, to_task)

                    # Get current waypoints
                    if connector_key not in self.connector_waypoints:
                        data = self._connector_data[connector_key]
                        self.connector_waypoints[connector_key] = {
                            'points': [list(wp) for wp in data['waypoints']]
                        }

                    waypoints = self.connector_waypoints[connector_key]['points']

                    # Need at least 2 waypoints for orthogonal routing
                    if len(waypoints) <= 2:
                        messagebox.showinfo("Cannot Remove",
                                            "Cannot remove waypoint. Minimum 2 waypoints required for orthogonal routing.")
                        return

                    # Remove the waypoint
                    waypoints.pop(wp_idx)

                    # Redraw
                    self.plot_gantt()
                    return

    def _on_mouse_release(self, event):
        """Handle mouse button release."""
        if self.dragging_point is not None:
            self.dragging_point = None
            # Redraw to clean up and finalize
            self.plot_gantt()

    def _on_mouse_motion(self, event):
        """Handle mouse motion for dragging control points."""
        if self.dragging_point is None:
            return
        if event.inaxes != self.ax:
            return

        connector_key, wp_idx, artist = self.dragging_point
        new_x, new_y = event.xdata, event.ydata

        # Apply snap-to-grid if enabled
        new_x, new_y = self._snap_coordinates(new_x, new_y)

        # Initialize waypoints if needed
        if connector_key not in self.connector_waypoints:
            if hasattr(self, '_connector_data') and connector_key in self._connector_data:
                data = self._connector_data[connector_key]
                self.connector_waypoints[connector_key] = {
                    'points': [list(wp) for wp in data['waypoints']]
                }
            else:
                return

        waypoints = self.connector_waypoints[connector_key]['points']

        # Update the waypoint position
        if 0 <= wp_idx < len(waypoints):
            waypoints[wp_idx] = [new_x, new_y]

        # Update artist position for immediate visual feedback
        artist.set_data([new_x], [new_y])

        # Update the connector path
        self._update_connector_path(connector_key)

        self.canvas.draw_idle()

    def _update_connector_path(self, connector_key):
        """Update an orthogonal connector's path after waypoint movement."""
        if connector_key not in self.connector_artists:
            return
        if connector_key not in self._connector_data:
            return

        data = self._connector_data[connector_key]

        # Get waypoints
        if connector_key in self.connector_waypoints:
            waypoints = self.connector_waypoints[connector_key].get('points', data['waypoints'])
        else:
            waypoints = data['waypoints']

        x_start, y_start = data['start']
        x_end, y_end = data['end']

        # Build path through all waypoints
        path_data = [(Path.MOVETO, (x_start, y_start))]
        for wp in waypoints:
            path_data.append((Path.LINETO, (wp[0], wp[1])))
        path_data.append((Path.LINETO, (x_end, y_end)))

        codes, verts = zip(*path_data)
        new_path = Path(verts, codes)

        # Update the path artist
        self.connector_artists[connector_key].set_path(new_path)

    # ------------------------------------------------------------------
    # MONTE CARLO INTEGRATION
    # ------------------------------------------------------------------
    def run_monte_carlo_ui(self):
        if not self.tasks:
            messagebox.showinfo("No Tasks", "Add some tasks first.")
            return

        n_sims = simpledialog.askinteger(
            "Monte Carlo Simulations",
            "Number of simulations to run:",
            minvalue=100,
            initialvalue=1000,
        )
        if not n_sims:
            return

        try:
            self.mc_results = run_monte_carlo(self.tasks, n_sims)
        except ValueError as e:
            messagebox.showerror("Monte Carlo Error", str(e))
            return

        fs = self.mc_results["finish_stats"]
        ci = self.mc_results["criticality"]
        ssi = self.mc_results["ssi"]

        project_start = min(t["start"] for t in self.tasks)

        def completion_date(duration_days):
            return (project_start + timedelta(days=duration_days)).strftime("%Y-%m-%d")

        nominal = fs["nominal"]
        mean_delta = fs["mean"] - nominal
        p40_delta = fs["p40"] - nominal
        p60_delta = fs["p60"] - nominal
        p90_delta = fs["p90"] - nominal

        msg_lines = [
            f"Project Start: {project_start.strftime('%Y-%m-%d')}",
            f"Nominal (no risk): {nominal:.1f}d → {completion_date(nominal)}",
            "",
            f"Simulations: {n_sims}",
            f"Mean:  {fs['mean']:.1f}d (Δ{mean_delta:+.1f}) → {completion_date(fs['mean'])}",
            f"P40:   {fs['p40']:.1f}d (Δ{p40_delta:+.1f}) → {completion_date(fs['p40'])}",
            f"P60:   {fs['p60']:.1f}d (Δ{p60_delta:+.1f}) → {completion_date(fs['p60'])}",
            f"P90:   {fs['p90']:.1f}d (Δ{p90_delta:+.1f}) → {completion_date(fs['p90'])}",
            "",
            "─" * 40,
            "Task Risk Analysis (sorted by combined CI+SSI):",
            f"{'Task':<12} {'CI':>6} {'SSI':>6} {'Combined':>8}",
            "─" * 40,
        ]

        # Sort by combined CI + SSI score
        combined_scores = []
        for name in ci.keys():
            ci_val = ci[name]
            ssi_val = ssi.get(name, 0)
            combined = ci_val + ssi_val
            combined_scores.append((name, ci_val, ssi_val, combined))

        combined_scores.sort(key=lambda x: x[3], reverse=True)

        for name, ci_val, ssi_val, combined in combined_scores:
            msg_lines.append(
                f"{name:<12} {ci_val * 100:>5.1f}% {ssi_val * 100:>5.1f}% {combined * 100:>7.1f}%"
            )

        msg_lines.append("")
        msg_lines.append("CI = Criticality Index (structural)")
        msg_lines.append("SSI = Schedule Sensitivity Index (statistical)")

        self.update_mc_summary(msg_lines)
        self.plot_gantt()


# ----------------------------------------------------------------------
# CORE MONTE CARLO ENGINE (no GUI dependencies)
# ----------------------------------------------------------------------
def run_monte_carlo(tasks, n_sims=1000):
    """
    Run Monte Carlo schedule simulation using calibrated I×P distributions.

    tasks: list of dicts
      {
        "name": str,
        "start": datetime,
        "end": datetime,
        "deps": [names],
        "prob": float in [0,1],   # Maps to distribution shape
        "impact": float in [0,1], # Maps to distribution scale
        "fail_mode": "early" or "late"  # Distribution skew behavior
      }

    Returns dict:
      {
        "criticality": {name: fraction of runs where task is critical},
        "ssi": {name: schedule sensitivity index (correlation)},
        "finish_stats": {"nominal", "mean", "p40", "p60", "p90"}
      }
    """
    if not tasks:
        raise ValueError("No tasks provided")

    name_to_idx = {}
    for i, t in enumerate(tasks):
        if t["name"] in name_to_idx:
            raise ValueError(
                f"Duplicate task name not allowed in Monte Carlo: {t['name']}"
            )
        name_to_idx[t["name"]] = i

    n = len(tasks)
    project_start = min(t["start"] for t in tasks)

    base_duration = [0.0] * n
    base_start_offset = [0.0] * n
    prob = [0.0] * n
    impact = [0.0] * n
    fail_mode = ["late"] * n
    preds = [[] for _ in range(n)]
    succs = [[] for _ in range(n)]

    for i, t in enumerate(tasks):
        d_days = (t["end"] - t["start"]).total_seconds() / 86400.0
        if d_days <= 0:
            raise ValueError(f"Non-positive duration for task {t['name']}")

        base_duration[i] = d_days
        base_start_offset[i] = (
                                       t["start"] - project_start
                               ).total_seconds() / 86400.0
        prob[i] = t.get("prob", 0.0)
        impact[i] = t.get("impact", 0.0)
        fail_mode[i] = t.get("fail_mode", "late")

        for dep_name in t.get("deps", []):
            if dep_name not in name_to_idx:
                raise ValueError(
                    f"Dependency {dep_name} not found for task {t['name']}"
                )
            j = name_to_idx[dep_name]
            preds[i].append(j)
            succs[j].append(i)

    # Topological sort
    indegree = [len(preds[i]) for i in range(n)]
    queue = [i for i in range(n) if indegree[i] == 0]
    topo_order = []

    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in succs[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)

    if len(topo_order) != n:
        raise ValueError(
            "Cycle detected in dependencies; Monte Carlo requires an acyclic task graph."
        )

    # --- Nominal deterministic schedule (no risk) ---
    es_nom = [0.0] * n
    ef_nom = [0.0] * n
    for i in topo_order:
        if preds[i]:
            es_nom[i] = max(ef_nom[j] for j in preds[i])
            es_nom[i] = max(es_nom[i], base_start_offset[i])
        else:
            es_nom[i] = base_start_offset[i]
        ef_nom[i] = es_nom[i] + base_duration[i]
    nominal_finish = max(ef_nom)

    # --- Monte Carlo with proper distributions ---
    critical_counts = [0] * n
    finish_times = []

    # Store sampled durations for SSI calculation
    duration_samples = [[] for _ in range(n)]  # duration_samples[task_idx] = list of sampled durations

    for sim in range(n_sims):
        dur = [0.0] * n

        # Sample durations from calibrated distributions
        for i in range(n):
            dur[i] = sample_task_duration(
                base_duration[i],
                prob[i],
                impact[i],
                fail_mode[i]
            )
            duration_samples[i].append(dur[i])

        # Forward pass: Early Start (ES), Early Finish (EF)
        es = [0.0] * n
        ef = [0.0] * n

        for i in topo_order:
            if preds[i]:
                es[i] = max(ef[j] for j in preds[i])
                es[i] = max(es[i], base_start_offset[i])
            else:
                es[i] = base_start_offset[i]
            ef[i] = es[i] + dur[i]

        project_finish = max(ef)
        finish_times.append(project_finish)

        # Backward pass: Late Finish (LF), Late Start (LS)
        lf = [0.0] * n
        ls = [0.0] * n

        for i in reversed(topo_order):
            if succs[i]:
                lf[i] = min(ls[j] for j in succs[i])
            else:
                lf[i] = project_finish
            ls[i] = lf[i] - dur[i]

        # Identify critical tasks (zero slack)
        for i in range(n):
            slack = ls[i] - es[i]
            if slack <= 1e-9:
                critical_counts[i] += 1

    # Compute Criticality Index (CI)
    criticality = {}
    for name, idx in name_to_idx.items():
        criticality[name] = critical_counts[idx] / float(n_sims)

    # Compute Schedule Sensitivity Index (SSI)
    # SSI = correlation between each task's duration and project finish time
    ssi = {}
    for name, idx in name_to_idx.items():
        corr = compute_correlation(duration_samples[idx], finish_times)
        # Clamp to [0, 1] since we care about positive correlation
        ssi[name] = max(0.0, corr)

    # Compute finish time statistics
    finish_times_sorted = sorted(finish_times)
    mean = sum(finish_times) / len(finish_times)

    def percentile(data, p):
        if not data:
            return 0.0
        k = (len(data) - 1) * p
        f = int(k)
        c = min(f + 1, len(data) - 1)
        if f == c:
            return data[f]
        return data[f] + (data[c] - data[f]) * (k - f)

    finish_stats = {
        "nominal": nominal_finish,
        "mean": mean,
        "p40": percentile(finish_times_sorted, 0.4),
        "p60": percentile(finish_times_sorted, 0.6),
        "p90": percentile(finish_times_sorted, 0.9),
    }

    return {
        "criticality": criticality,
        "ssi": ssi,
        "finish_stats": finish_stats,
    }


if __name__ == "__main__":
    root = tk.Tk()
    app = GanttApp(root)
    root.mainloop()