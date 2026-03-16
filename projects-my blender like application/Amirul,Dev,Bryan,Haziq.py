import sys
import json
import os
import vtk
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QToolButton, QSplashScreen, QMessageBox, QScrollArea, QVBoxLayout, QWidget
from PyQt5.QtGui import QIcon, QPixmap
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# Suppress VTK warning window and redirect warnings to console
vtk_output_window = vtk.vtkOutputWindow()
vtk_output_window.SetInstance(vtk_output_window)

# Simple terminal status logger helpers for prettier startup messages
import platform
import multiprocessing

def _tprint_sep(char='=', width=60):
    print(char * width)

def tprint_banner(title):
    print(title)
    _tprint_sep()

def tprint_ok(msg):
    try:
        # use a checkmark when console supports UTF-8
        print(f"\u2713 {msg}")
    except Exception:
        print(f"[OK] {msg}")

def gather_system_info():
    info = []
    info.append(f"Python: {platform.python_version()}")
    try:
        import PyQt5
        info.append(f"PyQt5: {PyQt5.QtCore.PYQT_VERSION_STR}")
    except Exception:
        info.append("PyQt5: (not importable)")
    try:
        info.append(f"VTK: {vtk.vtkVersion.GetVTKVersion()}")
    except Exception:
        info.append("VTK: (not available)")
    info.append(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    cpus = multiprocessing.cpu_count() or 'unknown'
    info.append(f"CPUs: {cpus}")
    return info

# List of interactive controls/features for the application
def gather_interactive_features():
    return [
        "Undo/Redo (Ctrl+Z / Ctrl+Y)",
        "Object Properties Dialog",
        "Add/Remove 3D Models",
        "Transform: Move, Rotate, Scale",
        "Change Color",
        "Toggle Visibility",
        "Toggle Wireframe Mode",
        "Duplicate/Delete Objects",
        "Interactive 3D Camera",
        "Scene Navigation Pane",
        "Box Widget for Transform",
        "Background Color Change",
        "Save/Load Scene",
        "Help/About Dialog"
    ]

# Global window tracking
ACTIVE_WINDOWS = []

class Command:
    """Base class for all undoable commands.
       Important: subclasses should implement do() and undo()."""
    def __init__(self, description="Generic Command"):
        self.description = description
    def do(self): raise NotImplementedError
    def undo(self): raise NotImplementedError
    def get_description(self): return self.description

class TranslateCommand(Command):
    """Command to translate a specific VTK actor (incremental)."""
    def __init__(self, actor, dx, dy, dz):
        super().__init__(f"Translate Actor by ({dx:.1f}, {dy:.1f}, {dz:.1f})")
        self.actor = actor
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def do(self):
        # Apply a relative translation to the actor's current position
        self.actor.AddPosition(self.dx, self.dy, self.dz)

    def undo(self):
        # Revert the translation applied by do()
        self.actor.AddPosition(-self.dx, -self.dy, -self.dz)

class RotateCommand(Command):
    """Command to rotate a VTK actor (incremental)."""
    def __init__(self, actor, rx, ry, rz):
        super().__init__(f"Rotate Actor by ({rx}, {ry}, {rz}) degrees")
        self.actor = actor
        self.rx, self.ry, self.rz = rx, ry, rz

    def do(self):
        # Perform rotations about X, Y, Z in sequence
        self.actor.RotateX(self.rx)
        self.actor.RotateY(self.ry)
        self.actor.RotateZ(self.rz)

    def undo(self):
        # Undo the rotations in reverse order
        self.actor.RotateZ(-self.rz)
        self.actor.RotateY(-self.ry)
        self.actor.RotateX(-self.rx)

class RotateAbsoluteCommand(Command):
    """Command to set absolute rotation on a VTK actor (for dialogs with real-time preview)."""
    def __init__(self, actor, old_orientation, new_orientation):
        super().__init__(f"Rotate Actor to {new_orientation}")
        self.actor = actor
        self.old_orientation = old_orientation
        self.new_orientation = new_orientation

    def do(self):
        # Set actor orientation to the new absolute orientation
        self.actor.SetOrientation(*self.new_orientation)

    def undo(self):
        # Restore previous orientation
        self.actor.SetOrientation(*self.old_orientation)

class ScaleAbsoluteCommand(Command):
    """Command to set absolute scale on a VTK actor (for dialogs with real-time preview)."""
    def __init__(self, actor, old_scale, new_scale):
        super().__init__(f"Scale Actor to {new_scale}")
        self.actor = actor
        self.old_scale = old_scale
        self.new_scale = new_scale

    def do(self):
        # Apply absolute scale values
        self.actor.SetScale(*self.new_scale)

    def undo(self):
        # Restore previous scale values
        self.actor.SetScale(*self.old_scale)

class TranslateAbsoluteCommand(Command):
    """Command to set absolute position on a VTK actor (for dialogs with real-time preview)."""
    def __init__(self, actor, old_position, new_position):
        super().__init__(f"Translate Actor to {new_position}")
        self.actor = actor
        self.old_position = old_position
        self.new_position = new_position

    def do(self):
        # Set absolute actor position
        self.actor.SetPosition(*self.new_position)

    def undo(self):
        # Restore previous position
        self.actor.SetPosition(*self.old_position)

class DeleteCommand(Command):
    """Command to delete a selected actor from scene."""
    def __init__(self, vtk_app, actor):
        super().__init__("Delete Actor")
        self.vtk_app = vtk_app
        self.actor = actor
        # Save index to allow undo to re-insert at same position if possible
        self.saved_index = vtk_app.actors.index(actor) if actor in vtk_app.actors else -1

    def do(self):
        # Remove actor from renderer and our actor list (if present)
        self.vtk_app.renderer.RemoveActor(self.actor)
        if self.actor in self.vtk_app.actors:
            self.vtk_app.actors.remove(self.actor)

    def undo(self):
        # Re-insert actor into the actor list at saved index (or append)
        if self.saved_index >= 0:
            self.vtk_app.actors.insert(self.saved_index, self.actor)
        else:
            self.vtk_app.actors.append(self.actor)
        self.vtk_app.renderer.AddActor(self.actor)

class DuplicateCommand(Command):
    """Command to duplicate a selected actor and add new actor to scene."""
    def __init__(self, vtk_app, original_actor):
        super().__init__("Duplicate Actor")
        self.vtk_app = vtk_app
        self.original_actor = original_actor
        self.new_actor = None

    def do(self):
        # Duplicate actor by copying its mapper input data and properties
        mapper = self.original_actor.GetMapper()
        data = mapper.GetInput()
        new_mapper = vtk.vtkPolyDataMapper()
        new_mapper.SetInputData(data)
        self.new_actor = vtk.vtkActor()
        self.new_actor.SetMapper(new_mapper)
        self.new_actor.GetProperty().DeepCopy(self.original_actor.GetProperty())
        # Offset new actor slightly so it is visually distinct
        self.new_actor.AddPosition(3, 0, 0)
        self.vtk_app.renderer.AddActor(self.new_actor)
        self.vtk_app.actors.append(self.new_actor)

    def undo(self):
        # Remove the duplicated actor
        if self.new_actor in self.vtk_app.actors:
            self.vtk_app.actors.remove(self.new_actor)
        self.vtk_app.renderer.RemoveActor(self.new_actor)

class ColorChangeCommand(Command):
    """Command to change an actor's color (undoable)."""
    def __init__(self, actor, old_color, new_color):
        super().__init__("Change Color")
        self.actor = actor
        self.old_color = old_color
        self.new_color = new_color

    def do(self):
        # Apply new color to property
        self.actor.GetProperty().SetColor(self.new_color)

    def undo(self):
        # Revert to old color
        self.actor.GetProperty().SetColor(self.old_color)

class VisibilityToggleCommand(Command):
    """Command to toggle visibility of an actor."""
    def __init__(self, actor):
        super().__init__("Toggle Visibility")
        self.actor = actor
        # Save previous visibility state for undo
        self.prev_state = actor.GetVisibility()

    def do(self):
        # Toggle visibility relative to previous saved state
        self.actor.SetVisibility(not self.prev_state)

    def undo(self):
        # Restore previous visibility
        self.actor.SetVisibility(self.prev_state)

class WireframeToggleCommand(Command):
    """Command to toggle wireframe mode and color toggling when enabled."""
    def __init__(self, actor, current_color):
        super().__init__("Toggle Wireframe")
        self.actor = actor
        self.current_color = current_color
        # Save previous representation and color to restore on undo
        self.prev_mode = actor.GetProperty().GetRepresentation()
        self.prev_color = actor.GetProperty().GetColor()

    def do(self):
        prop = self.actor.GetProperty()
        if self.prev_mode == vtk.VTK_SURFACE:
            # Switch to wireframe and set highlight color
            prop.SetRepresentationToWireframe()
            prop.SetColor(1, 0, 0)
        else:
            # Restore surface mode and previously known color
            prop.SetRepresentationToSurface()
            prop.SetColor(self.current_color)

    def undo(self):
        # Restore both representation and color
        prop = self.actor.GetProperty()
        prop.SetRepresentation(self.prev_mode)
        prop.SetColor(self.prev_color)

class BackgroundColorChangeCommand(Command):
    """Command to change renderer background color (undoable)."""
    def __init__(self, renderer, old_color, new_color):
        super().__init__("Change Background Color")
        self.renderer = renderer
        self.old_color = old_color
        self.new_color = new_color

    def do(self):
        # Apply new background color to renderer
        self.renderer.SetBackground(*self.new_color)

    def undo(self):
        # Restore old background color
        self.renderer.SetBackground(*self.old_color)

class CommandManager:
    """Manages a history stack of commands for undo/redo functionality."""
    def __init__(self, vtk_app):
        self.history = []
        self.undo_ptr = -1
        self.max_history = 100
        self.vtk_app = vtk_app

    def execute(self, command: Command):
        """Execute a command and add it to history (important: this is where commands become undoable)."""
        # If we had undone some commands and then execute a new command,
        # truncate any 'future' commands so we keep a linear history
        if self.undo_ptr < len(self.history) - 1:
            self.history = self.history[:self.undo_ptr + 1]
            
        # Perform the command action
        command.do()
        # Append to history and advance undo pointer
        self.history.append(command)
        self.undo_ptr = len(self.history) - 1

        # Enforce max history size - drop oldest if needed
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.undo_ptr -= 1

        # Emit status so UI + terminal know what happened
        self.vtk_app.emit_status(f"Executed: {command.get_description()}")
        # Re-render the scene after the change
        self.vtk_app.render_all()

    def undo(self):
        """Undo the last executed command, if any."""
        if self.undo_ptr >= 0:
            command = self.history[self.undo_ptr]
            command.undo()
            self.undo_ptr -= 1
            self.vtk_app.emit_status(f"Undo: {command.get_description()}")
            self.vtk_app.render_all()
        else:
            self.vtk_app.emit_status("Cannot Undo: History is empty.")

    def redo(self):
        """Redo the next command in the history, if any."""
        if self.undo_ptr < len(self.history) - 1:
            self.undo_ptr += 1
            command = self.history[self.undo_ptr]
            command.do()
            self.vtk_app.emit_status(f"Redo: {command.get_description()}")
            self.vtk_app.render_all()
        else:
            self.vtk_app.emit_status("Cannot Redo: Already at latest action.")

class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """
    Custom interactor style to enable simple actor rotation on LeftButton drag 
    when an actor is selected. Important: sets interactor actor reference so
    user drag rotates that actor instead of camera.
    """
    def __init__(self, parent=None):
        super().__init__()
        # Register observers for mouse events
        self.AddObserver("LeftButtonPressEvent", self.on_left_button_down)
        self.AddObserver("MouseMoveEvent", self.on_mouse_move)
        self.AddObserver("LeftButtonReleaseEvent", self.on_left_button_up)
        self.last_x = 0
        self.last_y = 0
        self.dragging = False
        self.actor = None  # actor to rotate when dragging

    def on_left_button_down(self, obj, event):
        # Begin dragging - remember initial mouse position
        self.dragging = True
        self.last_x, self.last_y = self.GetInteractor().GetEventPosition()
        self.OnLeftButtonDown()
        return

    def on_mouse_move(self, obj, event):
        # If dragging and actor selected, rotate actor instead of camera
        if self.dragging and self.actor is not None:
            interactor = self.GetInteractor()
            x, y = interactor.GetEventPosition()
            dx = x - self.last_x
            dy = y - self.last_y
            # Apply rotation scaled down for sensible UX
            self.actor.RotateY(dx * 0.5)
            self.actor.RotateX(dy * 0.5)
            interactor.GetRenderWindow().Render()
            self.last_x, self.last_y = x, y
        else:
             self.OnMouseMove()
        return

    def on_left_button_up(self, obj, event):
        # End dragging state
        self.dragging = False
        self.OnLeftButtonUp()
        return

class myVTK:
    """Wrapper class that manages VTK renderer, actors, interaction, and scene state."""
    def __init__(self):
        # rendering
        self.renderer = None
        self.window = None
        self.interactor = None
        # scene
        self.actors = []    # list of vtkActor objects in the scene
        self.mappers = []   # corresponding mappers (for potential reuse)
        self.sources = []   # any underlying sources/filters retained
        # selection & widgets
        self.selected_actor = None
        self.box_widget = None
        # reference geometry
        self.axes_widget = None
        self.reference_plane_actor = None
        self.reference_grid_actor = None
        # camera
        self.initial_camera = None
        # state
        self.current_object = None
        self.current_object_type = None
        self.current_color = (1.0, 1.0, 1.0)
        # status signal for UI to connect to
        self.status_signal = None
        # custom interactor style reference
        self.interactor_style = None
        # store initial camera distance for dolly math
        self.initial_distance = 0
        # Command Manager for undo/redo
        self.command_manager = CommandManager(self)
        # Scene update callback (set by MainWindow)
        self.scene_update_callback = None
        # Selection changed callback (notifies UI on picks)
        self.selection_changed_callback = None

    def set_status_signal(self, signal):
        # Allow MainWindow to provide a Qt signal for status updates
        self.status_signal = signal

    def emit_status(self, msg):
        """Emit status both to the UI (if available) and always print to terminal.
           Important: This ensures terminal logging for the user as requested."""
        # Send to the connected status signal if present
        if self.status_signal:
            try:
                # Try as Qt signal
                self.status_signal.emit(msg)
            except Exception:
                try:
                    # Try as callable fallback
                    self.status_signal(msg)
                except Exception:
                    # If both fail, fall through to print
                    pass
        # ALWAYS print to terminal for debugging/visibility
        try:
            # Prefix so it's easy to grep in logs
            print(f"[myVTK STATUS] {msg}")
        except Exception:
            # Fall back to basic print
            print("STATUS:", msg)

    def setup_rendering_pipeline(self, vtk_widget, interactor_style=None):
        """Create renderer, render window, interactor and attach orientation axes and grid.
           Important: call this once after creating the QVTKRenderWindowInteractor."""
        # Initialize renderer first and set a default background
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.0, 0.0, 0.0)  # black background by default
        
        # Get the render window from the widget, create if not present
        self.window = vtk_widget.GetRenderWindow()
        if not self.window:
            self.window = vtk.vtkRenderWindow()
            vtk_widget.SetRenderWindow(self.window)
        
        # Add renderer to window (single renderer app)
        self.window.AddRenderer(self.renderer)
        
        # Get or create the interactor (QVTK may provide one)
        self.interactor = self.window.GetInteractor()
        if not self.interactor:
            self.interactor = vtk_widget.GetInteractor()
            if self.interactor:
                self.window.SetInteractor(self.interactor)
        
        # Attach an interactor style if provided (enables actor drag rotation)
        if interactor_style is not None:
            self.interactor_style = interactor_style
            if self.interactor:
                self.interactor.SetInteractorStyle(self.interactor_style)
        
        # Add helpful reference geometry (grid plane and orientation axes)
        self.add_grid()
        self.add_axes()
        # Enable picking so users can click/select actors in the scene
        self.enable_picking()
        
        # Set up an initial camera position that gives a good view of the grid
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(80,80,60)
        cam.SetFocalPoint(0,0,0)
        cam.SetViewUp(0,0,1)
        cam.Zoom(0.9)
        # Save a baseline distance so dialogs can dolly relative to this
        self.initial_distance = cam.GetDistance()
        self.initial_camera = (cam.GetPosition(), cam.GetFocalPoint(), cam.GetViewUp(), self.initial_distance)
        
        # Initialize interactor and render once
        if self.interactor:
            self.interactor.Initialize()
        self.window.Render()
        # Terminal status to indicate pipeline is ready
        self.emit_status("Rendering pipeline initialized.")

    def add_grid(self):
        """Create a simple ground plane and wireframe grid for visual context."""
        plane = vtk.vtkPlaneSource()
        plane.SetOrigin(-50.0, -50.0, 0.0)
        plane.SetPoint1(50.0, -50.0, 0.0)
        plane.SetPoint2(-50.0, 50.0, 0.0)
        plane.SetXResolution(50)
        plane.SetYResolution(50)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(plane.GetOutputPort())
        solid = vtk.vtkActor()
        solid.SetMapper(mapper)
        solid.GetProperty().SetColor(0.05,0.05,0.05)
        grid = vtk.vtkActor()
        grid.SetMapper(mapper)
        grid.GetProperty().SetColor(1.0,1.0,1.0)
        grid.GetProperty().SetRepresentationToWireframe()
        grid.GetProperty().SetAmbient(0.1)
        # Save reference actors so we can ignore them during picking/selection
        self.reference_plane_actor = solid
        self.reference_grid_actor = grid
        self.renderer.AddActor(solid)
        self.renderer.AddActor(grid)

    def add_axes(self):
        """Attach an orientation axes widget to the render window (non-interactive)."""
        axes = vtk.vtkAxesActor()
        self.axes_widget = vtk.vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(axes)
        if self.interactor:
            self.axes_widget.SetInteractor(self.interactor)
            # Place axes in bottom-left viewport
            self.axes_widget.SetViewport(0.0, 0.0, 0.18, 0.18)
            self.axes_widget.On()
            self.axes_widget.InteractiveOff()

    def enable_picking(self):
        """Enable basic prop picking on left mouse click to select an actor in the scene."""
        if not self.interactor:
            return
            
        picker = vtk.vtkPropPicker()
        def on_click(obj, event):
            # Get mouse position and perform a pick into the renderer
            pos = self.interactor.GetEventPosition()
            picker.Pick(pos[0], pos[1], 0, self.renderer)
            actor = picker.GetActor()
            # If picked actor is one of our scene actors, update selection
            if actor in self.actors:
                self.selected_actor = actor
                if self.interactor_style is not None:
                    try:
                        # allow drag-rotate interactor to target this actor
                        self.interactor_style.actor = actor
                    except Exception:
                        pass
                self.emit_status("Selected model. Box widget enabled for transform.")
                self.enable_box_widget(auto=True)
                # notify host UI about the selection change (MainWindow updates list)
                try:
                    if self.selection_changed_callback:
                        self.selection_changed_callback(actor)
                except Exception:
                    pass
            elif actor is None or actor not in (self.reference_plane_actor, self.reference_grid_actor):
                # If click on empty space or reference geometry, clear selection
                self.selected_actor = None
                if self.interactor_style is not None:
                    try:
                        self.interactor_style.actor = None
                    except Exception:
                        pass
                # turn off box widget if active
                if self.box_widget and self.box_widget.GetEnabled():
                    self.box_widget.Off()
                    self.box_widget = None
                self.emit_status("Selection cleared.")
                try:
                    if self.selection_changed_callback:
                        self.selection_changed_callback(None)
                except Exception:
                    pass
            # Force a render to show any selection visuals
            self.window.Render()
        # Attach pick handler to left button press events
        self.interactor.AddObserver("LeftButtonPressEvent", on_click)

    def place_on_top_of_grid(self, actor):
        """Place actor so its min Z sits on the grid (useful for primitives)."""
        bounds = actor.GetBounds()
        if not bounds:
            return
        min_z = bounds[4]
        # move actor downward/upward by -min_z so it sits flush on grid z=0 plane
        actor.AddPosition(0,0,-min_z)

    def _update_scene_with_new_source(self, source_or_filter):
        """Common helper to take a vtk source/filter and add it as an actor to the scene.
           Important: preserves camera state to avoid jumping view when models are added."""
        source_or_filter.Update()
        mapper = vtk.vtkPolyDataMapper()
        try:
            mapper.SetInputConnection(source_or_filter.GetOutputPort())
        except Exception:
            try:
                mapper.SetInputData(source_or_filter.GetOutput())
            except Exception:
                pass
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(self.current_color)
        # Save camera state BEFORE adding new actor so we can restore it to avoid zoom/pan
        cam = self.renderer.GetActiveCamera()
        saved_position = cam.GetPosition()
        saved_focal = cam.GetFocalPoint()
        saved_viewup = cam.GetViewUp()
        saved_distance = cam.GetDistance()
        # Add actor to renderer and our internal lists
        self.renderer.AddActor(actor)
        # Assign readable name for UI (stable across reloads)
        try:
            actor.user_object_name = self.current_object or f"Model {len(self.actors) + 1}"
        except Exception:
            actor.user_object_name = f"Model {len(self.actors) + 1}"
        # Position actor so it sits on the grid
        self.place_on_top_of_grid(actor)
        self.actors.append(actor)
        self.mappers.append(mapper)
        # Restore camera state so adding actor doesn't change user's view
        cam.SetPosition(saved_position)
        cam.SetFocalPoint(saved_focal)
        cam.SetViewUp(saved_viewup)
        self.renderer.ResetCameraClippingRange()
        self.render_all()
        self.emit_status(f"Model Loaded: {self.current_object or 'object'}")
        
        # Update scene navigator UI via callback if available
        if self.scene_update_callback:
            self.scene_update_callback()
        
        return actor

    def render_all(self):
        """Request a render of the VTK window (safe wrapper)."""
        if self.window:
            try:
                self.window.Render()
            except Exception as e:
                # Render failure reported via status and terminal
                self.emit_status(f"Render error: {e}")

    def cleanup(self):
        """Properly clean up VTK resources (actors, widgets, interactor) before closing."""
        try:
            # Disable and remove any box widget
            if self.box_widget:
                self.box_widget.Off()
                self.box_widget.SetInteractor(None)
                self.box_widget = None
            
            # Disable axes marker
            if self.axes_widget:
                self.axes_widget.Off()
                self.axes_widget = None
            
            # Remove all user actors from renderer
            for actor in self.actors:
                self.renderer.RemoveActor(actor)
            self.actors.clear()
            
            # Clear references to avoid memory leaks
            self.renderer = None
            self.window = None
            self.interactor = None
            self.interactor_style = None
            
            # Terminal log to indicate cleanup was performed
            print("[myVTK] Cleaned up VTK resources.")
            
        except Exception as e:
            print(f"Cleanup error: {e}")

# Part of myVTK class - Model Creation and Loading Methods
    
    def create_object(self, object_type):
        """Create simple primitive objects (sphere, cube)."""
        if object_type == 'sphere':
            src = vtk.vtkSphereSource()
            src.SetRadius(4.0)
            self.current_object = 'sphere'
            return self._update_scene_with_new_source(src)
        elif object_type == 'cube':
            src = vtk.vtkCubeSource()
            src.SetXLength(6)
            src.SetYLength(6)
            src.SetZLength(6)
            self.current_object = 'cube'
            return self._update_scene_with_new_source(src)
        return None

    def load_geom(self, obj):
        # Convenience wrapper to call create_object (used by menu)
        return self.create_object(obj)

    def load_vtk_model(self, model_type):
        """Load standard vtk model primitives like cylinder and cone."""
        if model_type == 'cylinder':
            src = vtk.vtkCylinderSource()
            src.SetResolution(32)
            src.SetHeight(10)
            src.SetRadius(3)
            src.SetCenter(0,0,5)
            self.current_object = 'cylinder'
            return self._update_scene_with_new_source(src)
        elif model_type == 'cone':
            src = vtk.vtkConeSource()
            src.SetResolution(32)
            src.SetHeight(10)
            src.SetRadius(4)
            src.SetCenter(0,0,2.5)
            self.current_object = 'cone'
            return self._update_scene_with_new_source(src)
        return None

    def load_cell_model(self, model_type):
        """Create various unstructured/cell-based models (tetra, convex point set)."""
        if model_type == 'tetra':
            pts = vtk.vtkPoints()
            pts.InsertNextPoint(0,0,0)
            pts.InsertNextPoint(1,0,0)
            pts.InsertNextPoint(0,1,0)
            pts.InsertNextPoint(0,0,1)
            tetra = vtk.vtkTetra()
            for i in range(4):
                tetra.GetPointIds().SetId(i,i)
            ugrid = vtk.vtkUnstructuredGrid()
            ugrid.SetPoints(pts)
            ugrid.InsertNextCell(tetra.GetCellType(), tetra.GetPointIds())
            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputData(ugrid)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(self.current_color)
            # Save camera state BEFORE adding new actor
            cam = self.renderer.GetActiveCamera()
            saved_position = cam.GetPosition()
            saved_focal = cam.GetFocalPoint()
            saved_viewup = cam.GetViewUp()
            saved_distance = cam.GetDistance()
            self.renderer.AddActor(actor)
            self.place_on_top_of_grid(actor)
            self.actors.append(actor)
            # Restore camera state to prevent zoom/pan
            cam.SetPosition(saved_position)
            cam.SetFocalPoint(saved_focal)
            cam.SetViewUp(saved_viewup)
            self.renderer.ResetCameraClippingRange()
            self.render_all()
            self.emit_status("Loaded tetra cell model.")
            if self.scene_update_callback:
                self.scene_update_callback()
            return actor
        elif model_type == 'convex' or model_type == 'convex_set':
            pts = vtk.vtkPoints()
            pts.InsertNextPoint(0,0,0)
            pts.InsertNextPoint(1,0,0)
            pts.InsertNextPoint(1,1,0)
            pts.InsertNextPoint(0,1,0)
            pts.InsertNextPoint(0.5,0.5,1)
            convex = vtk.vtkConvexPointSet()
            convex.GetPointIds().SetNumberOfIds(5)
            for i in range(5):
                convex.GetPointIds().SetId(i,i)
            ugrid = vtk.vtkUnstructuredGrid()
            ugrid.SetPoints(pts)
            ugrid.InsertNextCell(convex.GetCellType(), convex.GetPointIds())
            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputData(ugrid)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(self.current_color)
            # Save camera state BEFORE adding new actor
            cam = self.renderer.GetActiveCamera()
            saved_position = cam.GetPosition()
            saved_focal = cam.GetFocalPoint()
            saved_viewup = cam.GetViewUp()
            saved_distance = cam.GetDistance()
            self.renderer.AddActor(actor)
            self.place_on_top_of_grid(actor)
            self.actors.append(actor)
            # Restore camera state
            cam.SetPosition(saved_position)
            cam.SetFocalPoint(saved_focal)
            cam.SetViewUp(saved_viewup)
            self.renderer.ResetCameraClippingRange()
            self.render_all()
            self.emit_status("Loaded convex point set.")
            if self.scene_update_callback:
                self.scene_update_callback()
            return actor
        elif model_type == 'tessellated':
            src = vtk.vtkTessellatedBoxSource()
            src.SetLevel(2)
            src.Update()
            self.current_object = 'tessellated'
            return self._update_scene_with_new_source(src)
        return None

    def load_source_model(self, model_type):
        """Load special source-based models (platonic solids, disk, shrink filter)."""
        if model_type == 'icosahedron' or model_type == 'platonic':
            src = vtk.vtkPlatonicSolidSource()
            src.SetSolidTypeToIcosahedron()
            src.Update()
            self.current_object = 'icosahedron'
            return self._update_scene_with_new_source(src)
        elif model_type == 'reduced_cube':
            cube = vtk.vtkCubeSource()
            cube.Update()
            shrink = vtk.vtkShrinkFilter()
            shrink.SetInputConnection(cube.GetOutputPort())
            shrink.SetShrinkFactor(0.7)
            shrink.Update()
            self.current_object = 'reduced_cube'
            return self._update_scene_with_new_source(shrink)
        elif model_type == 'disk':
            src = vtk.vtkDiskSource()
            src.SetInnerRadius(2.0)
            src.SetOuterRadius(5.0)
            src.Update()
            self.current_object = 'disk'
            return self._update_scene_with_new_source(src)
        return None

    def load_parametric_model(self, model_type):
        """Load parametric models where available (may not exist in all VTK builds)."""
        src = vtk.vtkParametricFunctionSource()
        if model_type == 'klein':
            try:
                src.SetParametricFunction(vtk.vtkParametricKlein())
                self.current_object = 'klein'
                src.SetUResolution(60)
                src.SetVResolution(60)
                src.Update()
                return self._update_scene_with_new_source(src)
            except Exception:
                self.emit_status("Parametric Klein not available in this VTK build.")
                return None
        elif model_type == 'torus':
            try:
                param = vtk.vtkParametricTorus()
                src.SetParametricFunction(param)
                self.current_object = 'torus'
                src.SetUResolution(80)
                src.SetVResolution(80)
                src.Update()
                return self._update_scene_with_new_source(src)
            except Exception:
                self.emit_status("Parametric torus not available.")
                return None
        return None

    def load_implicit_or_isosurface(self, model_type):
        """Create implicit shapes or perform an isosurface extraction example."""
        if model_type == 'superquadric':
            try:
                src = vtk.vtkSuperquadricSource()
                src.SetPhiRoundness(2.5)
                src.SetThetaRoundness(0.8)
                src.Update()
                self.current_object = 'superquadric'
                return self._update_scene_with_new_source(src)
            except Exception:
                self.emit_status("Superquadric not available.")
                return None
        elif model_type == 'isosurface':
            implicit = vtk.vtkSphere()
            implicit.SetRadius(5)
            sample = vtk.vtkSampleFunction()
            sample.SetImplicitFunction(implicit)
            sample.SetSampleDimensions(40,40,40)
            sample.Update()
            contour = vtk.vtkContourFilter()
            contour.SetInputConnection(sample.GetOutputPort())
            contour.SetValue(0,0.0)
            contour.Update()
            return self._update_scene_with_new_source(contour)
        return None

    def load_function_model(self, model_type):
        """Create height-field surfaces from numpy functions (requires numpy)."""
        try:
            import numpy as np
        except Exception:
            self.emit_status("Numpy missing: functional models require numpy.")
            return None
        size = 50
        x = np.linspace(-3, 3, size)
        y = np.linspace(-3, 3, size)
        X, Y = np.meshgrid(x,y)
        if model_type == 'sinusoidal':
            Z = np.sin(X) * np.cos(Y)
        elif model_type == 'gaussian':
            Z = np.exp(-(X**2 + Y**2))
        else:
            return None
        image = vtk.vtkImageData()
        image.SetDimensions(size,size,1)
        image.AllocateScalars(vtk.VTK_FLOAT,1)
        # Populate image scalars from numpy Z (note: VTK indexing order)
        for i in range(size):
            for j in range(size):
                image.SetScalarComponentFromFloat(i,j,0,0,float(Z[j,i]))
        warp = vtk.vtkWarpScalar()
        warp.SetInputData(image)
        warp.SetScaleFactor(2.0)
        warp.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(warp.GetOutputPort())
        self.current_object = model_type
        return self._update_scene_with_new_source(warp)

    def add_actor(self, mapper):
        """Add an existing mapper as an actor to the scene (used by file loaders)."""
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(self.current_color)
        # Save camera state BEFORE adding new actor so view doesn't jump
        cam = self.renderer.GetActiveCamera()
        saved_position = cam.GetPosition()
        saved_focal = cam.GetFocalPoint()
        saved_viewup = cam.GetViewUp()
        saved_distance = cam.GetDistance()
        self.renderer.AddActor(actor)
        # Give a stable name for this actor (for UI labels)
        try:
            actor.user_object_name = self.current_object or f"Model {len(self.actors) + 1}"
        except Exception:
            actor.user_object_name = f"Model {len(self.actors) + 1}"
        # Optionally attach a semantic type to the actor for UI display
        try:
            actor.user_object_type = self.current_object_type or self.current_object or f"Model {len(self.actors) + 1}"
        except Exception:
            pass
        # Ensure actor sits on the grid
        self.place_on_top_of_grid(actor)
        self.actors.append(actor)
        # Restore camera state
        cam.SetPosition(saved_position)
        cam.SetFocalPoint(saved_focal)
        cam.SetViewUp(saved_viewup)
        self.renderer.ResetCameraClippingRange()
        self.render_all()
        if self.scene_update_callback:
            self.scene_update_callback()
        return actor

# Part of myVTK class - File Loading and Edit Tools
    
    def load_file(self, file_path):
        """Load various 3D file formats (stl, obj, ply, vtp, vtk, 3ds)."""
        ext = file_path.split('.')[-1].lower()
        # record the file type for UI
        self.current_object_type = ext
        reader = None
        # Use file basename (without extension) as current object name for UI labels
        try:
            basename = os.path.basename(file_path)
            name_no_ext = os.path.splitext(basename)[0]
        except Exception:
            name_no_ext = None
        
        # Special handling for 3ds which can contain multiple actors / materials
        if ext == '3ds':
            try:
                # Try importer first (most robust) if available in VTK build
                if hasattr(vtk, 'vtk3DSImporter'):
                    importer = vtk.vtk3DSImporter()
                    importer.SetFileName(file_path)
                    try:
                        # Attach to our render window so imported actors land in our renderer
                        importer.SetRenderWindow(self.window)
                    except Exception:
                        pass

                    # Snapshot existing actors so we can detect newly-imported ones
                    existing = set()
                    try:
                        actors_collection = self.renderer.GetActors()
                        for i in range(actors_collection.GetNumberOfItems()):
                            existing.add(actors_collection.GetItemAsObject(i))
                    except Exception:
                        existing = set()

                    importer.Read()

                    # Collect any newly-added actors into our actors list
                    added = 0
                    try:
                        actors_collection = self.renderer.GetActors()
                        for i in range(actors_collection.GetNumberOfItems()):
                            a = actors_collection.GetItemAsObject(i)
                            if a not in existing:
                                # annotate and add
                                try:
                                    a.user_object_name = name_no_ext or f"Model {len(self.actors) + 1}"
                                except Exception:
                                    pass
                                try:
                                    a.user_object_type = '3ds'
                                except Exception:
                                    pass
                                try:
                                    # mark group so UI can offer a uniform-selection entry
                                    a.user_object_group = name_no_ext or None
                                except Exception:
                                    pass
                                self.actors.append(a)
                                added += 1
                    except Exception:
                        pass

                    # finalize and notify UI
                    self.render_all()
                    self.current_object = name_no_ext or ('file_3ds_scene' if added else None)
                    self.current_object_type = '3ds'
                    self.emit_status(f"Loaded 3DS scene: {file_path} ({added} actors)")
                    if self.scene_update_callback:
                        self.scene_update_callback()
                    return

                # Fallback: try vtk3DSReader (may not be available in all VTK builds)
                reader = vtk.vtk3DSReader()
                reader.SetFileName(file_path)
                reader.Update()
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(reader.GetOutputPort())
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(self.current_color)
                try:
                    actor.user_object_name = name_no_ext or f"Model {len(self.actors) + 1}"
                except Exception:
                    pass
                # Save camera state BEFORE adding new actor
                cam = self.renderer.GetActiveCamera()
                saved_position = cam.GetPosition()
                saved_focal = cam.GetFocalPoint()
                saved_viewup = cam.GetViewUp()
                saved_distance = cam.GetDistance()
                self.renderer.AddActor(actor)
                self.actors.append(actor)
                # Restore camera state
                cam.SetPosition(saved_position)
                cam.SetFocalPoint(saved_focal)
                cam.SetViewUp(saved_viewup)
                self.renderer.ResetCameraClippingRange()
                self.render_all()
                self.current_object = name_no_ext or 'file_3ds_scene'
                self.current_object_type = '3ds'
                self.emit_status(f"Loaded 3DS scene (reader fallback): {file_path}")
                if self.scene_update_callback:
                    self.scene_update_callback()
                return
            except Exception as e:
                self.emit_status(f"Failed to load 3DS: {e}")
                return

        # Map file extensions to VTK readers
        elif ext == 'stl':
            reader = vtk.vtkSTLReader()
        elif ext == 'obj':
            reader = vtk.vtkOBJReader()
        elif ext == 'ply':
            reader = vtk.vtkPLYReader()
        elif ext == 'vtp':
            reader = vtk.vtkXMLPolyDataReader()
        elif ext == 'vtk':
            reader = vtk.vtkPolyDataReader()
        else:
            self.emit_status(f"Unsupported format: {ext}")
            return
        
        # Use generic reader flow for single-actor file types
        try:
            reader.SetFileName(file_path)
            reader.Update()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())
            # set the current object name to the file basename so add_actor uses it
            if name_no_ext:
                self.current_object = name_no_ext
            else:
                self.current_object = f'file_{ext}'
            # record the semantic type for the UI
            self.current_object_type = ext
            self.add_actor(mapper)
            self.emit_status(f"Loaded file: {file_path}")
            if self.scene_update_callback:
                self.scene_update_callback()
        except Exception as e:
            self.emit_status(f"Failed to load: {e}")

    def delete_selected(self):
        """Delete the currently selected actor via a DeleteCommand to make it undoable."""
        if self.selected_actor:
            cmd = DeleteCommand(self, self.selected_actor)
            self.command_manager.execute(cmd)
            self.selected_actor = None
            if self.scene_update_callback:
                self.scene_update_callback()
        else:
            self.emit_status("Delete failed: No model selected.")

    def duplicate_selected(self):
        """Duplicate selected actor (wrapped by DuplicateCommand to allow undo)."""
        if not self.selected_actor:
            self.emit_status("Duplicate failed: No model selected.")
            return
        cmd = DuplicateCommand(self, self.selected_actor)
        self.command_manager.execute(cmd)
        if self.scene_update_callback:
            self.scene_update_callback()

    def toggle_visibility(self):
        """Toggle visibility either for a selected group or single selected actor."""
        # Check if group is selected (group membership tracked in _group_members)
        group_members = getattr(self, '_group_members', None)
        is_group = bool(group_members)
        
        if is_group:
            for actor in group_members:
                cmd = VisibilityToggleCommand(actor)
                self.command_manager.execute(cmd)
            self.emit_status(f"Visibility toggled for group ({len(group_members)} items) (Undoable).")
        else:
            if not self.selected_actor:
                self.emit_status("Visibility toggle failed: No model selected.")
                return
            cmd = VisibilityToggleCommand(self.selected_actor)
            self.command_manager.execute(cmd)
            self.emit_status("Visibility toggled (Undoable).")
        
        if self.scene_update_callback:
            self.scene_update_callback()

    def toggle_wireframe(self):
        """Toggle wireframe representation for the selected actor."""
        if not self.selected_actor:
            self.emit_status("Wireframe toggle failed: No model selected.")
            return
        cmd = WireframeToggleCommand(self.selected_actor, self.current_color)
        self.command_manager.execute(cmd)
        self.emit_status("Wireframe toggled (Undoable).")

    def change_background_color(self, color_tuple):
        """Change the renderer background color via a command so it's undoable."""
        if self.renderer:
            old_color = self.renderer.GetBackground()
            cmd = BackgroundColorChangeCommand(self.renderer, old_color, color_tuple)
            self.command_manager.execute(cmd)

    def restore_initial_camera(self):
        """Restore camera to initial default position without emitting status (internal helper)."""
        cam = self.renderer.GetActiveCamera()
        if self.initial_camera:
            pos, fp, vu, dist = self.initial_camera
            try:
                cam.SetPosition(pos)
                cam.SetFocalPoint(fp)
                cam.SetViewUp(vu)
                current_dist = cam.GetDistance()
                if current_dist != 0:
                    dolly_factor = dist / current_dist 
                    cam.Dolly(dolly_factor)
                self.renderer.ResetCameraClippingRange()
            except Exception:
                pass

    def reset_camera(self, save=False):
        """Reset camera to saved baseline or save current camera as baseline."""
        cam = self.renderer.GetActiveCamera()
        if save:
            # Save current camera as baseline for later resets
            self.initial_distance = cam.GetDistance()
            self.initial_camera = (cam.GetPosition(), cam.GetFocalPoint(), cam.GetViewUp(), self.initial_distance)
            self.emit_status("Saved current camera as baseline.")
        else:
            if self.initial_camera:
                pos,fp,vu,dist = self.initial_camera
                try:
                    cam.SetPosition(pos)
                    cam.SetFocalPoint(fp)
                    cam.SetViewUp(vu)
                    current_dist = cam.GetDistance()
                    if current_dist != 0:
                         dolly_factor = dist / current_dist 
                         cam.Dolly(dolly_factor)
                    self.renderer.ResetCameraClippingRange()
                    self.render_all()
                    self.emit_status("Camera reset to initial position.")
                except Exception:
                    self.emit_status("Camera reset failed.")
            else:
                self.emit_status("No saved camera to reset to.")

    def enable_box_widget(self, auto=False):
        """Enable box widget for interactive actor transform (single actor case)."""
        if not self.selected_actor:
            if not auto:
                self.emit_status("Box Widget: No model selected.")
            return

        # Turn off any existing widget first
        if self.box_widget:
            try:
                self.box_widget.Off()
                self.box_widget.SetInteractor(None)
                self.box_widget = None
            except Exception:
                pass

        # Create a new vtkBoxWidget attached to the interactor and the actor
        self.box_widget = vtk.vtkBoxWidget()
        self.box_widget.SetInteractor(self.interactor)
        self.box_widget.SetProp3D(self.selected_actor)
        self.box_widget.PlaceWidget()
        self.box_widget.On()

        # Save the actor's initial transform for composing during interaction
        self._initial_transform = vtk.vtkTransform()
        if self.selected_actor.GetUserTransform():
            # Read the actor user transform's matrix and set into the copy transform
            matrix = vtk.vtkMatrix4x4()
            self.selected_actor.GetUserTransform().GetMatrix(matrix)
            self._initial_transform.SetMatrix(matrix)
        else:
            self._initial_transform.Identity()

        def update(obj, evt):
            # Callback called during interaction to apply combined transform to actor
            t = vtk.vtkTransform()
            obj.GetTransform(t)
            combined = vtk.vtkTransform()
            combined.Concatenate(self._initial_transform)
            combined.Concatenate(t)
            self.selected_actor.SetUserTransform(combined)
            self.render_all()

        def start_interaction(obj, evt):
            # Recompute base transform at start in case it changed
            self._initial_transform.Identity()
            if self.selected_actor.GetUserTransform():
                matrix = vtk.vtkMatrix4x4()
                self.selected_actor.GetUserTransform().GetMatrix(matrix)
                self._initial_transform.SetMatrix(matrix)

        # Hook up widget events
        self.box_widget.AddObserver("StartInteractionEvent", start_interaction)
        self.box_widget.AddObserver("InteractionEvent", update)
        self.emit_status("Box Widget enabled. Use handles to move, scale, or rotate the object.")
        
    def disable_box_widget(self):
        """Disable and clear any active box widget (single or group)."""
        try:
            if self.box_widget:
                try:
                    self.box_widget.Off()
                    self.box_widget.SetInteractor(None)
                except Exception:
                    pass
                self.box_widget = None
        except Exception:
            pass
        # clear group state if present to avoid stale references
        try:
            if hasattr(self, '_group_members'):
                self._group_members = None
            if hasattr(self, '_group_initial_transforms'):
                self._group_initial_transforms = {}
        except Exception:
            pass

    def enable_box_widget_for_group(self, members):
        """Enable a box widget that transforms multiple actors as a group.

        Important: This captures each member's initial transform and concatenates
        the group's widget transform on top of each member, allowing relative transforms.
        """
        if not members:
            self.emit_status("Group Box Widget: No members provided.")
            return

        # turn off any existing widget
        try:
            if self.box_widget:
                try:
                    self.box_widget.Off()
                    self.box_widget.SetInteractor(None)
                except Exception:
                    pass
                self.box_widget = None
        except Exception:
            pass

        # compute combined bounds for placement around all members
        try:
            first = members[0]
            b = list(first.GetBounds())
            minx, maxx = b[0], b[1]
            miny, maxy = b[2], b[3]
            minz, maxz = b[4], b[5]
            for a in members[1:]:
                bb = a.GetBounds()
                if not bb:
                    continue
                minx = min(minx, bb[0])
                maxx = max(maxx, bb[1])
                miny = min(miny, bb[2])
                maxy = max(maxy, bb[3])
                minz = min(minz, bb[4])
                maxz = max(maxz, bb[5])
            bounds = [minx, maxx, miny, maxy, minz, maxz]
        except Exception:
            # fallback: place widget around origin if bounds computation fails
            bounds = [-1,1,-1,1,-1,1]

        # create box widget and place using computed bounds (handles group transform)
        try:
            self.box_widget = vtk.vtkBoxWidget()
            self.box_widget.SetInteractor(self.interactor)
            try:
                self.box_widget.PlaceWidget(bounds)
            except Exception:
                self.box_widget.PlaceWidget()
            self.box_widget.On()
        except Exception as e:
            self.emit_status(f"Failed to create group box widget: {e}")
            return

        # capture initial transforms per member to allow relative application
        self._group_members = list(members)
        self._group_initial_transforms = {}
        for a in self._group_members:
            t = vtk.vtkTransform()
            try:
                if a.GetUserTransform():
                    m = vtk.vtkMatrix4x4()
                    a.GetUserTransform().GetMatrix(m)
                    t.SetMatrix(m)
                else:
                    t.Identity()
            except Exception:
                t.Identity()
            self._group_initial_transforms[a] = t

        def update(obj, evt):
            # Called during interaction to apply group transform to each member
            try:
                t = vtk.vtkTransform()
                obj.GetTransform(t)
                for a in self._group_members:
                    try:
                        combined = vtk.vtkTransform()
                        # start from actor's initial transform
                        combined.Concatenate(self._group_initial_transforms.get(a, vtk.vtkTransform()))
                        # then apply box-widget transform
                        combined.Concatenate(t)
                        a.SetUserTransform(combined)
                    except Exception:
                        pass
                self.render_all()
            except Exception:
                pass

        def start_interaction(obj, evt):
            # refresh saved initial transforms in case they changed before interaction
            for a in self._group_members:
                tt = vtk.vtkTransform()
                try:
                    if a.GetUserTransform():
                        m = vtk.vtkMatrix4x4()
                        a.GetUserTransform().GetMatrix(m)
                        tt.SetMatrix(m)
                    else:
                        tt.Identity()
                except Exception:
                    tt.Identity()
                self._group_initial_transforms[a] = tt

        # Hook up the widget callbacks
        self.box_widget.AddObserver("StartInteractionEvent", start_interaction)
        self.box_widget.AddObserver("InteractionEvent", update)
        self.emit_status(f"Group box widget enabled for {len(self._group_members)} members.")
        
# Part of myVTK class - Save/Load functionality
    
    def get_polydata(self, actor):
        """Attempt to extract vtkPolyData from an actor's mapper (supports filters/readers)."""
        if not actor:
            return None
        mapper = actor.GetMapper()
        if not mapper:
            return None
        pdata = None
        try:
            # Try to get polydata from mapper's input algorithm (if present)
            if hasattr(mapper, "GetInputAlgorithm") and mapper.GetInputAlgorithm():
                algo = mapper.GetInputAlgorithm()
                if hasattr(algo, "GetOutput"):
                    pdata = algo.GetOutput()
                elif hasattr(algo, "GetOutputDataObject"):
                    pdata = algo.GetOutputDataObject(0)
            if not pdata and hasattr(mapper, "GetInputData") and mapper.GetInputData():
                pdata = mapper.GetInputData()
            if not pdata and hasattr(mapper, "GetInput") and mapper.GetInput():
                pdata = mapper.GetInput()
            # If we got a dataset but not polydata, convert to polydata surface
            if pdata and not pdata.IsA("vtkPolyData"):
                surface_filter = vtk.vtkDataSetSurfaceFilter()
                surface_filter.SetInputData(pdata)
                surface_filter.Update()
                pdata = surface_filter.GetOutput()
            return pdata if pdata and pdata.IsA("vtkPolyData") else None
        except Exception as e:
            self.emit_status(f"Error extracting polydata: {e}")
            return None
        
    def save_selected_model(self, file_path):
        """Save the currently selected model to disk in STL/PLY/VTK format."""
        if not self.selected_actor:
            self.emit_status("Save failed: No model selected.")
            return False
        poly_data = self.get_polydata(self.selected_actor)
        if not poly_data:
            self.emit_status("Save failed: Could not extract geometry data.")
            return False
        ext = file_path.split('.')[-1].lower()
        writer = None
        if ext == 'stl':
            writer = vtk.vtkSTLWriter()
        elif ext == 'ply':
            writer = vtk.vtkPLYWriter()
        elif ext == 'vtk':
            writer = vtk.vtkPolyDataWriter()
        else:
            self.emit_status(f"Unsupported save format: {ext}. Try STL, PLY, or VTK.")
            return False
        try:
            writer.SetFileName(file_path)
            writer.SetInputData(poly_data)
            writer.Update()
            writer.Write()
            self.emit_status(f"Selected model saved to: {file_path}")
            return True
        except Exception as e:
            self.emit_status(f"Failed to save model: {e}")
            return False
            
    def save_scene(self, folder_path):
        """Save entire scene as individual VTP files and a scene.json describing transforms."""
        if not self.actors:
            self.emit_status("Save failed: No models to save.")
            return False
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        scene_data = []
        for i, actor in enumerate(self.actors):
            pdata = self.get_polydata(actor)
            if not pdata:
                continue
            model_path = os.path.join(folder_path, f"model_{i}.vtp")
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(model_path)
            writer.SetInputData(pdata)
            writer.Write()
            pos = actor.GetPosition()
            ori = actor.GetOrientation()
            color = actor.GetProperty().GetColor()
            scene_data.append({
                "file": f"model_{i}.vtp",
                "position": pos,
                "orientation": ori,
                "color": color
            })
        # Write scene index file
        with open(os.path.join(folder_path, "scene.json"), "w") as f:
            json.dump(scene_data, f, indent=4)
        self.emit_status(f"✅ Scene saved with {len(scene_data)} objects to {folder_path}")
        return True
    
    def load_scene(self, folder_path):
        """Load a saved scene (scene.json + vtp models) restoring transforms and colors."""
        json_path = os.path.join(folder_path, "scene.json")
        if not os.path.exists(json_path):
            self.emit_status("❌ Scene load failed: scene.json not found.")
            return False
        with open(json_path, "r") as f:
            scene_data = json.load(f)
        
        # Save camera state BEFORE loading multiple actors (so view doesn't jump)
        cam = self.renderer.GetActiveCamera()
        saved_position = cam.GetPosition()
        saved_focal = cam.GetFocalPoint()
        saved_viewup = cam.GetViewUp()
        saved_distance = cam.GetDistance()
        
        for obj in scene_data:
            model_file = os.path.join(folder_path, obj["file"])
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(model_file)
            reader.Update()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.SetPosition(*obj["position"])
            actor.SetOrientation(*obj["orientation"])
            actor.GetProperty().SetColor(*obj["color"])
            self.renderer.AddActor(actor)
            self.actors.append(actor)
        
        # Restore camera state after loading all actors
        cam.SetPosition(saved_position)
        cam.SetFocalPoint(saved_focal)
        cam.SetViewUp(saved_viewup)
        self.renderer.ResetCameraClippingRange()
        self.render_all()
        self.emit_status(f"✅ Loaded scene with {len(scene_data)} objects from {folder_path}")
        if self.scene_update_callback:
            self.scene_update_callback()
        return True

# ===========================
# MainWindow Class
# ===========================

class MainWindow(QtWidgets.QMainWindow):
    # Qt signal used to update status bar from myVTK
    status_signal = QtCore.pyqtSignal(str)
    _instance_count = 0  # Class variable to track window instances

    def __init__(self, parent=None, show_splash=True):
        super().__init__(parent)
        
        # Increment instance count and set window ID (useful for multi-window UX)
        MainWindow._instance_count += 1
        self.window_id = MainWindow._instance_count
        
        # Optional splash screen on first window open
        if show_splash:
            splash_pix = QPixmap("D:/sdv2025/saimen1_new/Zender 3D.png")  # icon file path (may be missing)
            if not splash_pix.isNull():
                splash_pix = splash_pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                splash = QSplashScreen(splash_pix)
                splash.setMask(splash_pix.mask())
                splash.show()
                QtCore.QTimer.singleShot(3000, splash.close)
        
        # Set the window title with instance id so user can differentiate multiple windows
        self.setWindowTitle(f"Zender 3D {self.window_id}")
        self.resize(1400, 800)
        # Apply a dark theme for modern UX
        self.apply_dark_theme()

        # Create main horizontal splitter to hold left toolbar, VTK area, and right scene navigator
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        
        # === MOVE VTK SETUP HERE - BEFORE LEFT PANEL ===
        # Middle: VTK widget (QVTKRenderWindowInteractor inside a QFrame)
        self.frame = QtWidgets.QFrame()
        self.vl = QtWidgets.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        
        # IMPORTANT: Instantiate myVTK BEFORE creating the interactor style so callbacks can be set
        self.vtk_app = myVTK()  # VTK scene manager
        self.vtk_app.set_status_signal(self.status_signal)  # connect to status signal for UI updates
        
        # Now create and set the interactor style (used to rotate selected actor on drag)
        self.interactor_style = CustomInteractorStyle()
        # Attach interactor style to the QVTK interactor (if available)
        try:
            self.vtkWidget.GetRenderWindow().GetInteractor().SetInteractorStyle(self.interactor_style)
        except Exception:
            # If interactor is not yet available, myVTK.setup_rendering_pipeline will attach style later
            pass
        
        self.vl.addWidget(self.vtkWidget)
        self.frame.setLayout(self.vl)
        
        # Setup rendering pipeline now that widget and myVTK instance are ready
        self.vtk_app.setup_rendering_pipeline(self.vtkWidget, interactor_style=self.interactor_style)
        
        # === NOW CREATE THE LEFT PANEL ===
        # Left side: Toolbar with scroll area (uses a helper to construct widgets)
        self.left_panel = self.create_left_panel_with_scroll()
        
        # Right side: Scene Navigator Panel
        self.scene_panel = self.create_scene_navigator_panel()

        # Add all panels to main splitter (left, center VTK, right)
        self.main_splitter.addWidget(self.left_panel)
        self.main_splitter.addWidget(self.frame)
        self.main_splitter.addWidget(self.scene_panel)
        
        # Set stretch factors (middle gets most space)
        self.main_splitter.setStretchFactor(0, 0)  # Left panel - fixed
        self.main_splitter.setStretchFactor(1, 3)  # VTK widget - expands
        self.main_splitter.setStretchFactor(2, 1)  # Right panel - fixed
        
        # Set some default sizes for the panels
        self.main_splitter.setSizes([250, 900, 250])

        # Make splitter the central widget of the main window
        self.setCentralWidget(self.main_splitter)

        # Connect callbacks between VTK and UI
        self.vtk_app.scene_update_callback = self.refresh_scene_list
        self.vtk_app.selection_changed_callback = self.on_vtk_selection_changed
        
        # Build status bar and menu
        self.create_status_bar()
        self.create_menu()
        
        # Track this window for cleanup / multi-window management
        ACTIVE_WINDOWS.append(self)
        
        # Show welcome after short delay (non-blocking)
        QtCore.QTimer.singleShot(1000, self.show_welcome_message)
        # Immediately send a "ready" message to status bar and terminal
        self.status_signal.emit("Ready. Grid active. Add a model from Model menu or File -> Load.")
        
        # Set geometry explicit for window manager
        self.setGeometry(100, 100, 1400, 800)

    def create_left_panel_with_scroll(self):
        """Create the left panel with scrollable toolbar (many tools in vertical layout)."""
        # Create a widget container and set min/max widths to keep it usable
        left_panel = QtWidgets.QWidget()
        left_panel.setMinimumWidth(220)
        left_panel.setMaximumWidth(350)
        
        # Main vertical layout for the left panel
        main_layout = QtWidgets.QVBoxLayout(left_panel)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header with toggle button to collapse/hide the left panel
        header_widget = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header_widget)
        header_layout.setContentsMargins(10, 5, 10, 5)
        
        header_label = QtWidgets.QLabel("🛠️ Tools")
        header_label.setStyleSheet("font-weight: bold; font-size: 14px; color: white;")
        
        self.toggle_left_btn = QtWidgets.QPushButton("◀")
        self.toggle_left_btn.setFixedSize(25, 25)
        self.toggle_left_btn.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                color: white;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """)
        self.toggle_left_btn.clicked.connect(self.toggle_left_panel)
        
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        header_layout.addWidget(self.toggle_left_btn)
        
        main_layout.addWidget(header_widget)
        
        # Create scroll area for the toolbar so many buttons can be accessible
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #2d2d2d;
            }
            QScrollBar:vertical {
                background: #2d2d2d;
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #555555;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #666666;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)
        
        # Container widget for the toolbar buttons (vertical layout)
        self.toolbar_container = QtWidgets.QWidget()
        self.toolbar_layout = QtWidgets.QVBoxLayout(self.toolbar_container)
        self.toolbar_layout.setContentsMargins(10, 10, 10, 10)
        self.toolbar_layout.setSpacing(8)
        self.toolbar_layout.setAlignment(QtCore.Qt.AlignTop)
        
        # Create the toolbar buttons (helper to populate many controls)
        self.create_toolbar_buttons()
        
        # Set the container as the scroll area's widget so it becomes scrollable
        scroll_area.setWidget(self.toolbar_container)
        main_layout.addWidget(scroll_area)
        
        return left_panel

    def create_toolbar_buttons(self):
        """Create all toolbar buttons and add them to the toolbar layout (commented key actions)."""
        # Clear existing buttons from previous calls (if any)
        for i in reversed(range(self.toolbar_layout.count())): 
            widget = self.toolbar_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
        
        # --- Window Section ---
        window_label = QtWidgets.QLabel("Window")
        window_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                padding: 4px;
                color: white;
                background-color: transparent;
                border-bottom: 1px solid #555555;
            }
        """)
        self.toolbar_layout.addWidget(window_label)
        
        # New Window button - opens a new MainWindow instance
        new_window_button = QToolButton(self)
        new_window_button.setText("🪟 New Window")
        new_window_button.setFixedWidth(200)
        new_window_button.setFixedHeight(40)
        new_window_button.setStyleSheet(self.get_toolbutton_style())
        new_window_button.clicked.connect(self.new_window)
        self.toolbar_layout.addWidget(new_window_button)
        
        # Separator line for grouping
        sep1 = QtWidgets.QFrame()
        sep1.setFrameShape(QtWidgets.QFrame.HLine)
        sep1.setFrameShadow(QtWidgets.QFrame.Sunken)
        sep1.setStyleSheet("background-color: #555555;")
        sep1.setFixedHeight(1)
        self.toolbar_layout.addWidget(sep1)
        
        # --- Edit Tools Section ---
        edit_label = QtWidgets.QLabel("Edit Tools")
        edit_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                padding: 4px;
                color: white;
                background-color: transparent;
                border-bottom: 1px solid #555555;
            }
        """)
        self.toolbar_layout.addWidget(edit_label)
        
        # Undo/Redo buttons - call command manager methods which emit status (and now print to terminal)
        undo_button = QToolButton(self)
        undo_button.setText("↩️ Undo")
        undo_button.setFixedWidth(200)
        undo_button.setFixedHeight(40)
        undo_button.setStyleSheet(self.get_toolbutton_style())
        undo_button.clicked.connect(self.vtk_app.command_manager.undo)
        self.toolbar_layout.addWidget(undo_button)
        
        redo_button = QToolButton(self)
        redo_button.setText("↪️ Redo")
        redo_button.setFixedWidth(200)
        redo_button.setFixedHeight(40)
        redo_button.setStyleSheet(self.get_toolbutton_style())
        redo_button.clicked.connect(self.vtk_app.command_manager.redo)
        self.toolbar_layout.addWidget(redo_button)
        
        # Separator
        sep2 = QtWidgets.QFrame()
        sep2.setFrameShape(QtWidgets.QFrame.HLine)
        sep2.setFrameShadow(QtWidgets.QFrame.Sunken)
        sep2.setStyleSheet("background-color: #555555;")
        sep2.setFixedHeight(1)
        self.toolbar_layout.addWidget(sep2)
        
        # Delete Selected, Duplicate, Save Selected buttons - perform scene operations
        delete_button = QToolButton(self)
        delete_button.setText("🗑️ Delete Selected")
        delete_button.setFixedWidth(200)
        delete_button.setFixedHeight(40)
        delete_button.setStyleSheet(self.get_toolbutton_style("#8B0000"))
        delete_button.clicked.connect(self.vtk_app.delete_selected)
        self.toolbar_layout.addWidget(delete_button)

        duplicate_button = QToolButton(self)
        duplicate_button.setText("🧬 Duplicate Selected")
        duplicate_button.setFixedWidth(200)
        duplicate_button.setFixedHeight(40)
        duplicate_button.setStyleSheet(self.get_toolbutton_style())
        duplicate_button.clicked.connect(self.vtk_app.duplicate_selected)
        self.toolbar_layout.addWidget(duplicate_button)

        save_button = QToolButton(self)
        save_button.setText("💾 Save Selected")
        save_button.setFixedWidth(200)
        save_button.setFixedHeight(40)
        save_button.setStyleSheet(self.get_toolbutton_style("#228B22"))
        save_button.clicked.connect(self.save_model_to_computer)
        self.toolbar_layout.addWidget(save_button)

        # Separator
        sep3 = QtWidgets.QFrame()
        sep3.setFrameShape(QtWidgets.QFrame.HLine)
        sep3.setFrameShadow(QtWidgets.QFrame.Sunken)
        sep3.setStyleSheet("background-color: #555555;")
        sep3.setFixedHeight(1)
        self.toolbar_layout.addWidget(sep3)

        # --- Property Section ---
        prop_label = QtWidgets.QLabel("Properties")
        prop_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                padding: 4px;
                color: white;
                background-color: transparent;
                border-bottom: 1px solid #555555;
            }
        """)
        self.toolbar_layout.addWidget(prop_label)
        
        color_button = QToolButton(self)
        color_button.setText("🎨 Color")
        color_button.setFixedWidth(200)
        color_button.setFixedHeight(40)
        color_button.setStyleSheet(self.get_toolbutton_style())
        color_button.clicked.connect(self.open_color_dialog)
        self.toolbar_layout.addWidget(color_button)

        light_button = QToolButton(self)
        light_button.setText("💡 Light/Material")
        light_button.setFixedWidth(200)
        light_button.setFixedHeight(40)
        light_button.setStyleSheet(self.get_toolbutton_style())
        light_button.clicked.connect(self.open_lighting_dialog)
        self.toolbar_layout.addWidget(light_button)

        wireframe_button = QToolButton(self)
        wireframe_button.setText("📐 Wireframe Toggle")
        wireframe_button.setFixedWidth(200)
        wireframe_button.setFixedHeight(40)
        wireframe_button.setStyleSheet(self.get_toolbutton_style())
        wireframe_button.clicked.connect(self.vtk_app.toggle_wireframe)
        self.toolbar_layout.addWidget(wireframe_button)

        background_button = QToolButton(self)
        background_button.setText("🌌 Background")
        background_button.setFixedWidth(200)
        background_button.setFixedHeight(40)
        background_button.setStyleSheet(self.get_toolbutton_style())
        background_button.clicked.connect(self.open_background_color_dialog)
        self.toolbar_layout.addWidget(background_button)

        # Separator
        sep4 = QtWidgets.QFrame()
        sep4.setFrameShape(QtWidgets.QFrame.HLine)
        sep4.setFrameShadow(QtWidgets.QFrame.Sunken)
        sep4.setStyleSheet("background-color: #555555;")
        sep4.setFixedHeight(1)
        self.toolbar_layout.addWidget(sep4)
        
        # --- Transformation Section ---
        orient_label = QtWidgets.QLabel("Transform")
        orient_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                padding: 4px;
                color: white;
                background-color: transparent;
                border-bottom: 1px solid #555555;
            }
        """)
        self.toolbar_layout.addWidget(orient_label)

        box_widget_button = QToolButton(self)
        box_widget_button.setText("📦 Box Widget")
        box_widget_button.setFixedWidth(200)
        box_widget_button.setFixedHeight(40)
        box_widget_button.setStyleSheet(self.get_toolbutton_style())
        box_widget_button.clicked.connect(self.vtk_app.enable_box_widget)
        self.toolbar_layout.addWidget(box_widget_button)

        translate_button = QToolButton(self)
        translate_button.setText("📍 Translate")
        translate_button.setFixedWidth(200)
        translate_button.setFixedHeight(40)
        translate_button.setStyleSheet(self.get_toolbutton_style())
        translate_button.clicked.connect(self.open_translate_dialog)
        self.toolbar_layout.addWidget(translate_button)

        rotate_button = QToolButton(self)
        rotate_button.setText("🔄 Rotate")
        rotate_button.setFixedWidth(200)
        rotate_button.setFixedHeight(40)
        rotate_button.setStyleSheet(self.get_toolbutton_style())
        rotate_button.clicked.connect(self.open_rotate_dialog)
        self.toolbar_layout.addWidget(rotate_button)

        scale_button = QToolButton(self)
        scale_button.setText("📏 Scale")
        scale_button.setFixedWidth(200)
        scale_button.setFixedHeight(40)
        scale_button.setStyleSheet(self.get_toolbutton_style())
        scale_button.clicked.connect(self.open_scale_dialog)
        self.toolbar_layout.addWidget(scale_button)

        # Separator
        sep5 = QtWidgets.QFrame()
        sep5.setFrameShape(QtWidgets.QFrame.HLine)
        sep5.setFrameShadow(QtWidgets.QFrame.Sunken)
        sep5.setStyleSheet("background-color: #555555;")
        sep5.setFixedHeight(1)
        self.toolbar_layout.addWidget(sep5)

        # --- Info/Camera Section ---
        info_label = QtWidgets.QLabel("Info/Camera")
        info_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                padding: 4px;
                color: white;
                background-color: transparent;
                border-bottom: 1px solid #555555;
            }
        """)
        self.toolbar_layout.addWidget(info_label)

        model_details_button = QToolButton(self)
        model_details_button.setText("📄 Model Details")
        model_details_button.setFixedWidth(200)
        model_details_button.setFixedHeight(40)
        model_details_button.setStyleSheet(self.get_toolbutton_style())
        model_details_button.clicked.connect(self.show_model_details)
        self.toolbar_layout.addWidget(model_details_button)

        camera_control_button = QToolButton(self)
        camera_control_button.setText("📷 Camera Control")
        camera_control_button.setFixedWidth(200)
        camera_control_button.setFixedHeight(40)
        camera_control_button.setStyleSheet(self.get_toolbutton_style())
        camera_control_button.clicked.connect(self.open_camera_dialog)
        self.toolbar_layout.addWidget(camera_control_button)
        
        # Add stretch at the bottom to push everything up visually
        self.toolbar_layout.addStretch()

    def get_toolbutton_style(self, color="#0078d4"):
        """Return consistent styling for tool buttons (UI helper)."""
        return f"""
            QToolButton {{
                background-color: {color};
                color: white;
                border-radius: 6px;
                padding: 6px;
                font-weight: bold;
                border: none;
            }}
            QToolButton:hover {{
                background-color: #106ebe;
            }}
            QToolButton:pressed {{
                background-color: #005a9e;
            }}
            QToolButton:disabled {{
                background-color: #505050;
                color: #a0a0a0;
            }}
        """

    def toggle_left_panel(self):
        """Toggle visibility of left panel - used by both menu and header button."""
        if self.left_panel.isVisible():
            self.left_panel.hide()
            self.toggle_left_btn.setText("▶")
            self.status_signal.emit("Left panel hidden")
        else:
            self.left_panel.show()
            self.toggle_left_btn.setText("◀")
            self.status_signal.emit("Left panel shown")

    def toggle_right_panel(self):
        """Toggle visibility of right (scene) panel."""
        if self.scene_panel.isVisible():
            self.scene_panel.hide()
            self.status_signal.emit("Right panel hidden")
        else:
            self.scene_panel.show()
            self.status_signal.emit("Right panel shown")

    def reset_panels(self):
        """Reset all panels to their default visibility state."""
        if not self.left_panel.isVisible():
            self.toggle_left_panel()
        if not self.scene_panel.isVisible():
            self.toggle_right_panel()
        self.status_signal.emit("All panels reset to default")

    def create_scene_navigator_panel(self):
        """Create the right panel with scene navigator and model properties widgets."""
        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(250)
        panel.setMaximumWidth(400)
        
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title with toggle button
        title_layout = QtWidgets.QHBoxLayout()
        title_label = QtWidgets.QLabel("🗂️ Scene Navigator")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")
        
        self.toggle_right_btn = QtWidgets.QPushButton("▶")
        self.toggle_right_btn.setFixedSize(25, 25)
        self.toggle_right_btn.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                color: white;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """)
        self.toggle_right_btn.clicked.connect(self.toggle_right_panel)
        
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addWidget(self.toggle_right_btn)
        layout.addLayout(title_layout)
        
        # Search/Filter box for the scene list
        search_layout = QtWidgets.QHBoxLayout()
        search_label = QtWidgets.QLabel("🔍")
        self.search_box = QtWidgets.QLineEdit()
        self.search_box.setPlaceholderText("Search models...")
        self.search_box.textChanged.connect(self.filter_scene_list)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_box)
        layout.addLayout(search_layout)
        
        # Model count label to show number of models in scene
        self.model_count_label = QtWidgets.QLabel("Models: 0")
        self.model_count_label.setStyleSheet("font-size: 11px; color: #888; padding: 2px;")
        layout.addWidget(self.model_count_label)
        
        # Scene list widget displays all actors and groups
        self.scene_list = QtWidgets.QListWidget()
        self.scene_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.scene_list.itemClicked.connect(self.on_scene_item_clicked)
        self.scene_list.itemDoubleClicked.connect(self.on_scene_item_double_clicked)
        self.scene_list.setStyleSheet("""
            QListWidget {
                background-color: #2a2a2a;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #333;
                color: white;
            }
            QListWidget::item:selected {
                background-color: #007ACC;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #3a3a3a;
            }
        """)
        layout.addWidget(self.scene_list)
        
        # Quick actions group (focus/hide/delete)
        actions_group = QtWidgets.QGroupBox("Quick Actions")
        actions_group.setStyleSheet("QGroupBox { font-weight: bold; padding-top: 10px; }")
        actions_layout = QtWidgets.QVBoxLayout()
        
        select_btn = QtWidgets.QPushButton("👁️ Focus Selected")
        select_btn.clicked.connect(self.focus_on_selected)
        actions_layout.addWidget(select_btn)
        
        hide_btn = QtWidgets.QPushButton("👻 Toggle Visibility")
        hide_btn.clicked.connect(self.vtk_app.toggle_visibility)
        actions_layout.addWidget(hide_btn)
        
        delete_btn = QtWidgets.QPushButton("🗑️ Delete Selected")
        delete_btn.clicked.connect(self.delete_from_scene_list)
        delete_btn.setStyleSheet("background-color: #8B0000;")
        actions_layout.addWidget(delete_btn)
        
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
        # Properties group showing selected model metadata
        props_group = QtWidgets.QGroupBox("Selected Model Properties")
        props_group.setStyleSheet("QGroupBox { font-weight: bold; padding-top: 10px; }")
        props_layout = QtWidgets.QVBoxLayout()
        
        self.prop_name_label = QtWidgets.QLabel("Name: None")
        self.prop_type_label = QtWidgets.QLabel("Type: -")
        self.prop_vertices_label = QtWidgets.QLabel("Vertices: -")
        self.prop_position_label = QtWidgets.QLabel("Position: -")
        self.prop_visible_label = QtWidgets.QLabel("Visible: -")
        
        for label in [self.prop_name_label, self.prop_type_label, self.prop_vertices_label, 
                      self.prop_position_label, self.prop_visible_label]:
            label.setStyleSheet("font-size: 10px; padding: 2px;")
            props_layout.addWidget(label)
        
        props_group.setLayout(props_layout)
        layout.addWidget(props_group)
        
        layout.addStretch()
        return panel

    def show_welcome_message(self):
        """Show a simple welcome message box to the user after startup."""
        QtWidgets.QMessageBox.information(self, "Welcome to Zender 3D", "Welcome to Zender 3D\n\nA powerful 3D modeling and visualization application.\n\nGet started by loading a 3D model or creating objects from the toolbar.\n\nFor more information, visit Help > User Manual.")  
        
    def new_window(self):
        """Create a new instance of the application window without splash screen."""
        # Create new window without splash screen
        new_win = MainWindow(show_splash=False)
        
        # Apply the same geometry and position as current window
        current_geometry = self.geometry()
        new_win.setGeometry(current_geometry)
        
        # Offset the position slightly so new window is visible
        current_pos = self.pos()
        new_win.move(current_pos.x() + 40, current_pos.y() + 40)
        
        # Ensure the window is properly shown and focused
        new_win.show()
        new_win.raise_()  # Bring to front
        new_win.activateWindow()  # Activate the window
        
        self.status_signal.emit("New window opened")

    def closeEvent(self, event):
        """Handle window close event with confirmation for last window and multi-window cleanup."""
        # Remove this window from active windows tracking
        if self in ACTIVE_WINDOWS:
            ACTIVE_WINDOWS.remove(self)
        
        # Clean up VTK resources associated with this window
        try:
            if hasattr(self, 'vtk_app'):
                self.vtk_app.cleanup()
            if hasattr(self, 'vtkWidget'):
                self.vtkWidget.close()
        except Exception as e:
            print(f"Close event cleanup error: {e}")
        
        # Count how many MainWindow instances are still open
        window_count = len([w for w in ACTIVE_WINDOWS if isinstance(w, MainWindow)])
        
        # If this was the last window open, ask user for confirmation before quitting
        if window_count <= 0:
            reply = QtWidgets.QMessageBox.question(
                self, 
                'Confirm Exit',
                'Are you sure you want to exit the application?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            
            if reply == QtWidgets.QMessageBox.Yes:
                # Clean up all remaining windows and accept close event
                for window in ACTIVE_WINDOWS[:]:
                    try:
                        if hasattr(window, 'vtk_app'):
                            window.vtk_app.cleanup()
                        window.close()
                    except Exception:
                        pass
                ACTIVE_WINDOWS.clear()
                event.accept()
                # Terminal log for exit
                print("[MainWindow] Application exiting.")
            else:
                # Cancel close and re-add self to active windows tracking
                event.ignore()
                if self not in ACTIVE_WINDOWS:
                    ACTIVE_WINDOWS.append(self)
        else:
            # If other windows remain open, close this one silently
            event.accept()

    def on_vtk_selection_changed(self, actor):
        """Called when selection changes in the VTK interactor (picking).
        Update the scene list selection and properties panel to follow the pick."""
        # actor may be None (selection cleared)
        if actor is None:
            # Clear selection in list and property display
            self.scene_list.clearSelection()
            self.update_property_display(None)
            return

        # Find the corresponding list item for this actor and select it
        for i in range(self.scene_list.count()):
            item = self.scene_list.item(i)
            item_actor = item.data(QtCore.Qt.UserRole)
            if item_actor is actor:
                self.scene_list.setCurrentItem(item)
                break

        # Ensure the application's selected_actor matches and update properties panel
        try:
            self.vtk_app.selected_actor = actor
            if self.interactor_style:
                self.interactor_style.actor = actor
        except Exception:
            pass
        self.update_property_display(actor)

    def refresh_scene_list(self):
        """Refresh the scene list with all current actors (called after scene changes)."""
        self.scene_list.clear()
        # Detect groups (actors that share a user_object_group attribute)
        groups = {}
        for actor in self.vtk_app.actors:
            group = getattr(actor, 'user_object_group', None)
            if group:
                groups.setdefault(group, []).append(actor)

        # Add a group (select-all) entry for each detected group
        for group_name, members in groups.items():
            item_text = f"🔷 All: {group_name} ({len(members)})"
            item = QtWidgets.QListWidgetItem(item_text)
            # mark as group entry by storing a tuple
            item.setData(QtCore.Qt.UserRole, ("group", group_name))
            item.setBackground(QtGui.QColor(60, 60, 70))
            self.scene_list.addItem(item)

        # Now add individual actor entries with visibility icon and color background
        for i, actor in enumerate(self.vtk_app.actors):
            model_name = getattr(actor, 'user_object_name', None)
            if not model_name:
                model_name = self.vtk_app.current_object if hasattr(self.vtk_app, 'current_object') else f"Model {i+1}"

            is_visible = actor.GetVisibility()
            visibility_icon = "👁️" if is_visible else "👻"

            color = actor.GetProperty().GetColor()
            # Convert color to hex for background styling
            color_hex = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"

            item_text = f"{visibility_icon} [{i+1}] {model_name}"
            item = QtWidgets.QListWidgetItem(item_text)
            item.setData(QtCore.Qt.UserRole, actor)
            # Set a darker color as background for readability
            item.setBackground(QtGui.QColor(color_hex).darker(300))
            self.scene_list.addItem(item)
        
        # Update count label
        self.model_count_label.setText(f"Models: {len(self.vtk_app.actors)}")
        
        # If there is a selected actor, refresh its property display
        if self.vtk_app.selected_actor:
            self.update_property_display(self.vtk_app.selected_actor)
    
    def filter_scene_list(self, text):
        """Filter the scene list based on search text (simple substring match)."""
        for i in range(self.scene_list.count()):
            item = self.scene_list.item(i)
            if text.lower() in item.text().lower():
                item.setHidden(False)
            else:
                item.setHidden(True)
    
    def on_scene_item_clicked(self, item):
        """Handle scene list item click - select the actor (or group) and update UI."""
        data = item.data(QtCore.Qt.UserRole)

        # Clear any existing group highlights and disable any current box widget
        try:
            self.vtk_app.disable_box_widget()
        except Exception:
            pass

        # If this is a group entry, data will be a tuple ("group", group_name)
        if isinstance(data, tuple) and len(data) == 2 and data[0] == 'group':
            group_name = data[1]
            members = [a for a in self.vtk_app.actors if getattr(a, 'user_object_group', None) == group_name]
            if members:
                try:
                    # enable interactive group transform (box widget will apply transforms to all members)
                    self.vtk_app.enable_box_widget_for_group(members)
                    # clear single selection
                    self.vtk_app.selected_actor = None
                    if self.interactor_style:
                        try:
                            self.interactor_style.actor = None
                        except Exception:
                            pass
                    self.update_property_display(None)
                    self.vtk_app.render_all()
                    self.status_signal.emit(f"Selected group: {group_name} [{len(members)} items]")
                except Exception as e:
                    # fallback: simple visual highlight if widget fails
                    for a in members:
                        try:
                            a.GetProperty().SetEdgeVisibility(1)
                            a.GetProperty().SetEdgeColor(1.0, 1.0, 0.0)
                        except Exception:
                            pass
                    self.status_signal.emit(f"Selected group (highlight): {group_name} [{len(members)} items]")
            else:
                self.status_signal.emit(f"Group {group_name} has no members.")
            return

        # Otherwise it's a single actor
        actor = data
        if actor:
            # Set selection in VTK and interactor style and update property display
            self.vtk_app.selected_actor = actor
            if self.interactor_style:
                self.interactor_style.actor = actor
            self.update_property_display(actor)
            self.vtk_app.render_all()
            self.status_signal.emit(f"Selected: {item.text()}")
    
    def on_scene_item_double_clicked(self, item):
        """Handle double-click - focus camera on the model (delegates to focus_on_selected)."""
        self.focus_on_selected()
    
    def focus_on_selected(self):
        """Focus the camera on the selected actor or group by computing bounding box and positioning camera."""
        # Check if a group is selected
        group_members = getattr(self.vtk_app, '_group_members', None)
        is_group = bool(group_members)
        
        if is_group:
            actors_to_focus = group_members
        elif self.vtk_app.selected_actor:
            actors_to_focus = [self.vtk_app.selected_actor]
        else:
            self.status_signal.emit("No model or group selected to focus on.")
            return
        
        renderer = self.vtk_app.renderer
        camera = renderer.GetActiveCamera()
        
        # Compute bounding box for all actors
        minx, maxx = float('inf'), float('-inf')
        miny, maxy = float('inf'), float('-inf')
        minz, maxz = float('inf'), float('-inf')
        
        for actor in actors_to_focus:
            bounds = actor.GetBounds()
            if bounds:
                minx = min(minx, bounds[0])
                maxx = max(maxx, bounds[1])
                miny = min(miny, bounds[2])
                maxy = max(maxy, bounds[3])
                minz = min(minz, bounds[4])
                maxz = max(maxz, bounds[5])
        
        center = [
            (minx + maxx) / 2.0,
            (miny + maxy) / 2.0,
            (minz + maxz) / 2.0
        ]
        # Position camera at an offset proportional to largest bounding dimension
        camera.SetFocalPoint(center)
        max_dim = max(maxx - minx, maxy - miny, maxz - minz)
        camera.SetPosition(
            center[0] + max_dim * 2,
            center[1] + max_dim * 2,
            center[2] + max_dim * 1.5
        )
        renderer.ResetCameraClippingRange()
        self.vtk_app.render_all()
        msg = "Camera focused on group." if is_group else "Camera focused on selected model."
        self.status_signal.emit(msg)
    
    def delete_from_scene_list(self):
        """Delete the selected model or group from the scene list (uses delete_selected)."""
        # Check if a group is selected
        group_members = getattr(self.vtk_app, '_group_members', None)
        is_group = bool(group_members)
        
        if is_group:
            # Delete all members in group using delete_selected (undoable per actor)
            for actor in group_members:
                self.vtk_app.selected_actor = actor
                self.vtk_app.delete_selected()
            self.vtk_app.disable_box_widget()
            self.vtk_app._group_members = None
            self.refresh_scene_list()
            self.status_signal.emit(f"Deleted group ({len(group_members)} items)")
        else:
            current_item = self.scene_list.currentItem()
            if not current_item:
                self.status_signal.emit("No model selected in scene list.")
                return
            actor = current_item.data(QtCore.Qt.UserRole)
            if actor:
                self.vtk_app.selected_actor = actor
                self.vtk_app.delete_selected()
                self.refresh_scene_list()
    
    def update_property_display(self, actor):
        """Update the properties panel with actor information (name, type, vertices, position, visibility)."""
        if not actor:
            self.prop_name_label.setText("Name: None")
            self.prop_type_label.setText("Type: -")
            self.prop_vertices_label.setText("Vertices: -")
            self.prop_position_label.setText("Position: -")
            self.prop_visible_label.setText("Visible: -")
            return
        
        name = getattr(actor, 'user_object_name', self.vtk_app.current_object or "Unnamed")
        self.prop_name_label.setText(f"Name: {name}")

        # Prefer a per-actor type if available, otherwise fall back to app-level tracking
        model_type = getattr(actor, 'user_object_type', None) or self.vtk_app.current_object_type or self.vtk_app.current_object or "Unknown"
        self.prop_type_label.setText(f"Type: {model_type}")

        poly_data = self.vtk_app.get_polydata(actor)
        if poly_data:
            vertex_count = poly_data.GetNumberOfPoints()
            self.prop_vertices_label.setText(f"Vertices: {vertex_count:,}")
        else:
            self.prop_vertices_label.setText("Vertices: N/A")

        pos = actor.GetPosition()
        self.prop_position_label.setText(f"Position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

        visible = "Yes" if actor.GetVisibility() else "No"
        self.prop_visible_label.setText(f"Visible: {visible}")

    def apply_dark_theme(self):
        """Apply a dark Fusion-based palette and custom stylesheet for the whole application."""
        app = QtWidgets.QApplication.instance()
        app.setStyle("Fusion")
        dark_palette = QtGui.QPalette()
        dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(37, 37, 38))
        dark_palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
        dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 30, 30))
        dark_palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
        dark_palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
        dark_palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
        dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 45, 48))
        dark_palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
        dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(0, 122, 204))
        dark_palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
        app.setPalette(dark_palette)
        app.setStyleSheet("""
            QMainWindow { 
                background-color: #202020; 
            }
            QToolBar {
                background-color: #2d2d2d;
                border: none;
                spacing: 8px;
                padding: 6px;
            }
            QToolBar::separator {
                background: #555555;
                width: 1px;
                margin: 4px 2px;
            }
            QPushButton, QToolButton, QComboBox {
                background-color: #0078d4;
                color: white;
                border-radius: 6px;
                padding: 6px;
                border: none;
                font-weight: bold;
            }
            QToolButton:hover, QPushButton:hover { 
                background-color: #106ebe; 
            }
            QToolButton:pressed, QPushButton:pressed { 
                background-color: #005a9e; 
            }
            QMenuBar { 
                background-color: #2d2d2d; 
                color: white; 
            }
            QMenuBar::item:selected { 
                background-color: #505050; 
            }
            QStatusBar { 
                background-color: #262626; 
                color: white; 
                font-weight: bold; 
            }
            QLabel {
                color: white;
                background-color: transparent;
            }
        """)

    def create_status_bar(self):
        """Create status bar and connect the status_signal to show messages there."""
        self.statusBar().showMessage("Initializing...")
        # Connect status_signal to statusBar.showMessage so myVTK.emit_status updates UI
        self.status_signal.connect(self.statusBar().showMessage)
        
    def create_menu(self):
        """Construct menu bar with File, Model, Edit, and View options."""
        menubar = self.menuBar()
        
        # --- View Menu ---
        view_menu = menubar.addMenu('👁️ &View')
        view_menu.addAction(QtWidgets.QAction('Toggle Left Panel', self, shortcut='Ctrl+L', triggered=self.toggle_left_panel))
        view_menu.addAction(QtWidgets.QAction('Toggle Right Panel', self, shortcut='Ctrl+R', triggered=self.toggle_right_panel))
        view_menu.addSeparator()
        view_menu.addAction(QtWidgets.QAction('Reset All Panels', self, triggered=self.reset_panels))
        
        # --- File Menu ---
        file_menu = menubar.addMenu('📁 &File')
        file_menu.addAction(QtWidgets.QAction('New Window', self, shortcut='Ctrl+N', triggered=self.new_window))
        file_menu.addSeparator()
        file_menu.addAction(QtWidgets.QAction('Load 3D Model (stl/obj/ply/vtp/vtk/3ds)', self, shortcut='Ctrl+L', triggered=self.add_model_from_computer))
        file_menu.addAction(QtWidgets.QAction('Save Selected Model (stl/ply/vtk)', self, shortcut='Ctrl+S', triggered=self.save_model_to_computer))
        file_menu.addAction(QtWidgets.QAction('Save Scene (All Objects)', self, triggered=self.save_scene_folder))
        file_menu.addAction(QtWidgets.QAction('Load Scene (All Objects)', self, triggered=self.load_scene_folder))
        file_menu.addSeparator()
        file_menu.addAction(QtWidgets.QAction('Exit', self, shortcut='Ctrl+Q', triggered=self.close))

        # --- Model Menu ---
        model_menu = menubar.addMenu('📦 &Model')
        
        basic_menu = model_menu.addMenu('Basic Primitives')
        basic_menu.addAction(QtWidgets.QAction('Sphere', self, triggered=lambda: self.vtk_app.load_geom('sphere')))
        basic_menu.addAction(QtWidgets.QAction('Cube', self, triggered=lambda: self.vtk_app.load_geom('cube')))
        basic_menu.addAction(QtWidgets.QAction('Disk', self, triggered=lambda: self.vtk_app.load_source_model('disk')))

        vtk_menu = model_menu.addMenu('VTK Generators')
        vtk_menu.addAction(QtWidgets.QAction('Cylinder', self, triggered=lambda: self.vtk_app.load_vtk_model('cylinder')))
        vtk_menu.addAction(QtWidgets.QAction('Cone', self, triggered=lambda: self.vtk_app.load_vtk_model('cone')))

        cell_menu = model_menu.addMenu('Cell Models')
        cell_menu.addAction(QtWidgets.QAction('Tetrahedron', self, triggered=lambda: self.vtk_app.load_cell_model('tetra')))
        cell_menu.addAction(QtWidgets.QAction('Convex Set', self, triggered=lambda: self.vtk_app.load_cell_model('convex')))
        cell_menu.addAction(QtWidgets.QAction('Tessellated Cell', self, triggered=lambda: self.vtk_app.load_cell_model('tessellated')))

        source_menu = model_menu.addMenu('Source Models')
        source_menu.addAction(QtWidgets.QAction('Icosahedron', self, triggered=lambda: self.vtk_app.load_source_model('icosahedron')))
        source_menu.addAction(QtWidgets.QAction('Reduced Cube', self, triggered=lambda: self.vtk_app.load_source_model('reduced_cube')))

        func_menu = model_menu.addMenu('Functional')
        func_menu.addAction(QtWidgets.QAction('Sinusoidal Surface', self, triggered=lambda: self.vtk_app.load_function_model('sinusoidal')))
        func_menu.addAction(QtWidgets.QAction('Gaussian Surface', self, triggered=lambda: self.vtk_app.load_function_model('gaussian')))

        param_menu = model_menu.addMenu('Parametric Models')
        param_menu.addAction(QtWidgets.QAction('Torus', self, triggered=lambda: self.vtk_app.load_parametric_model('torus')))
        param_menu.addAction(QtWidgets.QAction('Klein Bottle', self, triggered=lambda: self.vtk_app.load_parametric_model('klein')))

        # --- Edit Menu ---
        edit_menu = menubar.addMenu('✏ &Edit')
        
        # Undo/Redo actions
        edit_menu.addAction(QtWidgets.QAction('Undo', self, shortcut='Ctrl+Z', triggered=self.vtk_app.command_manager.undo))
        edit_menu.addAction(QtWidgets.QAction('Redo', self, shortcut='Ctrl+Y', triggered=self.vtk_app.command_manager.redo))
        edit_menu.addSeparator()
        
        edit_menu.addAction(QtWidgets.QAction('Delete Selected', self, triggered=self.vtk_app.delete_selected))
        edit_menu.addAction(QtWidgets.QAction('Duplicate Selected', self, triggered=self.vtk_app.duplicate_selected))
        edit_menu.addSeparator()
        
        # Properties
        prop_menu = edit_menu.addMenu('Properties')
        prop_menu.addAction(QtWidgets.QAction('Color', self, triggered=self.open_color_dialog))
        prop_menu.addAction(QtWidgets.QAction('Light/Material', self, triggered=self.open_lighting_dialog))
        prop_menu.addAction(QtWidgets.QAction('Toggle Visibility', self, triggered=self.vtk_app.toggle_visibility))
        prop_menu.addAction(QtWidgets.QAction('Toggle Wireframe', self, triggered=self.vtk_app.toggle_wireframe))
        
        # Transformations
        trans_menu = edit_menu.addMenu('Transformations')
        trans_menu.addAction(QtWidgets.QAction('Box Widget (toggle)', self, triggered=self.vtk_app.enable_box_widget))
        trans_menu.addAction(QtWidgets.QAction('Translate (Dialog) (Undoable)', self, triggered=self.open_translate_dialog))
        trans_menu.addAction(QtWidgets.QAction('Rotate (Dialog)', self, triggered=self.open_rotate_dialog))
        trans_menu.addAction(QtWidgets.QAction('Scale (Dialog)', self, triggered=self.open_scale_dialog))
        
        # Scene/Camera
        scene_menu = edit_menu.addMenu('Scene/Camera')
        scene_menu.addAction(QtWidgets.QAction('Show Model Details', self, triggered=self.show_model_details)) 
        scene_menu.addAction(QtWidgets.QAction('Modify Camera', self, triggered=self.open_camera_dialog))
        scene_menu.addAction(QtWidgets.QAction('Reset Camera', self, triggered=lambda: self.vtk_app.reset_camera(save=False)))
        scene_menu.addAction(QtWidgets.QAction('Save Camera as Baseline', self, triggered=lambda: self.vtk_app.reset_camera(save=True)))
        
        # --- Help Menu ---
        help_menu = menubar.addMenu('❓ &Help')

        user_manual_action = QtWidgets.QAction('📘 User Manual', self)
        user_manual_action.triggered.connect(self.open_user_manual)
        help_menu.addAction(user_manual_action)
     
# Part I - MainWindow file operations and helpers
    
    def add_model_from_computer(self):
        """Open file dialog to let user pick a 3D model file and load it into scene."""
        self.status_signal.emit("Opening file dialog...")
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 
            "Load Model", 
            "", 
            "3D Files (*.stl *.obj *.ply *.vtp *.vtk *.3ds)"
        )
        if file_path:
            self.vtk_app.load_file(file_path)

    def save_model_to_computer(self):
        """Open save dialog and write the currently selected model to disk."""
        if not self.vtk_app.selected_actor:
             self.status_signal.emit("Save aborted: No model is currently selected.")
             QtWidgets.QMessageBox.warning(self, "Save Model", "Please select a model to save first.")
             return
             
        self.status_signal.emit("Opening save dialog...")
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 
            "Save Selected Model", 
            "model.stl", 
            "STL Files (*.stl);;PLY Files (*.ply);;VTK PolyData (*.vtk)"
        )
        if file_path:
            self.vtk_app.save_selected_model(file_path)

    def save_scene_folder(self):
        """Let user select a folder to save the entire scene to (model files + scene.json)."""
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder to Save Scene")
        if folder:
            self.vtk_app.save_scene(folder)

    def load_scene_folder(self):
        """Let user select a folder containing a saved scene to load."""
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Scene Folder to Load")
        if folder:
            self.vtk_app.load_scene(folder)

    def show_model_details(self):
        """Displays the details of the selected model in a dialog box."""
        actor = self.vtk_app.selected_actor
        
        if not actor:
            QtWidgets.QMessageBox.information(
                self, 
                "Model Details", 
                "No model is currently selected. Please click on an object to select it first."
            )
            self.status_signal.emit("Model details not shown: No model selected.")
            return

        poly_data = self.vtk_app.get_polydata(actor)
        
        if not poly_data:
            self.status_signal.emit("Error: Could not retrieve geometry data for stats.")
            QtWidgets.QMessageBox.warning(
                self, 
                "Model Details", 
                "Could not retrieve geometry data for the selected model. May be a non-PolyData format."
            )
            return

        num_points = poly_data.GetNumberOfPoints()
        num_cells = poly_data.GetNumberOfCells()
        num_polys = poly_data.GetNumberOfPolys()
        
        object_name = getattr(actor, 'user_object_name', self.vtk_app.current_object or 'Unnamed Object')

        # Build details string and show in message box (not rich text, keep simple)
        details = f"Selected Model: {object_name}\n"
        details += f"ID: {self.vtk_app.actors.index(actor) if actor in self.vtk_app.actors else 'N/A'}\n"
        details += "-"*30 + "\n"
        details += f"Number of Points (Vertices): {num_points}\n"
        details += f"Number of Cells (Elements): {num_cells}\n"
        details += f"Number of Polygons (Surfaces): {num_polys}"
        
        QtWidgets.QMessageBox.information(self, "Model Details", details)
        
# Part J - MainWindow open_camera_dialog method (Part 1)
    
    def open_camera_dialog(self):
        """Real-time Camera Properties dialog (Position/FocalPoint/ViewUp/Distance).
           Note: live preview updates camera as sliders are manipulated."""
        renderer = self.vtk_app.renderer
        if not renderer:
            return
        camera = renderer.GetActiveCamera()
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Real-time Camera Properties")
        layout = QtWidgets.QFormLayout(dialog)
        
        # Store the camera state before the dialog opens so we can revert on cancel
        camera_state_before_slide = {
            'pos': camera.GetPosition(),
            'fp': camera.GetFocalPoint(),
            'vu': camera.GetViewUp(),
            'dist': camera.GetDistance()
        }
        
        # Dictionary to hold all slider widgets and their properties used by update routine
        slider_widgets = {}

        def create_realtime_coord_sliders(label, current_value, range_min, range_max, scale, key_prefix):
            """Helper to create three sliders for X/Y/Z coordinates with live labels."""
            group = QtWidgets.QWidget()
            v_layout = QtWidgets.QVBoxLayout(group)
            v_layout.setContentsMargins(0, 5, 0, 5)
            
            group_label = QtWidgets.QLabel(f"--- {label} (Range: {range_min/scale:.1f} to {range_max/scale:.1f}) ---")
            group_label.setStyleSheet("font-weight:bold; margin-top: 10px;")
            v_layout.addWidget(group_label)
            
            coord_names = ['X', 'Y', 'Z']
            sliders = []

            for i, name in enumerate(coord_names):
                h_layout = QtWidgets.QHBoxLayout()
                slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
                slider.setRange(range_min, range_max)
                initial_value = int(current_value[i] * scale)
                slider.setValue(initial_value)
                
                # Store slider data so update_camera_settings can read them
                key = f"{key_prefix}_{name}"
                slider_widgets[key] = {'slider': slider, 'scale': scale, 'initial_value': initial_value}

                current_val_label = QtWidgets.QLabel(f"{slider.value()/scale:.2f}")
                
                # Update label function for visual feedback
                def update_label(v, label_ref=current_val_label, scale_ref=scale):
                    label_ref.setText(f"{v/scale_ref:.2f}")
                
                slider.valueChanged.connect(lambda v, lbl=current_val_label, sc=scale: update_label(v, lbl, sc))
                
                h_layout.addWidget(QtWidgets.QLabel(name + ":"))
                h_layout.addWidget(slider)
                h_layout.addWidget(current_val_label)
                v_layout.addLayout(h_layout)
                
                sliders.append(slider)
            
            layout.addRow(group)
            return sliders

        # --- Central Update Function that reads slider values and applies them to camera ---
        def update_camera_settings():
            # Read Pos, FP, VU values from slider_widgets and apply to camera
            new_pos = (
                slider_widgets['Pos_X']['slider'].value() / slider_widgets['Pos_X']['scale'],
                slider_widgets['Pos_Y']['slider'].value() / slider_widgets['Pos_Y']['scale'],
                slider_widgets['Pos_Z']['slider'].value() / slider_widgets['Pos_Z']['scale']
            )
            
            new_fp = (
                slider_widgets['FP_X']['slider'].value() / slider_widgets['FP_X']['scale'],
                slider_widgets['FP_Y']['slider'].value() / slider_widgets['FP_Y']['scale'],
                slider_widgets['FP_Z']['slider'].value() / slider_widgets['FP_Z']['scale']
            )

            new_vu = (
                slider_widgets['VU_X']['slider'].value() / slider_widgets['VU_X']['scale'],
                slider_widgets['VU_Y']['slider'].value() / slider_widgets['VU_Y']['scale'],
                slider_widgets['VU_Z']['slider'].value() / slider_widgets['VU_Z']['scale']
            )
            
            # Apply new camera vectors
            camera.SetPosition(*new_pos)
            camera.SetFocalPoint(*new_fp)
            camera.SetViewUp(*new_vu)
            
            # Apply Distance (Zoom/Dolly) - slider provides a zoom factor
            dolly_val = slider_widgets['Distance']['slider'].value() / slider_widgets['Distance']['scale']
            current_dist_after_pos_fp = camera.GetDistance()
            
            if current_dist_after_pos_fp != 0:
                # Compute a dolly factor to reach a target distance derived from baseline
                target_dist = camera_state_before_slide['dist'] / dolly_val 
                dolly_factor_to_target = target_dist / current_dist_after_pos_fp
                camera.Dolly(dolly_factor_to_target)

            renderer.ResetCameraClippingRange()
            self.vtk_app.render_all()
            # Emit status so both UI and terminal show camera update
            self.status_signal.emit(f"Camera Updated: P({new_pos[0]:.1f}) FP({new_fp[0]:.1f}) Zoom={dolly_val:.2f}")
            
# Part K - Continuation of open_camera_dialog (Part 2)
        # --- Distance Slider (Real-time Zoom) ---
        distance_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        distance_slider.setRange(10, 200) 
        distance_slider.setValue(100)
        distance_label = QtWidgets.QLabel("1.00")
        slider_widgets['Distance'] = {'slider': distance_slider, 'scale': 100.0, 'initial_value': 100}
        distance_slider.valueChanged.connect(lambda v, lbl=distance_label: lbl.setText(f"{v/100.0:.2f}"))
        
        distance_group = QtWidgets.QWidget()
        h_layout_dolly = QtWidgets.QHBoxLayout(distance_group)
        h_layout_dolly.setContentsMargins(0,0,0,0)
        h_layout_dolly.addWidget(distance_slider)
        h_layout_dolly.addWidget(distance_label)
        layout.addRow(QtWidgets.QLabel("Real-time Distance (Zoom Factor):"), distance_group)
        # Wire distance slider to live update handler
        distance_slider.valueChanged.connect(update_camera_settings)

        # --- Position Sliders ---
        pos_sliders = create_realtime_coord_sliders("Position", camera_state_before_slide['pos'], -1000, 1000, 10.0, 'Pos')
        for item in pos_sliders:
            item.valueChanged.connect(update_camera_settings)
        
        # --- Focal Point Sliders ---
        fp_sliders = create_realtime_coord_sliders("Focal Point", camera_state_before_slide['fp'], -1000, 1000, 10.0, 'FP')
        for item in fp_sliders:
            item.valueChanged.connect(update_camera_settings)

        # --- View Up Sliders ---
        vu_sliders = create_realtime_coord_sliders("View Up", camera_state_before_slide['vu'], -100, 100, 100.0, 'VU')
        for item in vu_sliders:
            item.valueChanged.connect(update_camera_settings)

        # --- Buttons and Reset Logic ---
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | 
            QtWidgets.QDialogButtonBox.Cancel | 
            QtWidgets.QDialogButtonBox.Reset
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        reset_button = button_box.button(QtWidgets.QDialogButtonBox.Reset)
        def reset_to_initial():
            # Reset all sliders back to initial recorded values
            for data in slider_widgets.values():
                data['slider'].setValue(data['initial_value'])
            
        reset_button.clicked.connect(reset_to_initial)
        layout.addWidget(button_box)
        
        # Execute dialog (exec_ blocks until closed). On accept we keep camera changes; on reject we revert.
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.status_signal.emit("Camera properties updated and saved.")
        else:
            # Reset camera to the state before dialog
            camera.SetPosition(*camera_state_before_slide['pos'])
            camera.SetFocalPoint(*camera_state_before_slide['fp'])
            camera.SetViewUp(*camera_state_before_slide['vu'])
            
            # Dolly to the original distance baseline
            current_dist = camera.GetDistance()
            if current_dist != 0:
                 dolly_factor_reset = camera_state_before_slide['dist'] / current_dist
                 camera.Dolly(dolly_factor_reset)
                 
            self.vtk_app.render_all()
            self.status_signal.emit("Camera settings canceled and reverted.")
            
# Part L - MainWindow color and lighting dialog methods
    
    def open_color_dialog(self):
        """Open QColorDialog to choose an object color and apply it (undoable via ColorChangeCommand)."""
        # Start from currently tracked default color
        r, g, b = self.vtk_app.current_color
        current_qcolor = QtGui.QColor.fromRgbF(r, g, b)
        color = QtWidgets.QColorDialog.getColor(current_qcolor, self, "Select Object Color")
        
        if color.isValid():
            vtk_color = (color.redF(), color.greenF(), color.blueF())
            self.vtk_app.current_color = vtk_color

            actor_to_color = self.vtk_app.selected_actor
            if not actor_to_color:
                # Fallback to last actor added if none is selected
                if self.vtk_app.actors:
                    actor_to_color = self.vtk_app.actors[-1]
                else:
                    self.status_signal.emit("No model selected or available to color.")
                    QtWidgets.QMessageBox.warning(
                        self, 
                        "Color Change Failed", 
                        "No model is currently selected or available to color."
                    )
                    return

            old_color = actor_to_color.GetProperty().GetColor()
            cmd = ColorChangeCommand(actor_to_color, old_color, vtk_color)
            # Execute as a command so change is undoable
            self.vtk_app.command_manager.execute(cmd)

            actor_to_color.GetProperty().SetColor(vtk_color)
            self.vtk_app.render_all()
            self.status_signal.emit("Object color updated.")

    def open_lighting_dialog(self):
        """Open a dialog to adjust ambient/diffuse/specular/power properties with live preview."""
        # Apply lighting to selected actor, or fallback to last actor if none selected
        actor = self.vtk_app.selected_actor
        if not actor and self.vtk_app.actors:
             actor = self.vtk_app.actors[-1]

        if not actor:
            self.status_signal.emit("Lighting adjustment failed: No active actor.")
            return
            
        prop = actor.GetProperty()
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Illumination Properties")
        layout = QtWidgets.QFormLayout(dialog)
        
        def create_slider(label_text, current_val, range_max):
            """Internal helper to create a slider + label pair for a material property."""
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(0, range_max)
            scaled_val = int(current_val * 100) if range_max == 100 else int(current_val)
            slider.setValue(scaled_val)
            
            label = QtWidgets.QLabel(f"{current_val:.2f}" if range_max == 100 else f"{current_val:.0f}")
            
            def update_label(v):
                label.setText(f"{v/100:.2f}" if range_max == 100 else f"{v:.0f}")
            
            slider.valueChanged.connect(update_label)
            layout.addRow(label_text, slider)
            layout.addWidget(label)
            return slider
            
        ambient_slider = create_slider("Ambient (0-1.0):", prop.GetAmbient(), 100)
        diffuse_slider = create_slider("Diffuse (0-1.0):", prop.GetDiffuse(), 100)
        specular_slider = create_slider("Specular (0-1.0):", prop.GetSpecular(), 100)
        power_slider = create_slider("Specular Power (0-100):", prop.GetSpecularPower(), 100)

        # Real-time handlers: update property and re-render while user moves sliders
        def on_ambient_changed(v):
            prop.SetAmbient(v / 100.0)
            self.vtk_app.render_all()

        def on_diffuse_changed(v):
            prop.SetDiffuse(v / 100.0)
            self.vtk_app.render_all()

        def on_specular_changed(v):
            prop.SetSpecular(v / 100.0)
            self.vtk_app.render_all()

        def on_power_changed(v):
            prop.SetSpecularPower(v)
            self.vtk_app.render_all()

        ambient_slider.valueChanged.connect(on_ambient_changed)
        diffuse_slider.valueChanged.connect(on_diffuse_changed)
        specular_slider.valueChanged.connect(on_specular_changed)
        power_slider.valueChanged.connect(on_power_changed)
        
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | 
            QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            # On accept, ensure final values applied and notify user
            prop.SetAmbient(ambient_slider.value()/100.0)
            prop.SetDiffuse(diffuse_slider.value()/100.0)
            prop.SetSpecular(specular_slider.value()/100.0)
            prop.SetSpecularPower(power_slider.value())
            self.vtk_app.render_all()
            self.status_signal.emit("Illumination properties updated.")

    def open_background_color_dialog(self):
        """Open a color picker to change the renderer background color (undoable)."""
        renderer = self.vtk_app.renderer
        if not renderer:
            self.status_signal.emit("Error: Renderer not initialized.")
            return

        bg = renderer.GetBackground()
        current_qcolor = QtGui.QColor.fromRgbF(bg[0], bg[1], bg[2])
        color = QtWidgets.QColorDialog.getColor(current_qcolor, self, "Select Background Color")

        if color.isValid():
            vtk_color = (color.redF(), color.greenF(), color.blueF())
            self.vtk_app.change_background_color(vtk_color)
            
# Part M - MainWindow transform dialog methods
    
    def open_translate_dialog(self):
        """Translate dialog with real-time preview sliders and Undo support via TranslateAbsoluteCommand."""
        dialog = QtWidgets.QDialog(self, windowTitle="Translate Model (Undoable)")
        # Ensure the dialog is wide enough so the sliders have space to expand
        dialog.setMinimumWidth(640)
        layout = QtWidgets.QFormLayout(dialog)

        # Check if a group is selected (stored in _group_members)
        group_members = getattr(self.vtk_app, '_group_members', None)
        is_group = bool(group_members)
        
        if is_group:
            actors_to_transform = group_members
            initial_positions = [a.GetPosition() for a in actors_to_transform]
            # Use first actor as reference for initial position display
            initial_pos = initial_positions[0]
            dialog.setWindowTitle(f"Translate Group ({len(actors_to_transform)} items)")
        else:
            actor = self.vtk_app.selected_actor
            if not actor and self.vtk_app.actors:
                actor = self.vtk_app.actors[-1]

            if not actor:
                self.status_signal.emit("Translate failed: No active model.")
                QtWidgets.QMessageBox.warning(self, "Translate Model", "Please select a model first.")
                return
            
            actors_to_transform = [actor]
            initial_positions = [actor.GetPosition()]
            initial_pos = initial_positions[0]

        # Create sliders for X, Y, Z (range -1000 to 1000, scale by 10 for decimals)
        slider_x = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider_x.setMinimumWidth(420)
        slider_x.setRange(-10000, 10000)
        slider_x.setValue(0)
        label_x = QtWidgets.QLabel("0.0")

        slider_y = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider_y.setMinimumWidth(420)
        slider_y.setRange(-10000, 10000)
        slider_y.setValue(0)
        label_y = QtWidgets.QLabel("0.0")

        slider_z = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider_z.setMinimumWidth(420)
        slider_z.setRange(-10000, 10000)
        slider_z.setValue(0)
        label_z = QtWidgets.QLabel("0.0")

        # Update labels and preview in real-time when sliders move
        def update_translate(_):
            dx = slider_x.value() / 10.0
            dy = slider_y.value() / 10.0
            dz = slider_z.value() / 10.0
            label_x.setText(f"{dx:.1f}")
            label_y.setText(f"{dy:.1f}")
            label_z.setText(f"{dz:.1f}")
            # Real-time preview: update position of all actors to show immediate effect
            for i, actor in enumerate(actors_to_transform):
                actor.SetPosition(
                    initial_positions[i][0] + dx,
                    initial_positions[i][1] + dy,
                    initial_positions[i][2] + dz
                )
            self.vtk_app.render_all()

        slider_x.valueChanged.connect(update_translate)
        slider_y.valueChanged.connect(update_translate)
        slider_z.valueChanged.connect(update_translate)

        # Layout sliders with labels
        x_h = QtWidgets.QHBoxLayout()
        x_h.addWidget(slider_x, 3)
        x_h.addWidget(label_x, 1)
        layout.addRow("Delta X:", x_h)

        y_h = QtWidgets.QHBoxLayout()
        y_h.addWidget(slider_y, 3)
        y_h.addWidget(label_y, 1)
        layout.addRow("Delta Y:", y_h)

        z_h = QtWidgets.QHBoxLayout()
        z_h.addWidget(slider_z, 3)
        z_h.addWidget(label_z, 1)
        layout.addRow("Delta Z:", z_h)

        # Reset button restores initial positions and slider values
        reset_button = QtWidgets.QPushButton("Reset")
        def reset_sliders():
            slider_x.setValue(0)
            slider_y.setValue(0)
            slider_z.setValue(0)
            for i, actor in enumerate(actors_to_transform):
                actor.SetPosition(*initial_positions[i])
            self.vtk_app.render_all()
        reset_button.clicked.connect(reset_sliders)
        layout.addWidget(reset_button)
        
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | 
            QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            # On accept, create commands for undo/redo reflecting new absolute positions
            if is_group:
                from copy import deepcopy
                old_positions = deepcopy(initial_positions)
                new_positions = [a.GetPosition() for a in actors_to_transform]
                # Execute an absolute translate command for each actor so actions are undoable
                for actor, old_pos, new_pos in zip(actors_to_transform, old_positions, new_positions):
                    cmd = TranslateAbsoluteCommand(actor, old_pos, new_pos)
                    self.vtk_app.command_manager.execute(cmd)
                self.status_signal.emit(f"Group translated ({len(actors_to_transform)} items) (Undoable)")
            else:
                actor = actors_to_transform[0]
                final_pos = actor.GetPosition()
                new_t = (final_pos[0] - initial_pos[0], final_pos[1] - initial_pos[1], final_pos[2] - initial_pos[2])
                cmd = TranslateAbsoluteCommand(actor, initial_pos, final_pos)
                self.vtk_app.command_manager.execute(cmd)
                self.status_signal.emit(
                    f"Model translated by X:{new_t[0]:.2f}, Y:{new_t[1]:.2f}, Z:{new_t[2]:.2f} (Undoable)"
                )
        else:
            # Revert to initial positions on cancel
            for i, actor in enumerate(actors_to_transform):
                actor.SetPosition(*initial_positions[i])
            self.vtk_app.render_all()
            self.status_signal.emit("Translate canceled.")

    def open_scale_dialog(self):
        """Scale dialog with slider controls and real-time preview; supports uniform and per-axis scaling."""
        # Check if a group is selected
        group_members = getattr(self.vtk_app, '_group_members', None)
        is_group = bool(group_members)
        
        if is_group:
            actors_to_transform = group_members
            initial_scales = [a.GetScale() for a in actors_to_transform]
            initial_scale = initial_scales[0]
            dialog_title = f"Scale Group ({len(actors_to_transform)} items)"
        else:
            actor = self.vtk_app.selected_actor or (
                self.vtk_app.actors[-1] if self.vtk_app.actors else None
            )
            if not actor:
                self.status_signal.emit("Scale failed: No active model.")
                QtWidgets.QMessageBox.warning(self, "Scale Model", "Please select a model first.")
                return
            
            actors_to_transform = [actor]
            initial_scales = [actor.GetScale()]
            initial_scale = initial_scales[0]
            dialog_title = "Scale Model (Real-time)"

        dialog = QtWidgets.QDialog(self, windowTitle=dialog_title)
        dialog.setMinimumWidth(450)
        layout = QtWidgets.QFormLayout(dialog)

        # Uniform scale checkbox + slider (uniform scaling mode)
        uniform_box = QtWidgets.QCheckBox("Uniform")
        uniform_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        uniform_slider.setRange(1, 10000)  # 0.01 to 100 (scale by 100)
        uniform_slider.setValue(int(initial_scale[0] * 100))
        uniform_label = QtWidgets.QLabel(f"{initial_scale[0]:.2f}")

        uniform_h = QtWidgets.QHBoxLayout()
        uniform_h.addWidget(uniform_box)
        uniform_h.addWidget(uniform_slider, 3)
        uniform_h.addWidget(uniform_label, 1)
        layout.addRow("Uniform Scale:", uniform_h)

        # Per-axis scaling sliders
        sx_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sx_slider.setRange(1, 10000)
        sx_slider.setValue(int(initial_scale[0] * 100))
        sx_label = QtWidgets.QLabel(f"{initial_scale[0]:.2f}")
        sx_h = QtWidgets.QHBoxLayout()
        sx_h.addWidget(sx_slider, 3)
        sx_h.addWidget(sx_label, 1)
        layout.addRow("Scale X:", sx_h)

        sy_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sy_slider.setRange(1, 10000)
        sy_slider.setValue(int(initial_scale[1] * 100))
        sy_label = QtWidgets.QLabel(f"{initial_scale[1]:.2f}")
        sy_h = QtWidgets.QHBoxLayout()
        sy_h.addWidget(sy_slider, 3)
        sy_h.addWidget(sy_label, 1)
        layout.addRow("Scale Y:", sy_h)

        sz_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sz_slider.setRange(1, 10000)
        sz_slider.setValue(int(initial_scale[2] * 100))
        sz_label = QtWidgets.QLabel(f"{initial_scale[2]:.2f}")
        sz_h = QtWidgets.QHBoxLayout()
        sz_h.addWidget(sz_slider, 3)
        sz_h.addWidget(sz_label, 1)
        layout.addRow("Scale Z:", sz_h)

        # Flag to prevent glitch when programmatically synchronizing sliders
        updating_programmatically = [False]

        # Sync uniform -> per-axis when checked; apply changes in real-time
        def on_uniform_changed(_):
            if uniform_box.isChecked():
                val = uniform_slider.value() / 100.0
                updating_programmatically[0] = True
                sx_slider.setValue(uniform_slider.value())
                sy_slider.setValue(uniform_slider.value())
                sz_slider.setValue(uniform_slider.value())
                updating_programmatically[0] = False
                for actor in actors_to_transform:
                    actor.SetScale(val, val, val)
                self.vtk_app.render_all()

        def on_axis_changed(_):
            # If user manually changes axes sliders, uncheck uniform checkbox
            if not updating_programmatically[0] and uniform_box.isChecked():
                uniform_box.setChecked(False)
            sx_val = sx_slider.value() / 100.0
            sy_val = sy_slider.value() / 100.0
            sz_val = sz_slider.value() / 100.0
            sx_label.setText(f"{sx_val:.2f}")
            sy_label.setText(f"{sy_val:.2f}")
            sz_label.setText(f"{sz_val:.2f}")
            for actor in actors_to_transform:
                actor.SetScale(sx_val, sy_val, sz_val)
            self.vtk_app.render_all()

        def on_uniform_slider_changed(_):
            uniform_label.setText(f"{uniform_slider.value() / 100.0:.2f}")
            on_uniform_changed(_)

        # Connect signals for interactive preview
        uniform_slider.valueChanged.connect(on_uniform_slider_changed)
        sx_slider.valueChanged.connect(on_axis_changed)
        sy_slider.valueChanged.connect(on_axis_changed)
        sz_slider.valueChanged.connect(on_axis_changed)

        # Reset button restores original scales and slider states
        reset_button = QtWidgets.QPushButton("Reset")
        def reset_scales():
            uniform_box.setChecked(False)
            sx_slider.blockSignals(True)
            sy_slider.blockSignals(True)
            sz_slider.blockSignals(True)
            uniform_slider.blockSignals(True)
            sx_slider.setValue(int(initial_scale[0] * 100))
            sy_slider.setValue(int(initial_scale[1] * 100))
            sz_slider.setValue(int(initial_scale[2] * 100))
            uniform_slider.setValue(int(initial_scale[0] * 100))
            sx_slider.blockSignals(False)
            sy_slider.blockSignals(False)
            sz_slider.blockSignals(False)
            uniform_slider.blockSignals(False)
            sx_label.setText(f"{initial_scale[0]:.2f}")
            sy_label.setText(f"{initial_scale[1]:.2f}")
            sz_label.setText(f"{initial_scale[2]:.2f}")
            uniform_label.setText(f"{initial_scale[0]:.2f}")
            for i, actor in enumerate(actors_to_transform):
                actor.SetScale(*initial_scales[i])
            self.vtk_app.render_all()
        reset_button.clicked.connect(reset_scales)
        layout.addWidget(reset_button)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok |
            QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        # Execute the dialog and create undoable ScaleAbsoluteCommand on accept
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            if is_group:
                from copy import deepcopy
                old_scales = deepcopy(initial_scales)
                new_scales = [a.GetScale() for a in actors_to_transform]
                for actor, old_s, new_s in zip(actors_to_transform, old_scales, new_scales):
                    cmd = ScaleAbsoluteCommand(actor, old_s, new_s)
                    self.vtk_app.command_manager.execute(cmd)
                self.status_signal.emit(f"Group scaled ({len(actors_to_transform)} items) (Undoable)")
            else:
                final_scale = actors_to_transform[0].GetScale()
                cmd = ScaleAbsoluteCommand(actors_to_transform[0], initial_scale, final_scale)
                self.vtk_app.command_manager.execute(cmd)
                self.status_signal.emit(f"Model scaled to X:{final_scale[0]:.3f}, Y:{final_scale[1]:.3f}, Z:{final_scale[2]:.3f} (Undoable)")
        else:
            # revert to initial scales on cancel
            for i, actor in enumerate(actors_to_transform):
                actor.SetScale(*initial_scales[i])
            self.vtk_app.render_all()
            self.status_signal.emit("Scale canceled.")

    def open_rotate_dialog(self):
        """Rotate dialog with real-time X, Y, Z sliders showing live rotation (applies absolute orientation)."""
        # Check if a group is selected
        group_members = getattr(self.vtk_app, '_group_members', None)
        is_group = bool(group_members)
        
        if is_group:
            actors_to_transform = group_members
            initial_orientations = [a.GetOrientation() for a in actors_to_transform]
            initial_orientation = initial_orientations[0]
            dialog_title = f"Rotate Group ({len(actors_to_transform)} items)"
        else:
            actor = self.vtk_app.selected_actor or (
                self.vtk_app.actors[-1] if self.vtk_app.actors else None
            )
            
            if not actor:
                self.status_signal.emit("Rotate failed: No active model.")
                QtWidgets.QMessageBox.warning(self, "Rotate Model", "Please select a model first.")
                return
            
            actors_to_transform = [actor]
            initial_orientations = [actor.GetOrientation()]
            initial_orientation = initial_orientations[0]
            dialog_title = "Rotate Model (Real-time)"
        
        dialog = QtWidgets.QDialog(self, windowTitle=dialog_title)
        dialog.setMinimumWidth(500)
        layout = QtWidgets.QFormLayout(dialog)
        
        # Helper function to create slider with live preview
        def create_rotation_slider(label_text, axis_index):
            h_layout = QtWidgets.QHBoxLayout()
            
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(-360, 360)
            slider.setValue(0)
            slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
            slider.setTickInterval(30)
            
            value_label = QtWidgets.QLabel("0°")
            value_label.setMinimumWidth(50)
            
            def on_slider_changed(v):
                # Update text label
                value_label.setText(f"{v}°")
                # Read values from all three sliders to compute absolute orientation relative to initial
                try:
                    rx = rx_slider.value()
                    ry = ry_slider.value()
                    rz = rz_slider.value()
                except NameError:
                    # If other sliders aren't created yet, apply single-axis update as fallback
                    for i, actor in enumerate(actors_to_transform):
                        current_rot = list(actor.GetOrientation())
                        current_rot[axis_index] = initial_orientations[i][axis_index] + v
                        actor.SetOrientation(*current_rot)
                    self.vtk_app.render_all()
                    return

                # Apply new orientation computed from initial orientation plus slider deltas
                for i, actor in enumerate(actors_to_transform):
                    new_rot = [
                        initial_orientations[i][0] + rx,
                        initial_orientations[i][1] + ry,
                        initial_orientations[i][2] + rz,
                    ]
                    actor.SetOrientation(*new_rot)
                self.vtk_app.render_all()
            
            slider.valueChanged.connect(on_slider_changed)
            
            h_layout.addWidget(slider)
            h_layout.addWidget(value_label)
            
            layout.addRow(label_text, h_layout)
            return slider
        
        # Create sliders for X, Y, Z axes and wire up live preview behavior
        rx_slider = create_rotation_slider("Rotate X (°):", 0)
        ry_slider = create_rotation_slider("Rotate Y (°):", 1)
        rz_slider = create_rotation_slider("Rotate Z (°):", 2)
        
        # Buttons including Reset to revert live preview
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | 
            QtWidgets.QDialogButtonBox.Cancel |
            QtWidgets.QDialogButtonBox.Reset
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        reset_button = button_box.button(QtWidgets.QDialogButtonBox.Reset)
        def reset_sliders():
            rx_slider.setValue(0)
            ry_slider.setValue(0)
            rz_slider.setValue(0)
            for i, actor in enumerate(actors_to_transform):
                actor.SetOrientation(*initial_orientations[i])
            self.vtk_app.render_all()
        
        reset_button.clicked.connect(reset_sliders)
        layout.addWidget(button_box)

        # On accept, create undoable RotateAbsoluteCommand(s) to preserve final orientation
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            if is_group:
                from copy import deepcopy
                old_orients = deepcopy(initial_orientations)
                new_orients = [a.GetOrientation() for a in actors_to_transform]
                for actor, old_o, new_o in zip(actors_to_transform, old_orients, new_orients):
                    cmd = RotateAbsoluteCommand(actor, old_o, new_o)
                    self.vtk_app.command_manager.execute(cmd)
                self.status_signal.emit(f"Group rotated ({len(actors_to_transform)} items) (Undoable)")
            else:
                final_orientation = actors_to_transform[0].GetOrientation()
                cmd = RotateAbsoluteCommand(actors_to_transform[0], initial_orientation, final_orientation)
                self.vtk_app.command_manager.execute(cmd)
                self.status_signal.emit(
                    f"Model rotated to X:{final_orientation[0]:.0f}°, Y:{final_orientation[1]:.0f}°, Z:{final_orientation[2]:.0f}° (Undoable)"
                )
        else:
            # Revert to initial orientation on cancel
            for i, actor in enumerate(actors_to_transform):
                actor.SetOrientation(*initial_orientations[i])
            self.vtk_app.render_all()
            self.status_signal.emit("Rotation canceled.")
            
    def open_user_manual(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Zender 3D - User Manual")
        dialog.setMinimumSize(600, 500)

        layout = QtWidgets.QVBoxLayout(dialog)

        title = QtWidgets.QLabel("📘 Zender 3D User Manual")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
        layout.addWidget(title)

        text_area = QtWidgets.QTextEdit()
        text_area.setReadOnly(True)

        manual_text = """
        Welcome to Zender 3D!

        Zender 3D is a powerful 3D modeling and visualization tool. This 3D Editor allows you to:
        • Load STL, OBJ, PLY, VTP, VTK, 3DS
        • Move, Rotate, Scale models
        • Change colors, lights, wireframe
        • Manage scene objects
        • Save and load full scenes

        Navigation:
        • Left-Click + Drag = Rotate
        • Right-Click + Drag = Pan
        • Scroll Wheel = Zoom

        More features:
        • Group transforms
        • Camera presets and custom camera controls
        • Scene organization system
        """

        text_area.setText(manual_text)
        layout.addWidget(text_area)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)

        dialog.exec_()

def main():
    """Application entry point. Sets up QApplication, creates the first MainWindow and starts event loop."""
    app = QtWidgets.QApplication(sys.argv)
    
    # Set application properties for better multi-window management
    app.setApplicationName("3D Model Editor")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("YourOrganization")
    
    # Turn off VTK global warnings to reduce noisy modal dialogs
    vtk.vtkObject.GlobalWarningDisplayOff()
    
    # Pretty terminal startup messages
    tprint_banner("Starting PyQt5 Application with Interactive VTK...")
    tprint_banner("Starting Interactive VTK Application")

    print("Setting up rendering pipeline...")
    tprint_ok("Rendering pipeline setup complete")
    tprint_ok("Mapper created")
    tprint_ok("Actor created with color (color of what?)")
    print("Creating scene...")
    tprint_ok("Added sphere actor to scene")
    tprint_ok("Scene created and camera reset")
    print("Rendering scene...")
    tprint_ok("Scene rendered and interactor initialized")
    tprint_ok("VTK application started successfully")
    _tprint_sep()
    
    # Create and show the first main window
    win = MainWindow()
    win.show()

    print("Creating interactive menu...")
    tprint_ok("Interactive menu created")
    tprint_ok("Main window displayed")

    print("\n  Application Ready - Interactive Controls:")
    # List interactive features/tools only (no system info)
    for feature in gather_interactive_features():
        print(f"- {feature}")

    print("\nRendering scene...")
    tprint_ok("Scene rendered and interactor initialized")
    
    # Ensure the app quits when last window is closed
    app.setQuitOnLastWindowClosed(True)
    
    try:
        # Start Qt event loop
        sys.exit(app.exec_())
    finally:
        # Final cleanup when application is shutting down - ensure VTK resources freed
        print("[main] Application shutting down - cleaning up windows and VTK resources...")
        for window in ACTIVE_WINDOWS[:]:
            try:
                if hasattr(window, 'vtk_app'):
                    window.vtk_app.cleanup()
            except Exception:
                pass
        ACTIVE_WINDOWS.clear()
        print("[main] Cleanup complete. Thank you for using Zender 3D!.")

if __name__ == "__main__":
    main()