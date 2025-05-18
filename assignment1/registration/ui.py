import bpy
import mathutils
import bmesh

from .iterative_closest_point import *


class ObjectICPRegistration(bpy.types.Operator):
    bl_idname = "object.icp_rigid_registration"
    bl_label = "Rigid Registration with ICP"
    bl_options = {'REGISTER', 'UNDO'}

    # Input parameters
    bpy.types.WindowManager.rigid_registration_destination = bpy.props.PointerProperty(
        name="Destination", description="Destination mesh for rigid registration procedure",
        type=bpy.types.Object,
        poll=lambda _, obj: obj.type == 'MESH'
    )
    iterations: bpy.props.IntProperty(
        name="Iterations", description="Maximum number of iterations",
        min=1, max=100, default=10
    )
    epsilon: bpy.props.FloatProperty(
        name="ε", description="Minimum distance, below which the mesh is considered converged",
        min=0.0, step=0.005, max=0.05, default=0.01
    )
    k: bpy.props.FloatProperty(
        name="k", description="Point-pairs greater than k times the median distance apart are disregarded",
        min=0.1, step=0.01, max=5.0, default=2.0
    )
    num_points: bpy.props.IntProperty(
        name="# of Points", description="Maximum number of points to sample from the meshed for registration",
        min=1, step=1, default=500
    )
    distance_metric: bpy.props.EnumProperty(
        name="Distance Metric", description="Strategy to use when determining the optimal transformation",
        items=[
            ('POINT_TO_POINT', "Point-to-Point", ""),
            ('POINT_TO_PLANE', "Point-to-Plane", ""),
        ]
    )
    p: bpy.props.EnumProperty(
        name="Culling Scheme", description="The way to calculate the distance when culling",
        items=[
            ("1", "One", ""),
            ("2", "Two", ""),
            ("inf", "Inf", ""),
        ]
    )
    sampling_method: bpy.props.EnumProperty(
        name="Sampling Method", description="The way to sample the points from the target mesh",
        items=[
            ("random", "Random", ""),
            ("farthest_point", "Farthest Point", ""),
            ("normal_space", "Normal Space", ""),
        ]
    )
    matching_metric: bpy.props.EnumProperty(
        name="Matching Metric",
        description="The way to match the points from the source and target mesh",
        items=[
            ("euclid", "Euclidean", ""),
            ("normals", "Normals", "")
        ]
    )

    matching_method: bpy.props.EnumProperty(
        name="Matching Method", description="The way to match the points from the source and target mesh",
        items=[
            ("brute_force", "Brute Force", ""),
            ("kdtree", "KDTree", ""),
        ]
    )


    # Output parameters
    status: bpy.props.StringProperty(
        name="Registration Status", default="Status not set"
    )

    mse: bpy.props.StringProperty(
        name="Mean Squared Error", default="Not calculated"
    )
    hausdorff_distance: bpy.props.StringProperty(
        name="Hausdorff Distance", default="Not calculated"
    )

    @classmethod
    def poll(self, context):
        meshes = [obj for obj in context.view_layer.objects if obj.type == 'MESH']
        # Rigid registration is only available when a mesh is selected and more than one mesh is in the scene
        return (
                context.view_layer.objects.active.type == 'MESH' and
                len(meshes) > 1
        )

    def invoke(self, context, event):

        # This chooses a sensible default for the destination (any mesh other than the source)
        if context.window_manager.rigid_registration_destination is None:
            meshes = [obj for obj in context.view_layer.objects if obj.type == 'MESH']
            other_meshes = [m for m in meshes if m is not context.view_layer.objects.active]
            context.window_manager.rigid_registration_destination = other_meshes[-1]

        return self.execute(context)

    def execute(self, context):

        source_object = context.view_layer.objects.active
        destination_object = context.window_manager.rigid_registration_destination

        # Make sure a target is chosen
        if destination_object is None:
            return {'FINISHED'}

        # Produce BMesh types to work with
        source, destination = bmesh.new(), bmesh.new()
        source.from_mesh(source_object.data), destination.from_mesh(destination_object.data)

        # World transformations must be included in both meshes (we're not working in object-space)
        source.transform(source_object.matrix_world), destination.transform(destination_object.matrix_world)

        # Find a transformation for the source mesh
        try:
            # We call your implementation here!
            transformations = iterative_closest_point_registration(
                source, destination,
                self.k, self.num_points,
                self.iterations, self.epsilon,
                self.distance_metric,
                # TODO: Any additional configuration options you add can be passed in here
                p_norms = self.p,
                sampling_method = self.sampling_method,
                matching_method = self.matching_method,
                matching_metric = self.matching_metric,
            )
        except Exception as error:
            self.report({'WARNING'}, f"Rigid registration failed with error '{error}'")
            return {'CANCELLED'}

        # Determine if convergence was reached
        converged = len(transformations) < self.iterations
        self.status = (f"Converged in {len(transformations)} iterations" if converged
                       else f"Failed to converge after {self.iterations} iterations")

        # Apply the transformation to the source
        # This is done in world-space, leaving the mesh's coordinate space untouched
        source_object.matrix_world = net_transformation(transformations) @ source_object.matrix_world
        source_object.data.update()

        # BONUS: You could do more with this list of transformations; producing an animation for example!
        source_after = bmesh.new()
        source_after.from_mesh(source_object.data)
        source_after.transform(source_object.matrix_world)
        # Calculate the MSE and Hausdorff distance
        mse_value = calculate_mse(source_after, destination)
        hausdorff_value = calculate_hausdorff_distance(source_after, destination)
        # Update the properties with the calculated values
        self.mse = f"{mse_value:.2f}"
        self.hausdorff_distance = f"{hausdorff_value:.2f}"
        return {'FINISHED'}

    def draw_enum_pair(self, label, prop_name):
        row = self.layout.row()
        split = row.split(factor=0.6)
        split.label(text=label)
        split.prop(self, prop_name, text="")

    def draw(self, context):
        layout = self.layout

        # Hint to select an object
        if context.window_manager.rigid_registration_destination is None:
            layout.label(text="Select a destination for registration")

        # Object selection
        row = layout.row(align=True)
        row.prop(context.view_layer.objects, 'active', text="", expand=True, emboss=False)
        row.label(icon='RIGHTARROW')
        row.prop(context.window_manager, 'rigid_registration_destination', text="", expand=True)
        layout.separator()

        # Convergence parameters
        row = layout.row(align=True)
        row.prop(self, 'iterations')
        row.separator()
        row.prop(self, 'epsilon')
        layout.separator()

        # Other hyperparameters
        box = layout.box()
        box.label(text="Hyperparameters")
        self.draw_enum_pair( "Point Rejection Coefficient", "k")
        self.draw_enum_pair("Number of Points", "num_points")
        self.draw_enum_pair("Distance Metric","distance_metric")
        self.draw_enum_pair("Culling Scheme", "p")
        self.draw_enum_pair("Sampling Method", 'sampling_method')
        self.draw_enum_pair("Matrching Metric", 'matching_metric')
        if self.sampling_method == "euclid":
            self.draw_enum_pair("Matching Method", 'matching_method')

        layout.separator()

        # TODO: If you add more features to your ICP implementation, you can provide UI to configure them

        layout.prop(self, 'status', text="Status", emboss=False)
        row = layout.row()
        split = row.split(factor=0.6)
        split.label(text="Mean Squared Error")
        split.label(text=self.mse)

        row = layout.row()
        split = row.split(factor=0.6)
        split.label(text="Hausdorff Distance")
        split.label(text=self.hausdorff_distance)


    @staticmethod
    def menu_func(menu, context):
        menu.layout.operator(ObjectICPRegistration.bl_idname)

    @classmethod
    def register(cls):
        # Add a menu item ('Layout -> Object -> Rigid Registration with ICP')
        bpy.types.VIEW3D_MT_object.append(cls.menu_func)
