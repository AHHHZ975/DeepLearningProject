from os import path
import os
import numpy as np
from numpy.core.fromnumeric import shape
import open3d as o3d
import sys
sys.path.append('/home/ahz/Desktop/3D-Reconstruction/3D-Reconstruction')
import Config as cfg
import matplotlib.pyplot as plt
from Utils import Utils
import cv2
import open3d.visualization.gui as gui
from pathlib import Path


def loadImage(img):
    """
    It loads an image file.
    Supported extenstions file:
        1- .jpg 
        2- .png
    """
    return o3d.io.read_image(img)

def loadMesh(mesh):
    """
    It loads a mesh file.
    Supported extenstions file:
        1- .obj 
        2- .ply
        3- .stl
        4- .off
        5- .gltf/.glb
    """
    return o3d.io.read_triangle_mesh(mesh)

def loadPointCloud(pcl):
    """
    It loads a point cloud file.
    Supported extenstions file:
        1- .ply 
        2- .pd
        3- .pts
        4- .xyz/.xyzn/.xyzrgb
        5- TODO .npz
    """
    return o3d.io.read_point_cloud(pcl)

def pcl2Voxel(pointCloud, voxelSize=20):
    """
    This function converts a point cloud to its 
    corresponding voxel representation.
    """
    return o3d.geometry.VoxelGrid.create_from_point_cloud(pointCloud, voxel_size=voxelSize)

def visualize(object, displayBoundingBox=False):
    """
     It visualizes a point cloud shape.
     It also detects the bounding box of the point cloud and displays it.
    """
    if displayBoundingBox:
        boundingBox = getBoundingBox(object)
        boundingBox.color = (1, 0, 0)
        # boundingBox = pointCloud.get_oriented_bounding_box()
        o3d.visualization.draw_geometries([object, boundingBox], 
                                    # zoom=0.3412, 
                                    # front=[0.4257, -0.2125, -0.8795], 
                                    # lookat=[2.6172, 2.0475, 1.532], 
                                    # up=[-0.0694, -0.9768, 0.2024]
                                    ) # Visualize the point cloud
    else:
        o3d.visualization.draw_geometries([object], 
                                    # zoom=0.3412, 
                                    # front=[0.4257, -0.2125, -0.8795], 
                                    # lookat=[2.6172, 2.0475, 1.532], 
                                    # up=[-0.0694, -0.9768, 0.2024]
                                    ) # Visualize the point cloud

def paint(object, color=[0.6, 0.3, 0.3]):
    """
      It paints a point cloud with a ceratin color.
      The color should be normalized to a value in the range of [0, 1].
    """
    return object.paint_uniform_color(color)

def getBoundingBox(obj):
    """
    The PointCloud geometry type has bounding volumes as all 
    other geometry types in Open3D. Currently, Open3D implements
    an AxisAlignedBoundingBox and an OrientedBoundingBox that
    can also be used to crop the geometry.
    """
    return obj.get_axis_aligned_bounding_box()

def getPointCloudDistance(srcPC, targetPC):
    """
    Open3D provides the method compute_point_cloud_distance to 
    compute the distance from a source point cloud to a target 
    point cloud. I.e., it computes for each point in the source
    point cloud the distance to the closest point in the target 
    point cloud. This function computes the difference between 
    two point clouds. Note that this method could also be used
    to compute the "Chamfer Distance" between two point clouds.
    """
    pass

def sampleMesh(mesh, sampleSize):
    """
    This function loads a mesh file (.obj or .ply) and then smaples it
    and converts it to point cloud.
    
    1- "Uniform sampling": The simplest method is sample_points_uniformly
    that uniformly samples points from the 3D surface based on the triangle
    area. The parameter number_of_points defines how many points are sampled
    from the triangle surface. This methodcan yield clusters of points on the
    surface.
    
    2- "Poisson disk sampling": It can evenly distribute the points on the surface. 
    The method sample_points_poisson_disk implements sample elimination. It starts
    with a sampled point cloud and removes points to satisfy the sampling criterion. 
    The method supports two options to provide the initial point cloud:
        a) Default via the parameter init_factor: The method first samples uniformly 
           a point cloud from the mesh with (init_factor x number_of_points) and uses 
           this for the elimination.
        b) One can provide a point cloud and pass it to the sample_points_poisson_disk 
           method. Then, this point cloud is used for elimination.
    """
    # meshSampled = mesh.sample_points_poisson_disk(number_of_points=samplesNumber)
    meshSampled = mesh.sample_points_uniformly(number_of_points=sampleSize)
    return meshSampled

def XYZ2PointCloud(xyz):
    """
    This function converts a numpy 3d array to
    an Open3D point cloud data format.
    """
    pointCloud = o3d.geometry.PointCloud()
    pointCloud.points = o3d.utility.Vector3dVector(xyz)
    return pointCloud

def pointCloud2XYZ(pointCloud):
    """
    This function converts an Open3D point cloud 
    data format to a numpy 3d array.
    """
    return np.asarray(pointCloud.points)

def customVisualize(object):

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(15.0, 15.0)
        return False

    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    def capture_depth(vis):
        depth = vis.capture_depth_float_buffer()
        plt.imshow(np.asarray(depth))
        plt.show()
        return False

    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.imshow(np.asarray(image))
        plt.show()
        return False
    

    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    key_to_callback[ord("R")] = rotate_view
    key_to_callback[ord("D")] = capture_depth
    key_to_callback[ord("C")] = capture_image

    o3d.visualization.draw_geometries_with_key_callbacks([object], key_to_callback)
    # o3d.visualization.draw_geometries_with_animation_callback([object], rotate_view)

def visualizeTrainData(path):
    """
    This function visualizes a set of data.
    """    
    imagePaths = Utils.getRGBsInDirectory(path)
    pointCloudPaths = Utils.getPLYsInDirectory(path)
    
    if len(imagePaths) != len(pointCloudPaths):
        print("The size of images and pointclouds are not equal in the directory.")

    for i in range(len(imagePaths)):        
        image = loadImage(imagePaths[i])        
        pointcloud = loadPointCloud(pointCloudPaths[i])
        print(imagePaths[i])
        print(pointCloudPaths[i])        
        o3d.visualization.draw_geometries([pointcloud, image],
                                           zoom=0.9, 
                                           front=[0.4, 0.5, -0.5],
                                           lookat=[0, 0, 0], 
                                           up=[0, 1, 0]
                                        )
        
def meshesToPointClouds(path, sampleSize):
    """
    This function loads the .obj model of ShapeNet 
    dataset that has been placed in the directory 
    of the project. Then converts them to pointclouds
    and returns them as an output.
    """       
    meshObject = loadMesh(path)
    pointCloud = sampleMesh(meshObject, sampleSize)
    return pointCloud

def meshesToRGBs(meshPath, imageName, width, height, visible=False):
    """
    This function load a mesh with its textures,
    renders it in a 3d scene and then returns
    an image from a desired view point. 
    """
    model = o3d.io.read_triangle_model(meshPath)
    for mi in model.meshes:
        mi.mesh.compute_vertex_normals()

    if visible:
        app = gui.Application.instance
        app.initialize()
        vis = o3d.visualization.O3DVisualizer("Rendered data", 2048, 1024)     
        app.add_window(vis)

        # Add the model (mesh + material + texture) to the scene.
        for mi in model.meshes:            
            vis.add_geometry(mi.mesh_name, mi.mesh, model.materials[mi.material_idx])
        
        # Setting for lighting of the scene
        direction = (0.577, 0.577, 0.577)
        vis.scene.set_lighting(vis.scene.LightingProfile.NO_SHADOWS, direction)

        # Add a directional light to the environment
        # color = np.array([25, 25, 25], dtype='float32')
        # position = np.array([0.4, 0.5, 0.5], dtype='float32')
        # intensity = 7000
        # cast_shadows = 0
        # vis.scene.scene.add_directional_light("Directional light", color, position, intensity, cast_shadows)        

        # Optionally set the camera field of view (to zoom in a bit)
        vertical_field_of_view = 65.0  # between 5 and 90 degrees
        aspect_ratio = width / height  # azimuth over elevation
        near_plane = 0.1
        far_plane = 50
        fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
        vis.scene.camera.set_projection(vertical_field_of_view, aspect_ratio, near_plane, far_plane, fov_type)

        # vis.reset_camera_to_default()
        # Look at the origin from the front (along the -Z direction, into the screen), with Y as Up.
        center = [0, 0, 0]  # look_at target
        eye =  [0.5, 0.4, -0.6]  # camera position
        up = [0, 1, 0]  # camera orientation        
        vis.setup_camera(0.0, center, eye, up)

        # Display the image in a separate window
        # (Note: OpenCV expects the color in BGR format, so swop red and blue.)
        img_o3d = app.instance.render_to_image(vis.scene, width, height)        
        image = cv2.cvtColor(np.array(img_o3d), cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(imageName, image)
            
        app.instance.run()
    
    else:
        render = o3d.visualization.rendering.OffscreenRenderer(width, height, headless=False)
        
        # Pick a background colour (default is light gray)
        render.scene.set_background([10, 10, 10, 10])     
        
        # Add the model (mesh + material + texture) to the scene.
        render.scene.add_model('Model', model)
        
        # Setting for lighting of the scene
        direction = (0.577, -0.577, 0.577)
        render.scene.set_lighting(render.scene.LightingProfile.NO_SHADOWS, direction)
        
        # Optionally set the camera field of view (to zoom in a bit)
        vertical_field_of_view = 65.0  # between 5 and 90 degrees
        aspect_ratio = width / height  # azimuth over elevation
        near_plane = 0.1
        far_plane = 50
        fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
        render.scene.camera.set_projection(vertical_field_of_view, aspect_ratio, near_plane, far_plane, fov_type)

        # Look at the origin from the front (along the -Z direction, into the screen), with Y as Up.
        center = [0, 0, 0]  # look_at target
        eye =  [0.5, 0.4, -0.6]  # camera position
        up = [0, 1, 0]  # camera orientation
        render.scene.camera.look_at(center, eye, up)
    
        # Display the image in a separate window
        # (Note: OpenCV expects the color in BGR format, so swop red and blue.)
        img_o3d = render.render_to_image()
        image = cv2.cvtColor(np.array(img_o3d), cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(imageName, image)

def generateData(srcPath, destPath, width, height, numberOfData, visible=False, train=False):
    """
    This function generates train data (synthetic RGB images)
    and labels (pointclouds) from shapenet dataset.
    """
    # Create source and destination directory
    if train:
        srcPath += '/Train'
        destPath += '/Train'
    else:
        srcPath += '/Test'
        destPath += '/Test'

    Path(destPath).mkdir(parents=True, exist_ok=True)
    
    # Loading mesh objects that have been placed in the ShapeNet dataset directory
    shapeNetPaths = Utils.getOBJsInDirectory(srcPath)
    
    # Displaying the number of mesh objects (models) found in the shapenet directory 
    print(f"{len(shapeNetPaths)} mesh objects found in your shapenet directory.\n")    
    
    shapeNetPaths = shapeNetPaths[:numberOfData]

    # Generate train data
    for i, shapeNetPath in enumerate(shapeNetPaths):        
        Path(f'{destPath}/{i}').mkdir(parents=True, exist_ok=True)
        # Generate train data (synthetic RGB image) (x)
        imagePath = f'{cfg.ROOT_DIR}/{destPath}/{i}/{str(i)}.jpg'
        print(shapeNetPath)        
        meshesToRGBs(shapeNetPath, imagePath, width, height, visible)

        # Generate labels (pointclouds) (y)
        pointCloud = meshesToPointClouds(shapeNetPath, cfg.SAMPLE_SIZE)
            
        # Save pointclouds as .ply files        
        pointCloudPath = f'{cfg.ROOT_DIR}/{destPath}/{i}/{str(i)}.ply'
        o3d.io.write_point_cloud(pointCloudPath, pointCloud)
        
        # Save pointclouds as .txt files
        xyzPath = f'{cfg.ROOT_DIR}/{destPath}/{i}/{str(i)}.txt'        
        xyzPoints = pointCloud2XYZ(pointCloud)
        for xyzPoint in xyzPoints:
            xyzPoint[0] = round(xyzPoint[0], 6) 
            xyzPoint[1] = round(xyzPoint[1], 6)
            xyzPoint[2] = round(xyzPoint[2], 6)
            content = str(xyzPoint[0]) + '    ' + str(xyzPoint[1]) + '    ' + str(xyzPoint[2]) + '\n' 
            f = open(xyzPath, "a")
            f.write(content)
        f.close()

def main():
    # Generate train data
    destinationPath = 'Output/GeneratedData_1'
    sourcePath = f'{cfg.SHAPENET_DIR}'
    trainData = generateData(sourcePath, destinationPath, cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT, 1373, train=True, visible=False)
    testData = generateData(sourcePath, destinationPath, cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT, 221, train=False, visible=False)

    # Show te generated train data
    destinationPath = cfg.ROOT_DIR + '/Output/GeneratedData_1'
    visualizeTrainData(destinationPath)

if __name__ == "__main__":
    # main()
    
    
    pcl = meshesToPointClouds(cfg.SHAPENET_DIR + '/Test/03691459/508e66670fb3830c2cd6a352b52d97b9/models/model_normalized.obj', 700)
    visualize(pcl)


    
    
    
    ####### To generate test result for the paper

    # destinationPath = cfg.ROOT_DIR + '/Output/GeneratedData/Test/02876657/21/21.ply'
    # pointcloud = loadPointCloud(destinationPath)
    # o3d.visualization.draw_geometries([pointcloud],
    #                                 zoom=0.9, 
    #                                 front=[0.4, 0.5, -0.5],
    #                                 lookat=[0, 0, 0], 
    #                                 up=[0, 1, 0]
    #                             )
