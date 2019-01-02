import util
import time
import numpy as np
from subprocess import STDOUT, check_output
from open3d import *
import copy
import uuid
import logging
logger = logging.getLogger(__name__)
def super4pcs(pc_src, pc_tgt):
    """
    Compute the relative transform between two point clouds using super4pcs algorithm. 
    pc_src,pc_tgt: [n1,3],[n2,3]
    return:
    R_hat: [4,4]
    """
    filenameS = f"tmp/pc_{str(uuid.uuid4())}.obj"
    filenameT = f"tmp/pc_{str(uuid.uuid4())}.obj"
    util.pc2obj(filenameS,pc_src.T)
    util.pc2obj(filenameT,pc_tgt.T)

    filenameO = f"tmp/{str(uuid.uuid4())}"
    cmd = './Super4PCS -i {obj0} {obj1} -o 0.5 -d 0.1 -t 1000 -n 200 -m {fileO}'.format(obj0=filenameT,obj1=filenameS,fileO=filenameO)
    try:
        output = check_output(cmd.strip().split(' '), stderr=STDOUT, timeout=60*4)
        R_hat = util.read_super4pcs_mat('{fileO}'.format(fileO=filenameO))
    except:
        R_hat = np.eye(4)
    try:
        cmd = f"rm {filenameO} {filenameS} {filenameT}"
        check_output(cmd.strip().split(' '), stderr=STDOUT)
    except:
        pass

    return R_hat

def preprocess_point_cloud(pcd, voxel_size):
    logger.info(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = voxel_down_sample(pcd, voxel_size)

    radius_normal = voxel_size * 2
    logger.info(":: Estimate normal with search radius %.3f." % radius_normal)
    estimate_normals(pcd_down, KDTreeSearchParamHybrid(
            radius = radius_normal, max_nn = 30))

    radius_feature = voxel_size * 5
    logger.info(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = compute_fpfh_feature(pcd_down,
            KDTreeSearchParamHybrid(radius = radius_feature, max_nn = 100))
    return pcd_down, pcd_fpfh

def open3d_global_registration(pc_src, pc_tgt):
    
    pcd_src = PointCloud()
    pcd_src.points = Vector3dVector(pc_src)
    pcd_tgt = PointCloud()
    pcd_tgt.points = Vector3dVector(pc_tgt)

    voxel_size = 0.05 # means 5cm for the dataset
    distance_threshold = voxel_size * 1.5

    source_down, source_fpfh = preprocess_point_cloud(pcd_src, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(pcd_tgt, voxel_size)

    start = time.time()
    
    logger.info(":: RANSAC registration on downsampled point clouds.")
    logger.info("   Since the downsampling voxel size is %.3f," % voxel_size)
    logger.info("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            distance_threshold,
            TransformationEstimationPointToPoint(False), 4,
            [CorrespondenceCheckerBasedOnEdgeLength(0.9),
            CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            RANSACConvergenceCriteria(4000000, 500))

    logger.info(result)
    logger.info("Global registration took %.3f sec.\n" % (time.time() - start))
    
    return result.transformation
            
def open3d_fast_global_registration(pc_src, pc_tgt):
    
    pcd_src = PointCloud()
    pcd_src.points = Vector3dVector(pc_src)
    pcd_tgt = PointCloud()
    pcd_tgt.points = Vector3dVector(pc_tgt)

    voxel_size = 0.05 # means 5cm for the dataset
    distance_threshold = voxel_size * 1.5

    source_down, source_fpfh = preprocess_point_cloud(pcd_src, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(pcd_tgt, voxel_size)
    start = time.time()
    logger.info(":: Apply fast global registration with distance threshold %.3f" % distance_threshold)
    result = registration_fast_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            FastGlobalRegistrationOption(
            maximum_correspondence_distance = distance_threshold))
    logger.info("Fast global registration took %.3f sec.\n" % (time.time() - start))

    return result.transformation

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    draw_geometries([source_temp, target])

def open3d_color_registration(pc_src,pc_tgt,color_src,color_tgt):

    pcd_src = PointCloud()
    pcd_src.points = Vector3dVector(pc_src)
    pcd_src.colors = Vector3dVector(color_src)
    pcd_tgt = PointCloud()
    pcd_tgt.points = Vector3dVector(pc_tgt)
    pcd_tgt.colors = Vector3dVector(color_tgt)

    voxel_size = 0.05 # means 5cm for the dataset
    distance_threshold = voxel_size * 1.5

    source_down, source_fpfh = preprocess_point_cloud(pcd_src, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(pcd_tgt, voxel_size)
    start = time.time()
    logger.info(":: RANSAC registration on downsampled point clouds.")
    logger.info("   Since the downsampling voxel size is %.3f," % voxel_size)
    logger.info("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            distance_threshold,
            TransformationEstimationPointToPoint(False), 4,
            [CorrespondenceCheckerBasedOnEdgeLength(0.9),
            CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            RANSACConvergenceCriteria(4000000, 500))

    # point to plane ICP
    # colored pointcloud registration
    # This is implementation of following paper
    # J. Park, Q.-Y. Zhou, V. Koltun,
    # Colored Point Cloud Registration Revisited, ICCV 2017
    voxel_radius = [ 0.04, 0.02, 0.01 ]
    max_iter = [ 50, 30, 14 ]
    #current_transformation = np.identity(4)
    current_transformation =result.transformation
    logger.info("3. Colored point cloud registration")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        logger.info([iter,radius,scale])

        logger.info("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = voxel_down_sample(pcd_src, radius)
        target_down = voxel_down_sample(pcd_tgt, radius)

        logger.info("3-2. Estimate normal.")
        estimate_normals(source_down, KDTreeSearchParamHybrid(
                radius = radius * 2, max_nn = 30))
        estimate_normals(target_down, KDTreeSearchParamHybrid(
                radius = radius * 2, max_nn = 30))

        logger.info("3-3. Applying colored point cloud registration")
        result_icp = registration_colored_icp(source_down, target_down,
                radius, current_transformation,
                ICPConvergenceCriteria(relative_fitness = 1e-6,
                relative_rmse = 1e-6, max_iteration = iter))
        current_transformation = result_icp.transformation

    return result_icp.transformation
