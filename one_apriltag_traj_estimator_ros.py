#!/usr/bin/env python

import cv2
import apriltag
import numpy as np
import tf
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

class AprilTagPoseEstimator:
    def __init__(self, camera_matrix, dist_coeffs, tag_size, target_id):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.tag_size = tag_size
        self.target_id = target_id
        self.detector = apriltag.Detector(apriltag.DetectorOptions(families="tag36h11"))
        self.tracked_points = []
        self.poses = []
        self.bridge = CvBridge()
        self.image_timestamp = None

        self.initial_offset_x = None
        self.initial_offset_y = None
        self.initial_offset_z = None
        # self.initial_orientation_adjusted = False

        self.pose_pub = rospy.Publisher('/apriltag_pose', PoseStamped, queue_size=10)
        self.path_pub = rospy.Publisher('/apriltag_path', Path, queue_size=10)  # Path publisher
        self.image_pub = rospy.Publisher('/processed_image', Image, queue_size=10)
        self.path = Path()
        self.path.header.frame_id = "map"
        self.image_sub = rospy.Subscriber('/camera/color/image_raw/compressed', CompressedImage, self.image_callback)

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        display_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.detect_and_plot(cv_image, display_image)
        self.image_timestamp = msg.header.stamp

    def detect_and_plot(self, cv_image, display_image):
        cv_image = cv2.GaussianBlur(cv_image, (5, 5), 0)
        detections = self.detector.detect(cv_image)
        for detection in detections:
            if detection.tag_id == self.target_id:
                self.publish_pose(detection)
                self.display_trajectory(display_image)
                break

    def publish_pose(self, detection):
        img_points = detection.corners
        center_x = np.mean([point[0] for point in img_points])
        center_y = np.mean([point[1] for point in img_points])
        self.tracked_points.append((int(center_x), int(center_y)))

        obj_points = np.array([
            [-self.tag_size / 2, self.tag_size / 2, 0],
            [self.tag_size / 2, self.tag_size / 2, 0],
            [self.tag_size / 2, -self.tag_size / 2, 0],
            [-self.tag_size / 2, -self.tag_size / 2, 0]
        ])

        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, self.camera_matrix, self.dist_coeffs)

        if success:
            if not self.initial_offset_x:
                self.initial_offset_x = tvec[0]
                self.initial_offset_y = tvec[1]
                self.initial_offset_z = tvec[2]

            rotation_matrix, _ = cv2.Rodrigues(rvec)
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            quaternion = tf.transformations.quaternion_from_matrix(transform_matrix)

            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.image_timestamp if self.image_timestamp else rospy.Time.now()
            pose_msg.header.frame_id = "map"
            
            # pose_msg.pose.position.x = tvec[0]
            # pose_msg.pose.position.y = tvec[1]
            # pose_msg.pose.position.z = tvec[2]
            
            pose_msg.pose.position.x = tvec[0] - self.initial_offset_x
            pose_msg.pose.position.y = tvec[1] - self.initial_offset_y
            pose_msg.pose.position.z = tvec[2] - self.initial_offset_z

            pose_msg.pose.orientation.x = quaternion[0]
            pose_msg.pose.orientation.y = quaternion[1]
            pose_msg.pose.orientation.z = quaternion[2]
            pose_msg.pose.orientation.w = quaternion[3]

            self.pose_pub.publish(pose_msg)

            # Update the path with the new pose
            self.path.header.stamp = pose_msg.header.stamp
            self.path.poses.append(pose_msg)
            self.path_pub.publish(self.path)

            # Save the pose
            self.poses.append({
                'timestamp': pose_msg.header.stamp.to_sec(),
                'position': tvec.flatten(),
                'orientation': quaternion
            })

    def display_trajectory(self, image):
        overlay_image = image.copy()
        num_points = len(self.tracked_points)

        if num_points < 2:
            return

        for i in range(1, num_points):
            pt1 = self.tracked_points[i - 1]
            pt2 = self.tracked_points[i]
            color = (0, 0, 255)
            cv2.line(image, pt1, pt2, color, 4)

        image_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        self.image_pub.publish(image_msg)

    def save_poses_to_tum_format(self, filename):
        with open(filename, 'w') as f:
            for pose in self.poses:
                timestamp = pose['timestamp']
                tx, ty, tz = pose['position']
                qx, qy, qz, qw = pose['orientation']
                f.write(f"{timestamp} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")

if __name__ == '__main__':
    rospy.init_node('apriltag_pose_estimator')

    camera_matrix = np.array([
        [452.9435, 0, 652.1466],
        [0, 453.0292, 453.4423],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.array([-0.0033, 0.0129, 2.5278e-04, 0.0013], dtype=np.float32)
    tag_size = 0.3
    target_id = 0  # For the experiments at PolyUPool

    estimator = AprilTagPoseEstimator(camera_matrix, dist_coeffs, tag_size, target_id)
    rospy.spin()

    # Save the poses to a TXT file in TUM format
    # estimator.save_poses_to_tum_format('gt_poses_0207trail01_polyu.txt')
