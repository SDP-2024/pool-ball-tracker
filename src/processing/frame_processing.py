import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)

def get_top_down_view(frame_1, frame_2, table_pts_cam1, table_pts_cam2):
    # Compute homography matrices
    output_size = (800, 400)
    table_rect = np.float32([[0, 0], [output_size[0], 0], [0, output_size[1]], [output_size[0], output_size[1]]])

    H1 = cv2.getPerspectiveTransform(table_pts_cam1, table_rect)
    H2 = cv2.getPerspectiveTransform(table_pts_cam2, table_rect)

    top_down_view1 = cv2.warpPerspective(frame_1, H1, output_size)
    top_down_view2 = cv2.warpPerspective(frame_2, H2, output_size)

    # Stack frames vertically
    stitched_frame = np.vstack((top_down_view1, top_down_view2))

    return stitched_frame