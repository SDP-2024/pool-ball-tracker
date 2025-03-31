import cv2

def get_top_down_view(frame : cv2.Mat, hom_matrix : cv2.Mat, output=(1200,600)) -> cv2.Mat:
    """
    Gets the top down view of the frame

    Args:
        frame (numpy.ndarray): The frame to get the top-down view for
        hom_matrix (numpy.ndarray): Homography matrix for transformation
        output (tuple, optional): The dimensions of the output frame. Defaults to (1200,600).

    Returns:
        numpy.ndarray: The warped frame
    """

    top_down_view : cv2.Mat = cv2.warpPerspective(frame, hom_matrix, output, borderMode=cv2.BORDER_REPLICATE)
    top_down_view : cv2.Mat = cv2.flip(top_down_view, 0)

    return top_down_view