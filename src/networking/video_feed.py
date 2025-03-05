import cv2
import asyncio
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaStreamTrack
from aiohttp import web
from flask import app
from src.processing.frame_processing import get_top_down_view

class VideoTrack(MediaStreamTrack):
    def __init__(self, config=None, table_pts_cam1=None, table_pts_cam2=None):
        super().__init__()
        self.config = config
        self.camera_1 = cv2.VideoCapture(self.config.get("camera_port_1", 0))
        self.camera_2 = None
        self.table_pts_cam1 = table_pts_cam1
        self.table_pts_cam2 = table_pts_cam2
        # if self.config["camera_port_2"] != -1:
        #     self.camera_2 = cv2.VideoCapture(self.config.get("camera_port_2", 0))

    async def recv(self):
        ret_1, frame_1 = self.camera_1.read()
        if self.camera_2 is not None:
            ret_2, frame_2 = self.camera_2.read()
            if not ret_2:
                return None
            
        if not ret_1:
            return None
        
        stitched_frame = frame_1 if frame_2 is None else get_top_down_view(frame_1, frame_2, self.table_pts_cam1, self.table_pts_cam2)
        return cv2.cvtColor(stitched_frame, cv2.COLOR_BGR2RGB)
    

async def offer(request):
    params = await request.json()
    pc = RTCPeerConnection()
    track = VideoTrack()
    pc.addTrack(track)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=params["sdp"], type=params["type"]))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})


def start_stream():
    app = web.Application()
    app.router.add_post("/offer", offer)
    web.run_app(app, host="0.0.0.0", port=8080)