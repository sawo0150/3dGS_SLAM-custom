# utils/dataset_zmq.py

import queue
import threading
import time
import zmq
import numpy as np
import torch

from utils.dataset import BaseDataset
from gaussian_splatting.utils.graphics_utils import focal2fov


class ZmqDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)

        ds_cfg = config["Dataset"]
        calib = ds_cfg["Calibration"]

        # MonoGS frontend가 쓰는 intrinsics
        self.fx = float(calib["fx"])
        self.fy = float(calib["fy"])
        self.cx = float(calib["cx"])
        self.cy = float(calib["cy"])
        self.width = int(calib["width"])
        self.height = int(calib["height"])
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        self.has_depth = False
        self.num_imgs = 999999  # live stream처럼 동작
        self.closed = False
        self.frames_emitted = 0

        self.frame_queue = queue.Queue(maxsize=int(ds_cfg.get("queue_size", 8)))
        # MonoGS 내부에는 world-to-camera(T_cw)를 넣는 것이 안전함
        self.last_pose_w2c = np.eye(4, dtype=np.float32)

        self.ctx = zmq.Context.instance()

        self.bundle_sock = self.ctx.socket(zmq.SUB)
        self.bundle_sock.connect(ds_cfg["bundle_endpoint"])
        self.bundle_sock.setsockopt(zmq.SUBSCRIBE, b"")

        self.ctrl_sock = None
        if "control_endpoint" in ds_cfg:
            self.ctrl_sock = self.ctx.socket(zmq.SUB)
            self.ctrl_sock.connect(ds_cfg["control_endpoint"])
            self.ctrl_sock.setsockopt(zmq.SUBSCRIBE, b"")

        self.poller = zmq.Poller()
        self.poller.register(self.bundle_sock, zmq.POLLIN)

        if self.ctrl_sock is not None:
            self.poller.register(self.ctrl_sock, zmq.POLLIN)

        self.rx_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.rx_thread.start()

    def _safe_put(self, item):
        # backlog가 쌓이면 가장 오래된 frame drop
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put(item)
        self.frames_emitted += 1

    def _emit_frame(self, frame_msg, pose_w2c_np):
        image = frame_msg.image
        if image is None:
            return

        pose_w2c_np = np.asarray(pose_w2c_np, dtype=np.float32)
        self._safe_put((frame_msg.seq, frame_msg.device_time_ns, image, pose_w2c_np))

    def _recv_loop(self):
        while True:
            events = dict(self.poller.poll(timeout=5))

            if self.ctrl_sock is not None and self.ctrl_sock in events:
                msg = self.ctrl_sock.recv_pyobj()
                if type(msg).__name__ == "ControlMsg" and msg.command in ("EOS", "STOP"):
                    self.closed = True
                    self.num_imgs = self.frames_emitted
                    break

            if self.bundle_sock in events:
                msg = self.bundle_sock.recv_pyobj()
                msg_type = type(msg).__name__

                if msg_type == "ControlMsg" and msg.command in ("EOS", "STOP"):
                    self.closed = True
                    self.num_imgs = self.frames_emitted
                    break

                if msg_type == "BundleMsg":
                    frame_msg = msg.frame
                    odom_msg = msg.odom

                    if frame_msg is None:
                        continue

                    # 기본값: 마지막 pose 재사용
                    pose_w2c = self.last_pose_w2c

                    # Aria/MPS 쪽 T_world_device는 보통 camera-to-world(T_wc) 해석이 자연스러움.
                    # MonoGS dataset pose는 기존 parser들 기준으로 world-to-camera(T_cw) 쪽이므로 inverse해서 넣음.
                    if odom_msg is not None and getattr(odom_msg, "T_world_device", None) is not None:
                        T_wc = np.asarray(odom_msg.T_world_device, dtype=np.float32)
                        pose_w2c = np.linalg.inv(T_wc)
                        self.last_pose_w2c = pose_w2c

                    self._emit_frame(frame_msg, pose_w2c)

        self.num_imgs = self.frames_emitted

    def __len__(self):
        return self.frames_emitted if self.closed else self.num_imgs

    def __getitem__(self, idx):
        # Frontend가 idx를 넘기지만, live stream에서는 사실상 next-frame consumer처럼 동작
        if self.closed and self.frame_queue.empty() and idx >= self.frames_emitted:
            raise IndexError("Stream ended")

        seq, ts_ns, image, pose_np = self.frame_queue.get()

        # publisher 쪽 image가 RGB np.uint8라고 가정
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)

        image_t = (
            torch.from_numpy(image.astype(np.float32) / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )

        pose_t = torch.from_numpy(pose_np).to(device=self.device, dtype=self.dtype)

        depth = None
        return image_t, depth, pose_t