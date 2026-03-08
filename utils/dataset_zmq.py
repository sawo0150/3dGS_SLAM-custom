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
        self.pending_frames = {}   # seq -> (recv_time, frame_msg)
        self.pending_odom = {}     # seq -> 4x4 pose
        self.last_pose = np.eye(4, dtype=np.float32)
        self.odom_wait_ms = float(ds_cfg.get("odom_wait_ms", 10.0))

        self.ctx = zmq.Context.instance()

        self.frame_sock = self.ctx.socket(zmq.SUB)
        self.frame_sock.connect(ds_cfg["frame_endpoint"])
        self.frame_sock.setsockopt(zmq.SUBSCRIBE, b"")

        self.odom_sock = self.ctx.socket(zmq.SUB)
        self.odom_sock.connect(ds_cfg["odom_endpoint"])
        self.odom_sock.setsockopt(zmq.SUBSCRIBE, b"")

        self.ctrl_sock = None
        if "control_endpoint" in ds_cfg:
            self.ctrl_sock = self.ctx.socket(zmq.SUB)
            self.ctrl_sock.connect(ds_cfg["control_endpoint"])
            self.ctrl_sock.setsockopt(zmq.SUBSCRIBE, b"")

        self.poller = zmq.Poller()
        self.poller.register(self.frame_sock, zmq.POLLIN)
        self.poller.register(self.odom_sock, zmq.POLLIN)
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

    def _emit_frame(self, frame_msg, pose_np):
        image = frame_msg.image
        if image is None:
            return

        pose_np = np.asarray(pose_np, dtype=np.float32)
        self._safe_put((frame_msg.seq, frame_msg.device_time_ns, image, pose_np))

    def _flush_old_pending_frames(self):
        now = time.monotonic()
        old_seqs = []
        for seq, (t0, frame_msg) in self.pending_frames.items():
            age_ms = (now - t0) * 1000.0
            if age_ms >= self.odom_wait_ms:
                # matching odom이 너무 늦으면 latest pose 또는 identity로 진행
                self._emit_frame(frame_msg, self.last_pose)
                old_seqs.append(seq)

        for seq in old_seqs:
            self.pending_frames.pop(seq, None)

    def _recv_loop(self):
        while True:
            events = dict(self.poller.poll(timeout=5))

            if self.ctrl_sock is not None and self.ctrl_sock in events:
                msg = self.ctrl_sock.recv_pyobj()
                if type(msg).__name__ == "ControlMsg" and msg.command in ("EOS", "STOP"):
                    self.closed = True
                    self.num_imgs = self.frames_emitted
                    break

            if self.frame_sock in events:
                msg = self.frame_sock.recv_pyobj()
                msg_type = type(msg).__name__

                if msg_type == "ControlMsg" and msg.command in ("EOS", "STOP"):
                    self.closed = True
                    self.num_imgs = self.frames_emitted
                    break

                if msg_type == "FrameMsg":
                    if msg.seq in self.pending_odom:
                        pose_np = self.pending_odom.pop(msg.seq)
                        self.last_pose = pose_np
                        self._emit_frame(msg, pose_np)
                    else:
                        self.pending_frames[msg.seq] = (time.monotonic(), msg)

            if self.odom_sock in events:
                msg = self.odom_sock.recv_pyobj()
                msg_type = type(msg).__name__

                if msg_type == "ControlMsg" and msg.command in ("EOS", "STOP"):
                    self.closed = True
                    self.num_imgs = self.frames_emitted
                    break

                if msg_type == "OdomMsg":
                    pose_np = np.asarray(msg.T_world_device, dtype=np.float32)
                    self.last_pose = pose_np

                    if msg.seq in self.pending_frames:
                        _, frame_msg = self.pending_frames.pop(msg.seq)
                        self._emit_frame(frame_msg, pose_np)
                    else:
                        self.pending_odom[msg.seq] = pose_np

            self._flush_old_pending_frames()

        # 종료 직전 남은 frame은 마지막 pose로라도 배출
        for _, frame_msg in list(self.pending_frames.values()):
            self._emit_frame(frame_msg, self.last_pose)
        self.pending_frames.clear()

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