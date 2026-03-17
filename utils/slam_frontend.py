# utils/slam_frontend.py

import time

import numpy as np
import torch
import torch.multiprocessing as mp

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth


# 이 코드는 SLAM 시스템의 눈과 발 역할을 하는 Frontend (트래킹 및 키프레임 관리) 부분을 담당
# 카메라가 실시간으로 어디로 이동하고 있는지(Pose Tracking) 계산하고, 
# 언제 맵을 업데이트할지(Keyframe Selection) 결정하는 아주 핵심적인 모듈입니다.

# 💡 [리뷰 포인트 - 프로세스 분리]: 
# FrontEnd 클래스 자체가 mp.Process를 상속받아 독립된 프로세스로 동작하도록 설계되었습니다.
# 카메라 추적(Tracking)이 렌더링/최적화(Mapping) 때문에 끊기지 않도록 하는 훌륭한 패턴입니다.
class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None

        # 프로세스 간 통신(IPC)을 위한 큐
        self.frontend_queue = None  # Backend -> Frontend (맵 업데이트 정보 등)
        self.backend_queue = None   # Frontend -> Backend (새 키프레임, 트래킹 정보 등)
        self.q_main2vis = None      # Frontend -> GUI
        self.q_vis2main = None      # GUI -> Frontend

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]     # 단일 카메라(RGB) 모드인지 여부

        self.use_external_pose = bool(config["Dataset"].get("use_external_pose", False))
        self.optimize_tracking_pose = bool(
            config["Training"].get("optimize_tracking_pose", not self.use_external_pose)
        )

        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []        # 현재 최적화에 사용할 키프레임들의 슬라이딩 윈도우

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False

    def set_hyperparams(self):
        # ... (설정값 불러오기 생략) ...
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        # 💡 [리뷰 포인트 - 단안(Monocular) 깊이 추정]: 
        # RGB-D 카메라가 아닌 단일 RGB 카메라일 경우, 깊이(Depth) 센서 값이 없으므로 
        # 처음에는 임의의 깊이값(2.0 + 노이즈)으로 초기화하거나, 기존 가우시안 렌더링 결과의 
        # 중앙값(Median depth)을 활용해 초기 깊이를 추정(Guessing)하는 매우 흥미로운 로직입니다.
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:
                # 초기화 단계: 임의의 뎁스 생성 (2.0 근처)
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                # 트래킹 중: 렌더링된 뎁스의 중앙값과 표준편차를 기반으로 아웃라이어를 걸러냄
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False

                # ... (역심도(Inverse Depth) 또는 일반 심도 기반 아웃라이어 제거 로직) ...
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                # 유효하지 않은 RGB 픽셀은 뎁스 무시
                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            return initial_depth.cpu().numpy()[0]
        # use the observed depth
        # RGB-D 모드: 실제 센서에서 관측된 뎁스를 그대로 사용
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()

    def initialize(self, cur_frame_idx, viewpoint):
        # 첫 번째 프레임을 시스템에 등록하고 초기화하는 함수
        self.initialized = self.use_external_pose or (not self.monocular)
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        # 큐 초기화 (이전 잔여물 제거)
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        # 💡 [리뷰 포인트]: 첫 프레임은 원점(또는 Ground Truth)으로 고정하여 좌표계의 기준을 잡습니다.
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    def tracking(self, cur_frame_idx, viewpoint):
        # 외부 odom pose를 그대로 사용하는 모드:
        # frontend에서는 pose estimation을 하지 않고, 들어온 pose를 현재 프레임 pose로 사용
        if self.use_external_pose and not self.optimize_tracking_pose:
            viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            self.median_depth = get_median_depth(
                render_pkg["depth"], render_pkg["opacity"]
            )
            return render_pkg

        # 💡 [리뷰 포인트 - Photometric Tracking]: 
        # 카메라의 현재 위치를 찾기 위해, 직전 프레임 위치에서 시작하여(prev.R, prev.T)
        # 렌더링 이미지와 실제 들어온 이미지를 비교(loss_tracking)하며 카메라 포즈를 미세 조정합니다.
        if self.use_external_pose:
            # external pose가 들어오지만 소량 refinement는 허용하는 모드
            viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
        else:
            prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
            viewpoint.update_RT(prev.R, prev.T)

        # 최적화할 대상: 카메라의 회전(Rot), 이동(Trans), 그리고 노출값(Exposure a, b)
        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )

        pose_optimizer = torch.optim.Adam(opt_params)

        # tracking_itr_num (예: 100회) 만큼 반복하면서 포즈를 최적화
        for tracking_itr in range(self.tracking_itr_num):
            # 현재 추정된 위치에서 3D 씬(가우시안)을 렌더링
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            # 실제 카메라 이미지와 렌더링 이미지 간의 차이(Loss) 계산
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)
            
            # 10번 반복마다 GUI로 현재 상태 전송하여 시각화
            if tracking_itr % 10 == 0:
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
            if converged:
                break   # 로스가 수렴하면 조기 종료

        self.median_depth = get_median_depth(depth, opacity)
        return render_pkg

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        # 💡 [리뷰 포인트 - 키프레임 등록 조건]: 
        # 카메라가 직전 키프레임 대비 얼마나 이동했는지(dist), 
        # 그리고 바라보는 뷰가 얼마나 겹치는지(point_ratio_2)를 계산하여 새 키프레임 여부를 결정합니다.
        # 시각적 SLAM에서 매우 교과서적이고 안정적인 접근법입니다.
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        # 로컬 매핑을 위한 슬라이딩 윈도우 관리 함수
        # 윈도우 사이즈가 초과되면, 시야가 너무 안 겹치거나 공간상 가장 덜 중요한(중복되는) 키프레임을 버림
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    # 💡 [리뷰 포인트 - 프로세스 간 통신 (요청)]:
    # Frontend는 실시간 Tracking에 집중해야 하므로, 연산이 무거운 Mapping 작업이나 
    # Keyframe 추가 작업은 Queue를 통해 Backend 프로세스로 넘깁니다.
    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        # 새로운 키프레임 생성을 백엔드에 요청 (이미지, 카메라 포즈, 현재 윈도우, 초기 뎁스 전달)
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        # 현재 프레임에 대한 매핑(최적화) 요청
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        # 시스템의 첫 프레임 초기화 요청
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    # 💡 [리뷰 포인트 - 프로세스 간 통신 (동기화)]:
    # Backend에서 최적화(Mapping)가 끝난 최신 가우시안 맵과 보정된 카메라 포즈를 
    # Frontend로 가져와(Sync) 다음 Tracking이 더 정확해지도록 업데이트합니다.
    def sync_backend(self, data):
        self.gaussians = data[1]            # 백엔드에서 최적화된 최신 3D 가우시안 맵
        occ_aware_visibility = data[2]      # 프레임별 가시성(Visibility) 정보 업데이트
        keyframes = data[3]                 # 백엔드에서 보정된 키프레임 포즈(R, T)들
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            # 백엔드에서 Bundle Adjustment 등으로 최적화된 카메라 위치로 Frontend 카메라 갱신
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    # 💡 [리뷰 포인트 - 메모리 관리]:
    # 처리 완료된 프레임 이미지나 텐서들을 정리하여 GPU VRAM이 터지지(OOM) 않도록 방지합니다.
    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()# 10프레임마다 강제로 캐시를 비워 메모리 단편화 해소

    # 💡 [리뷰 포인트 - 프론트엔드 메인 파이프라인]:
    # FrontEnd 프로세스가 시작되면 실행되는 무한 루프입니다.
    # [데이터 로드 -> 트래킹(카메라 위치 찾기) -> 키프레임 판별 -> 백엔드와 동기화] 의 사이클을 돕니다.
    def run(self):
        cur_frame_idx = 0
        # 카메라 내부 파라미터(Intrinsics)를 바탕으로 투영 매트릭스 생성
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)

        # 타이밍 측정을 위한 CUDA Event 객체
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            # 1. GUI 컨트롤 처리 (일시정지/재개)
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            # 2. 메인 Tracking 루프 (Backend에서 온 메시지가 없을 때 진행)
            if self.frontend_queue.empty():
                tic.record()
                # 데이터셋을 끝까지 다 돌았을 경우: 최종 평가(ATE) 및 맵 저장 후 종료
                if cur_frame_idx >= len(self.dataset):
                    if self.save_results:
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break
                # Backend가 초기화 중이거나, 싱글스레드 모드에서 키프레임 처리 중이면 대기 (병목 현상 방지)
                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue
                
                # 3. 새로운 프레임 데이터 로드
                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)

                self.cameras[cur_frame_idx] = viewpoint

                # 4. 초기화 모드일 경우 (시스템 첫 시작 혹은 Tracking 실패로 인한 Reset)
                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue
                
                # 슬라이딩 윈도우가 다 차면 초기화 완료로 간주
                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Tracking
                # 5. 현재 프레임 Tracking 수행 (가장 중요한 부분)
                render_pkg = self.tracking(cur_frame_idx, viewpoint)

                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

                # 6. GUI 업데이트: 트래킹이 완료된 현재 뷰와 키프레임 윈도우 정보를 시각화 툴로 전송
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )
                
                # Backend에 넘겨놓은 키프레임 작업이 밀려있으면 일단 대기
                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue
                
                # 7. 키프레임 등록 여부 판단 로직
                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()

                # 이동 거리와 시야 겹침 정도를 계산하여 새 키프레임이 필요한지 결정
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                # 윈도우가 아직 안 찼을 때는 강제성을 부여하여 키프레임을 좀 더 적극적으로 생성
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                if self.single_thread:
                    create_kf = check_time and create_kf

                # 8. 새 키프레임으로 채택된 경우의 처리
                if create_kf:
                    # 슬라이딩 윈도우에 추가 (필요 시 가장 안 중요한 프레임 제거)
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    # 단안 카메라인데 오버랩이 너무 적어 윈도우에서 프레임이 제거되었다면 리셋 고려
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    # 새로운 키프레임 생성 및 백엔드에 매핑 요청
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )
                else:
                    # 키프레임이 아닌 일반 프레임은 Tracking 후 메모리에서 해제
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += 1

                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()

                # 키프레임 추가 시 시스템 과부하를 막기 위해 일시적인 속도 조절(스로틀링) 적용 (3fps 제한)
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))

            # 9. Backend에서 보낸 데이터(메시지)가 큐에 있을 경우 수신하여 처리
            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    # 일반 동기화 요청
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    # 키프레임 처리가 완료되어 백엔드에서 맵/포즈 갱신 데이터 전송됨
                    self.sync_backend(data)
                    self.requested_keyframe -= 1        # 대기 중인 키프레임 수 감소

                elif data[0] == "init":
                    # 초기화 완료 데이터 전송됨
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    # 시스템 종료 메시지
                    Log("Frontend Stopped.")
                    break
