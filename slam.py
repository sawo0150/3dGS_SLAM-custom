import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

import wandb
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd


class SLAM:
    def __init__(self, config, save_dir=None):
        # 💡 [리뷰 포인트 - 아키텍처]: 현재 클래스의 __init__ 메서드 안에 초기화 로직뿐만 아니라
        # 실제 시스템을 실행하고(start), 평가하고(eval), 종료하는(join) 모든 생명주기 로직이 들어있습니다.
        # 객체 지향 설계 관점에서는 __init__에서는 큐와 프로세스 초기화만 진행하고, 
        # 실제 실행 흐름은 아래에 비어있는 run() 메서드로 분리하는 것이 유지보수에 훨씬 좋습니다.

        # CUDA 비동기 연산의 정확한 시간 측정을 위해 Event 객체 생성
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        self.config = config
        self.save_dir = save_dir
        
        # munchify를 사용해 딕셔너리 키를 객체 속성처럼 접근할 수 있게 변환 (예: dict["key"] -> dict.key)
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        # 설정값 로드 (라이브 모드 여부, 단일 카메라 여부, 구면 조화 함수(SH) 사용 여부 등)
        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        # SH 차수 설정 (사용하면 3, 아니면 0)
        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        # 핵심 3D 표현인 Gaussian 모델 초기화
        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)     # 💡 [리뷰 포인트]: 학습률 6.0은 하드코딩되어 있는데, config 파일로 빼는 것이 유연성에 좋습니다.
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )

        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # 💡 [리뷰 포인트 - 병렬 처리]: Frontend(카메라 트래킹)와 Backend(맵/가우시안 최적화) 간의 
        # 데이터를 주고받기 위한 멀티프로세싱 큐(Queue) 생성. 매우 적절한 설계입니다.
        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()

        # GUI 사용 여부에 따라 실제 큐 또는 Dummy 큐(FakeQueue)를 할당하여 분기문 최소화 (좋은 패턴)
        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        # 트래킹과 맵핑을 담당할 객체 생성
        self.frontend = FrontEnd(self.config)
        self.backend = BackEnd(self.config)
        
        # Frontend 초기화 및 통신 큐 연결
        self.frontend.dataset = self.dataset
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()

        # Backend 초기화 및 통신 큐 연결
        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0   # 💡 [리뷰 포인트]: extent 값 6.0도 씬(scene)의 크기에 따라 달라지므로 하드코딩을 피하는 것이 좋습니다.
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode

        self.backend.set_hyperparams()

        # GUI 파라미터 래퍼 객체 생성
        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )

        # 💡 [리뷰 포인트 - 프로세스 실행]: Backend를 별도의 백그라운드 프로세스로 실행합니다.
        backend_process = mp.Process(target=self.backend.run)
        if self.use_gui:
            # GUI 프로세스도 병렬로 실행
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(5)       # GUI가 초기화될 시간을 벌어줌 (조금 투박하지만 간단한 방법)

        # Backend 프로세스 시작
        backend_process.start()
        # 💡 [리뷰 포인트 - 블로킹 호출]: Frontend는 메인 프로세스에서 그대로 실행됩니다.
        # self.frontend.run()이 끝날 때까지 (데이터셋을 다 돌 때까지) 메인 스레드는 여기서 블로킹됩니다.
        self.frontend.run()

        # 트래킹이 끝나면 Backend 최적화를 일시정지 시킴
        backend_queue.put(["pause"])

        # 전체 실행 시간 기록 완료
        end.record()
        torch.cuda.synchronize()# CUDA 연산이 끝날 때까지 대기
        
        # FPS 계산
        # empty the frontend queue
        N_frames = len(self.frontend.cameras)
        FPS = N_frames / (start.elapsed_time(end) * 0.001)
        Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")

        # ==========================================================
        # 아래부터는 평가(Evaluation) 및 렌더링 품질 측정 로직입니다.
        # ==========================================================
        if self.eval_rendering:
            self.gaussians = self.frontend.gaussians
            kf_indices = self.frontend.kf_indices

            # 1. ATE (Absolute Trajectory Error) - 카메라 포즈 추정 오차 측정
            ATE = eval_ate(
                self.frontend.cameras,
                self.frontend.kf_indices,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
            )

            # 2. 색상 최적화 전 렌더링 품질 측정 (PSNR, SSIM, LPIPS)
            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="before_opt",
            )
            columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
            metrics_table = wandb.Table(columns=columns)
            metrics_table.add_data(
                "Before",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )

            # 💡 [리뷰 포인트 - 큐 비우기]: 통신용 큐를 재활용하기 위해 비우는 과정.
            # 데드락(Deadlock)을 방지하기 위한 중요한 처리가 들어갔습니다.
            # re-used the frontend queue to retrive the gaussians from the backend.
            while not frontend_queue.empty():
                frontend_queue.get()

            # Backend에 "color_refinement"(색상 추가 미세조정) 명령을 내림
            backend_queue.put(["color_refinement"])
            while True:
                if frontend_queue.empty():
                    time.sleep(0.01)
                    continue
                data = frontend_queue.get()
                if data[0] == "sync_backend" and frontend_queue.empty():
                    gaussians = data[1]
                    self.gaussians = gaussians
                    break

            # 3. 색상 최적화 후 렌더링 품질 재측정
            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="after_opt",
            )
            metrics_table.add_data(
                "After",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )
            # 최종 결과 WandB 로깅 및 파일 저장
            wandb.log({"Metrics": metrics_table})
            save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)

        # 시스템 종료 로직: Backend 프로세스 정상 종료 대기
        backend_queue.put(["stop"])
        backend_process.join()
        Log("Backend stopped and joined the main thread")
        if self.use_gui:
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join()
            Log("GUI Stopped and joined the main thread")

    def run(self):
        pass


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args(sys.argv[1:])

    # 💡 [리뷰 포인트]: PyTorch + 멀티프로세싱을 사용할 때 CUDA 컨텍스트 공유 문제를 
    # 해결하기 위해 'spawn' 방식을 강제하는 매우 중요한 설정입니다.
    mp.set_start_method("spawn")

    # 설정 파일 파싱 (YAML)
    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None

    # 평가 모드일 경우 강제로 덮어쓰는 설정들
    if args.eval:
        Log("Running MonoGS in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=True")
        config["Results"]["use_wandb"] = True

    # 결과 저장을 위한 디렉토리 구조 생성 (날짜/시간 기반 고유 폴더)
    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = config["Dataset"]["dataset_path"].split("/")
        save_dir = os.path.join(
            config["Results"]["save_dir"], path[-3] + "_" + path[-2], current_datetime
        )
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)

        # 재현성을 위해 현재 사용된 config를 결과 폴더에 백업
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)

        # WandB (Weights & Biases) 초기화: 실험 추적용
        run = wandb.init(
            project="MonoGS",
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")

    # SLAM 객체 생성 (현재 구조상 생성과 동시에 전체 파이프라인이 끝까지 실행됨)
    slam = SLAM(config, save_dir=save_dir)

    # 아무것도 하지 않는 껍데기 메서드 호출
    slam.run()

    # WandB 세션 종료
    wandb.finish()

    # All done
    Log("Done.")
