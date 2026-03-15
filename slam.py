# slam.py

import os
import sys
import time
import traceback
import faulthandler
from datetime import datetime
from pathlib import Path

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

# 기존 프로젝트 모듈 임포트
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

def _safe_qsize(q):
    """
    플랫폼/Queue 구현에 따라 qsize()가 안 되는 경우가 있어 안전하게 감쌉니다.
    """
    try:
        return q.qsize()
    except Exception as e:
        return f"unavailable({type(e).__name__}: {e})"


def _run_backend_debug(backend):
    """
    기능 변경 없이, child process 내부 예외 traceback을 확실히 보이게 하기 위한 래퍼.
    """
    try:
        Log(
            f"[BackEndDebug] child started | "
            f"pid={os.getpid()} | ppid={os.getppid()}"
        )
        backend.run()
        Log(f"[BackEndDebug] child exited normally | pid={os.getpid()}")
    except Exception as e:
        Log(
            f"[BackEndDebug] child crashed | "
            f"pid={os.getpid()} | exc={repr(e)}"
        )
        traceback.print_exc()
        raise
 
class SLAM:
    def __init__(self, config, save_dir=None):
        """
        초기화 단계: 설정 로드 및 객체 생성만 수행 (프로세스 실행은 run에서 담당)
        """
        self.config = config
        self.save_dir = save_dir
        
        # 설정값 로드 및 변환
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        self.live_mode = self.config["Dataset"]["type"] in ["realsense", "zmq"]
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        self.use_wandb = self.config["Results"].get("use_wandb", False)
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        # SH 차수 설정 및 Gaussian 모델 초기화
        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0
        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        
        # 💡 [개선]: 하드코딩된 학습률을 config에서 가져오도록 수정 가능 (현재는 기존 유지)
        self.gaussians.init_lr(6.0)

        dataset_cfg = self.config.get("Dataset", {})
        dataset_path = (
            dataset_cfg.get("dataset_path")
            or dataset_cfg.get("data_path")
            or model_params.source_path
        )
        if not getattr(model_params, "source_path", "") and dataset_path:
            model_params.source_path = dataset_path
    
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )
        self.gaussians.training_setup(opt_params)
        
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        
        # 통신 큐 초기화
        self.frontend_queue = mp.Queue()
        self.backend_queue = mp.Queue()
        self.q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        self.q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        if save_dir is not None:
            self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        # 모듈 생성 및 설정
        self.frontend = FrontEnd(self.config)
        self.backend = BackEnd(self.config)
        
        self._init_frontend()
        self._init_backend()

        # GUI 파라미터 래퍼
        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=self.q_main2vis,
            q_vis2main=self.q_vis2main,
        )

    def _init_frontend(self):
        self.frontend.dataset = self.dataset
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = self.frontend_queue
        self.frontend.backend_queue = self.backend_queue
        self.frontend.q_main2vis = self.q_main2vis
        self.frontend.q_vis2main = self.q_vis2main
        self.frontend.set_hyperparams()

    def _init_backend(self):
        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = self.frontend_queue
        self.backend.backend_queue = self.backend_queue
        self.backend.live_mode = self.live_mode
        self.backend.set_hyperparams()

    def run(self):
        """
        실행 단계: 프로세스를 생성하고 트래킹/매핑 루프를 돌림
        """
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # 백그라운드 프로세스 시작
        backend_process = mp.Process(
            target=_run_backend_debug,
            args=(self.backend,),
            name="BackEndProcess",
        )
        if self.use_gui:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(5)
        
        backend_process.start()
        Log(
            f"[MainDebug] backend process started | "
            f"pid={backend_process.pid} | alive={backend_process.is_alive()} | "
            f"exitcode={backend_process.exitcode}"
        )
        
        # 메인 트래킹 루프 실행 (Frontend)
        Log("Frontend started")
        try:
            self.frontend.run()
            Log(
                f"[MainDebug] frontend.run() returned normally | "
                f"backend_alive={backend_process.is_alive()} | "
                f"backend_pid={backend_process.pid} | "
                f"backend_exitcode={backend_process.exitcode}"
            )
        except Exception as e:
            Log(
                f"[MainDebug] frontend.run() raised | "
                f"exc={repr(e)}"
            )
            Log(
                f"[MainDebug] backend status at failure | "
                f"alive={backend_process.is_alive()} | "
                f"pid={backend_process.pid} | "
                f"exitcode={backend_process.exitcode}"
            )
            Log(
                f"[MainDebug] queue state at failure | "
                f"frontend_queue={_safe_qsize(self.frontend_queue)} | "
                f"backend_queue={_safe_qsize(self.backend_queue)} | "
                f"gui_queue={_safe_qsize(self.q_main2vis) if self.use_gui else 'disabled'}"
            )
            traceback.print_exc()
            raise

        # 트래킹 종료 후 처리
        self.backend_queue.put(["pause"])
        end_event.record()
        torch.cuda.synchronize()
        
        # 성능 지표 계산
        n_frames = len(self.frontend.cameras)
        total_time = start_event.elapsed_time(end_event) * 0.001
        fps = n_frames / total_time
        Log(f"Total time: {total_time:.2f}s | FPS: {fps:.2f}", tag="Eval")

        # 평가 로직 수행
        if self.eval_rendering:
            self._perform_evaluation(fps)

        # 시스템 종료
        self.backend_queue.put(["stop"])
        backend_process.join()
        if self.use_gui:
            self.q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join()
        
        Log("SLAM System successfully joined and stopped.")

    def _perform_evaluation(self, fps):
        """렌더링 품질 및 트래킹 오차 평가"""
        self.gaussians = self.frontend.gaussians
        kf_indices = self.frontend.kf_indices

        # 1. Trajectory Error (ATE)
        ate_score = eval_ate(
            self.frontend.cameras, kf_indices, self.save_dir, 0,
            final=True, monocular=self.monocular
        )

        # 2. Before Color Refinement Rendering
        res_before = eval_rendering(
            self.frontend.cameras, self.gaussians, self.dataset, self.save_dir,
            self.pipeline_params, self.background, kf_indices=kf_indices, iteration="before_opt"
        )

        # 3. Backend Color Refinement 요청 및 동기화
        while not self.frontend_queue.empty(): self.frontend_queue.get()
        self.backend_queue.put(["color_refinement"])
        
        while True:
            if self.frontend_queue.empty():
                time.sleep(0.01)
                continue
            data = self.frontend_queue.get()
            if data[0] == "sync_backend" and self.frontend_queue.empty():
                self.gaussians = data[1]
                break

        # 4. After Color Refinement Rendering
        res_after = eval_rendering(
            self.frontend.cameras, self.gaussians, self.dataset, self.save_dir,
            self.pipeline_params, self.background, kf_indices=kf_indices, iteration="after_opt"
        )

        # WandB Table 기록
        columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
        metrics_table = wandb.Table(columns=columns)
        metrics_table.add_data("Before", res_before["mean_psnr"], res_before["mean_ssim"], res_before["mean_lpips"], ate_score, fps)
        metrics_table.add_data("After", res_after["mean_psnr"], res_after["mean_ssim"], res_after["mean_lpips"], ate_score, fps)
        
        wandb.log({"Metrics": metrics_table})
        save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)

@hydra.main(version_base=None, config_path="hydra_configs", config_name="config")
def main(cfg: DictConfig):
    # low-level crash가 나도 stderr에 더 잘 남도록 함
    faulthandler.enable()

    # 멀티프로세싱 스폰 방식 강제
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    Log(
        f"[MainDebug] process started | "
        f"pid={os.getpid()} | cwd={os.getcwd()}"
    )
    try:
        Log(f"[MainDebug] mp start method = {mp.get_start_method(allow_none=True)}")
    except Exception as e:
        Log(f"[MainDebug] failed to query start method: {repr(e)}")
    try:
        Log(f"[MainDebug] mp sharing strategy = {mp.get_sharing_strategy()}")
    except Exception as e:
        Log(f"[MainDebug] failed to query sharing strategy: {repr(e)}")
 
    config = OmegaConf.to_container(cfg, resolve=True)
    results_cfg = config["Results"]
    hydra_output_dir = HydraConfig.get().runtime.output_dir

    # 평가 모드 동적 설정
    if cfg.get("eval", False):
        Log("Overriding config for Evaluation Mode")
        config["Results"].update({
            "save_results": True,
            "use_gui": False,
            "eval_rendering": True,
            "use_wandb": True
        })
        results_cfg = config["Results"]


    save_dir = None
    dataset_cfg = config.get("Dataset", {})
    dataset_path = (
        dataset_cfg.get("dataset_path")
        or dataset_cfg.get("data_path")
        or ""
    )
    scene_name = (
        config.get("experiment", {}).get("name")
        or dataset_cfg.get("name")
        or (Path(dataset_path).name if dataset_path else dataset_cfg.get("type", "scene"))
    )

    if config["Results"]["save_results"]:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        base_save_dir = results_cfg["save_dir"]
        save_dir = os.path.join(base_save_dir, f"{scene_name}_{current_time}")
        results_cfg["save_dir"] = save_dir
        mkdir_p(save_dir)

        # Config 백업
        with open(os.path.join(save_dir, "config.yml"), "w") as f:
            yaml.dump(config, f)

    # WandB 초기화: save_results와 분리
    if results_cfg.get("use_wandb", False):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = results_cfg.get("wandb_name") or f"{scene_name}_{current_time}"

        # 방식 A:
        # wandb_dir가 비어 있으면 Hydra output dir 아래에 wandb 폴더 생성
        wandb_dir = results_cfg.get("wandb_dir")
        if not wandb_dir:
            wandb_dir = os.path.join(hydra_output_dir, "wandb")
        mkdir_p(wandb_dir)

        # artifact 다운로드 경로를 따로 두고 싶을 때만 설정
        wandb_artifact_dir = results_cfg.get("wandb_artifact_dir")
        if wandb_artifact_dir:
            mkdir_p(wandb_artifact_dir)
            os.environ["WANDB_ARTIFACT_DIR"] = wandb_artifact_dir

        Log(f"Hydra output dir: {hydra_output_dir}")
        Log(f"W&B local dir: {wandb_dir}")

        wandb.init(
            project=results_cfg.get("wandb_project", "MonoGS_Hydra"),
            entity=results_cfg.get("wandb_entity", None),
            name=run_name,
            tags=results_cfg.get("wandb_tags", None),
            notes=results_cfg.get("wandb_notes", None),

            config=config,
            dir=wandb_dir,
            mode=results_cfg.get("wandb_mode", "online"),

        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("tracking/*", step_metric="frame_idx")
        wandb.define_metric("mapping/*", step_metric="frame_idx")
        wandb.define_metric("eval/*", step_metric="frame_idx")
        wandb.define_metric("runtime/*")

    # SLAM 실행 (구조적 분리 완료)
    slam = SLAM(config, save_dir=save_dir)
    slam.run()

    if results_cfg.get("use_wandb", False):
        wandb.finish()
    Log("Done.")

if __name__ == "__main__":
    main()