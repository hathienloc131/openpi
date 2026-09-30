"""Dummy client for the BDS (VR_H5D) pi0.5 policy server.

Start the server first (see scripts/serve_bds.sh), then:

    # Random inputs (just checks the server round trip and output shape):
    python examples/bds/client.py --host <server_ip>

    # A real frame from the dataset, compared against the recorded actions:
    python examples/bds/client.py --host <server_ip> \
        --dataset-root /mnt/data/sftp/data/vla/data_sim_ac/20260929_VR_H5D_VFE_sim_pick_success_openpi \
        --episode 0 --frame 50

Observation format expected by the server (see src/openpi/policies/bds_policy.py):
    "observation.images.color.head":    uint8 (H, W, 3) head camera image
    "observation.images.color.outside": uint8 (H, W, 3) outside camera image
    "observation.state": float32 (16,) = 14 arm joint positions + 2 vacuums, in this order
        left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow_pitch,
        left_wrist_yaw, left_wrist_roll, left_wrist_pitch,
        right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow_pitch,
        right_wrist_yaw, right_wrist_roll, right_wrist_pitch, left_vacuum, right_vacuum
        (the full 96-dim robot state is also accepted; it is sliced on the server)
    "prompt": str

Response: {"actions": float32 (action_horizon, 16)} -- absolute joint position targets + vacuum commands,
in the same order as the state.
"""

import dataclasses
import logging
import time

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import tyro

STATE_INDICES = [*range(14, 28), 30, 31]
DEFAULT_PROMPT = (
    "Use the vacuum attached to your left hand to suck up the large curved metal frame and lift it from the table."
)


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    prompt: str = DEFAULT_PROMPT
    # Number of requests to send (to measure latency).
    num_requests: int = 5
    # If set, use a real frame from this local LeRobot dataset instead of random inputs.
    dataset_root: str | None = None
    episode: int = 0
    frame: int = 0


def random_observation(prompt: str) -> dict:
    return {
        "observation.images.color.head": np.random.randint(256, size=(480, 720, 3), dtype=np.uint8),
        "observation.images.color.outside": np.random.randint(256, size=(480, 720, 3), dtype=np.uint8),
        "observation.state": np.random.uniform(-0.5, 0.5, size=(16,)).astype(np.float32),
        "prompt": prompt,
    }


def dataset_observation(root: str, episode: int, frame: int) -> tuple[dict, np.ndarray]:
    """Loads one frame (images, state, prompt) and the recorded future actions from a local LeRobot dataset."""
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    ds = lerobot_dataset.LeRobotDataset("bds/local", root=root, episodes=[episode])
    item = ds[frame]

    def to_hwc_uint8(img) -> np.ndarray:
        img = np.asarray(img)
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        return (255 * img).astype(np.uint8) if np.issubdtype(img.dtype, np.floating) else img

    obs = {
        "observation.images.color.head": to_hwc_uint8(item["observation.images.color.head"]),
        "observation.images.color.outside": to_hwc_uint8(item["observation.images.color.outside"]),
        "observation.state": np.asarray(item["observation.state"])[STATE_INDICES].astype(np.float32),
        "prompt": item["task"],
    }
    # Recorded actions for the following frames of this episode (for comparison).
    actions = np.stack([np.asarray(ds.hf_dataset[i]["action"]) for i in range(frame, min(frame + 64, len(ds)))])
    return obs, actions


def main(args: Args) -> None:
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    logging.info("Server metadata: %s", client.get_server_metadata())

    gt_actions = None
    if args.dataset_root:
        obs, gt_actions = dataset_observation(args.dataset_root, args.episode, args.frame)
    else:
        obs = random_observation(args.prompt)

    # Resize on the client to cut bandwidth. The server applies the same resize-with-pad to 224x224,
    # so this gives the same model input as sending full-resolution images.
    for key in ("observation.images.color.head", "observation.images.color.outside"):
        obs[key] = image_tools.convert_to_uint8(image_tools.resize_with_pad(obs[key], 224, 224))

    for i in range(args.num_requests):
        start = time.monotonic()
        result = client.infer(obs)
        latency_ms = 1000 * (time.monotonic() - start)
        actions = np.asarray(result["actions"])
        logging.info("request %d: actions %s, latency %.1f ms", i, actions.shape, latency_ms)

    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print("state        :", obs["observation.state"])
    print("pred action 0:", actions[0])
    print("pred action -1:", actions[-1])
    if gt_actions is not None:
        n = min(len(actions), len(gt_actions))
        err = np.abs(actions[:n] - gt_actions[:n])
        print("gt   action 0:", gt_actions[0])
        print(f"mean |pred - gt| over {n} steps, per dim:", err.mean(0))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
