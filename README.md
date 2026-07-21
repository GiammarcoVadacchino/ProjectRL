# Autonomous Racing with PPO

The agent uses **Proximal Policy Optimization (PPO)** and controls two continuous actions:

* throttle/brake;
* steering angle.

The program exposes two main commands:

* `train`: trains a model from scratch or continues training from an existing model;
* `watch`: evaluates and visualizes an already trained model.

> Additional training does not have a separate command. It is performed using `train` together with `--model-path`.

---

## Main files

```text
main.py           Command-line interface
training.py       Training, checkpoints, logging, and evaluation
f1_env.py         Gymnasium environment and rendering
car_dynamics.py   Simplified dynamics and observation
reward.py         Reward function
```

Tracks are loaded from CSV files containing the centerline and track widths.

---

## Help

```bash
python main.py --help
python main.py train --help
python main.py watch --help
```

---

# `train` command

`train` can start a new training run or continue from an existing model.

## Training from scratch

When training from scratch, at least one architecture must be specified using `--archs`.

```bash
python main.py train \
  --tracks racetrack-database/tracks/Monza.csv \
           racetrack-database/tracks/Melbourne.csv \
           racetrack-database/tracks/Silverstone.csv \
  --archs 64,64,32,16 \
  --timesteps 15000000 \
  --seed 42 \
  --output-dir runs_training
```

It is possible to train multiple architectures sequentially:

```bash
python main.py train \
  --tracks racetrack-database/tracks/Monza.csv \
           racetrack-database/tracks/Melbourne.csv \
           racetrack-database/tracks/Silverstone.csv \
  --archs 64,64,32,16 128,128,64,32 256,256,128,64,32
```

Each architecture generates an independent model.

## Continuing an existing model

To resume training, use `--model-path`.

```bash
python main.py train \
  --tracks racetrack-database/tracks/Monza.csv \
  --model-path runs_training/modello_pretrained.zip \
  --timesteps 3000000 \
  --learning-rate 1e-4 \
  --no-random-start \
  --checkpoint-freq 500000 \
  --seed 42 \
  --output-dir runs_monza_fixed
```

In this mode:

* the architecture is loaded from the model;
* `--archs` must not be used;
* the specified timesteps are additional;
* the timestep counter is not reset.

## Multi-track continuation

A model can continue training on multiple tracks:

```bash
python main.py train \
  --tracks racetrack-database/tracks/Monza.csv \
           racetrack-database/tracks/Melbourne.csv \
           racetrack-database/tracks/Silverstone.csv \
  --model-path runs_training/modello_pretrained.zip \
  --timesteps 3000000 \
  --learning-rate 1e-4 \
  --no-random-start \
  --checkpoint-freq 500000 \
  --seed 42 \
  --output-dir runs_multitrack_fixed
```

One environment is created for each track, and PPO collects experience from all tracks.

## Fixed or random starting position

```bash
--random-start
```

Starts each episode from a random position.

```bash
--no-random-start
```

Starts each episode from the standard initial position.

If the option is not specified:

* training from scratch: random starting position;
* training with `--model-path`: fixed starting position.

## Main `train` options

| Option              | Description                                       |
| ------------------- | ------------------------------------------------- |
| `--tracks`          | One or more track CSV files                       |
| `--archs`           | Architectures used when training from scratch     |
| `--model-path`      | PPO model to load                                 |
| `--timesteps`       | Run timesteps or additional timesteps             |
| `--learning-rate`   | PPO learning rate                                 |
| `--n-steps`         | Steps collected per environment before the update |
| `--batch-size`      | Mini-batch size                                   |
| `--ent-coef`        | Entropy coefficient                               |
| `--seed`            | Run seed                                          |
| `--random-start`    | Enables random starting positions                 |
| `--no-random-start` | Uses the fixed starting position                  |
| `--checkpoint-freq` | Checkpoint frequency; `0` disables them           |
| `--output-dir`      | Results directory                                 |
| `--run-name`        | Custom run name                                   |
| `--max-steps`       | Maximum number of steps per episode               |

---

# `watch` command

`watch` loads a model, evaluates it deterministically, and displays the rendering.

```bash
python main.py watch \
  --model-path runs_training/modello.zip \
  --track racetrack-database/tracks/Monza.csv \
  --episodes 1
```

During evaluation:

* the weights are not updated;
* the actions are deterministic;
* the starting position is fixed;
* the track boundaries, raceline, vehicle position, and vehicle orientation are displayed.

The raceline must be located in:

```text
racetrack-database/racelines/
```

and must have the same name as the track file.

## Main `watch` options

| Option           | Description                          |
| ---------------- | ------------------------------------ |
| `--model-path`   | PPO model to evaluate                |
| `--track`        | Evaluation track                     |
| `--episodes`     | Number of episodes                   |
| `--render-sleep` | Delay between frames                 |
| `--max-steps`    | Maximum number of steps per episode  |
| `--width-scale`  | Visual scale of the track boundaries |

---

# Generated files

For each training run, the following files are saved:

```text
<nome_run>.zip
<nome_run>_training_history.csv
<nome_run>_metadata.json
checkpoints/<nome_run>/
```

The CSV contains episode-by-episode metrics, including:

* seed;
* track;
* reward;
* progress;
* episode duration;
* completion;
* lap time;
* off-track termination;
* vehicle stall.

The results of the `watch` command are appended to:

````text
watch_results.csv


## Multi-seed plot generation

To generate aggregated training plots using the results from the three seeds:

```bash
python plots.py \
  path/to/seed_42_training_history.csv \
  path/to/seed_43_training_history.csv \
  path/to/seed_44_training_history.csv \
  --output-dir plots \
  --prefix ppo_multiseed
````

For example, for the `256-256-128-64-32` architecture:

```bash
python plots.py \
  runs/ppo_f1_256_256_128_64_32_seed_42_training_history.csv \
  runs/ppo_f1_256_256_128_64_32_seed_43_training_history.csv \
  runs/ppo_f1_256_256_128_64_32_seed_44_training_history.csv \
  --output-dir plots/256_256_128_64_32 \
  --prefix ppo_256_256_128_64_32_multiseed
```

The command generates:

* `<prefix>_mean_lap_progress.png`
* `<prefix>_completion_rate.png`
* `<prefix>_mean_reward.png`

The CSV files must contain at least the following columns:

```text
timesteps, track_name, rewards, progress, success
```

It is also possible to control aggregation and smoothing:

```bash
python plots.py \
  seed_42.csv seed_43.csv seed_44.csv \
  --output-dir plots \
  --prefix ppo_multiseed \
  --bin-size 200000 \
  --smooth-bins 3 \
  --dpi 300
```
