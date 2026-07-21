# Autonomous Racing con PPO

Progetto di Reinforcement Learning per il controllo di un'auto Formula-style su uno o più circuiti.

L'agente utilizza **Proximal Policy Optimization (PPO)** e controlla due azioni continue:

- acceleratore/freno;
- angolo di sterzata.

Il programma espone due comandi principali:

- `train`: addestra un modello da zero oppure continua il training da un modello esistente;
- `watch`: valuta e visualizza un modello già allenato.

> Il fine-tuning non ha un comando separato. Si esegue con `train` insieme a `--model-path`.

---

## File principali

```text
main.py           Interfaccia da riga di comando
training.py       Training, checkpoint, logging e valutazione
f1_env.py         Ambiente Gymnasium e rendering
car_dynamics.py   Dinamica semplificata e osservazione
reward.py         Funzione di reward
```

I circuiti sono letti da file CSV contenenti centerline e larghezze della pista.

---

## Help

```bash
python main.py --help
python main.py train --help
python main.py watch --help
```

---

# Comando `train`

`train` può avviare un nuovo addestramento oppure continuare da un modello già esistente.

## Training da zero

Nel training da zero è necessario specificare almeno un'architettura tramite `--archs`.

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

È possibile allenare più architetture in sequenza:

```bash
python main.py train \
  --tracks racetrack-database/tracks/Monza.csv \
           racetrack-database/tracks/Melbourne.csv \
           racetrack-database/tracks/Silverstone.csv \
  --archs 64,64,32,16 128,128,64,32 256,256,128,64,32
```

Ogni architettura genera un modello indipendente.

## Continuare un modello esistente

Per riprendere il training si usa `--model-path`.

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

In questa modalità:

- l'architettura viene caricata dal modello;
- `--archs` non deve essere usato;
- i timesteps indicati sono aggiuntivi;
- il contatore dei timesteps non viene azzerato.

## Continuazione multi-track

Un modello può continuare il training su più circuiti:

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

Viene creato un ambiente per ogni circuito e PPO raccoglie esperienza da tutte le piste.

## Partenza fissa o casuale

```bash
--random-start
```

Avvia ogni episodio da una posizione casuale.

```bash
--no-random-start
```

Avvia ogni episodio dalla posizione iniziale standard.

Se l'opzione non viene specificata:

- training da zero: partenza casuale;
- training con `--model-path`: partenza fissa.

## Opzioni principali di `train`

| Opzione | Descrizione |
|---|---|
| `--tracks` | Uno o più file CSV dei circuiti |
| `--archs` | Architetture usate nel training da zero |
| `--model-path` | Modello PPO da caricare |
| `--timesteps` | Timesteps del run o timesteps aggiuntivi |
| `--learning-rate` | Learning rate PPO |
| `--n-steps` | Step raccolti per ambiente prima dell'update |
| `--batch-size` | Dimensione dei mini-batch |
| `--ent-coef` | Coefficiente di entropia |
| `--seed` | Seed del run |
| `--random-start` | Abilita la partenza casuale |
| `--no-random-start` | Usa la partenza fissa |
| `--checkpoint-freq` | Frequenza dei checkpoint; `0` li disattiva |
| `--output-dir` | Directory dei risultati |
| `--run-name` | Nome personalizzato del run |
| `--max-steps` | Numero massimo di step per episodio |

---

# Comando `watch`

`watch` carica un modello, lo valuta deterministicamente e mostra il rendering.

```bash
python main.py watch \
  --model-path runs_training/modello.zip \
  --track racetrack-database/tracks/Monza.csv \
  --episodes 1
```

Durante la valutazione:

- i pesi non vengono aggiornati;
- le azioni sono deterministiche;
- la partenza è fissa;
- vengono mostrati bordi della pista, raceline, posizione e orientamento dell'auto.

La raceline deve trovarsi in:

```text
racetrack-database/racelines/
```

e deve avere lo stesso nome del file della pista.

## Opzioni principali di `watch`

| Opzione | Descrizione |
|---|---|
| `--model-path` | Modello PPO da valutare |
| `--track` | Circuito di valutazione |
| `--episodes` | Numero di episodi |
| `--render-sleep` | Pausa tra i frame |
| `--max-steps` | Numero massimo di step per episodio |
| `--width-scale` | Scala visuale dei bordi |

---

# File prodotti

Per ogni training vengono salvati:

```text
<nome_run>.zip
<nome_run>_training_history.csv
<nome_run>_metadata.json
checkpoints/<nome_run>/
```

Il CSV contiene metriche episodio per episodio, tra cui:

- seed;
- circuito;
- reward;
- progresso;
- durata dell'episodio;
- completamento;
- lap time;
- uscita di pista;
- arresto del veicolo.

I risultati del comando `watch` vengono aggiunti a:

```text
watch_results.csv


## Generazione dei grafici multi-seed

Per generare i grafici aggregati del training usando i risultati dei tre seed:

```bash
python plots.py \
  path/to/seed_42_training_history.csv \
  path/to/seed_43_training_history.csv \
  path/to/seed_44_training_history.csv \
  --output-dir plots \
  --prefix ppo_multiseed
```

Ad esempio, per l’architettura `256-256-128-64-32`:

```bash
python plots.py \
  runs/ppo_f1_256_256_128_64_32_seed_42_training_history.csv \
  runs/ppo_f1_256_256_128_64_32_seed_43_training_history.csv \
  runs/ppo_f1_256_256_128_64_32_seed_44_training_history.csv \
  --output-dir plots/256_256_128_64_32 \
  --prefix ppo_256_256_128_64_32_multiseed
```

Il comando genera:

* `<prefix>_mean_lap_progress.png`
* `<prefix>_completion_rate.png`
* `<prefix>_mean_reward.png`

I CSV devono contenere almeno le colonne:

```text
timesteps, track_name, rewards, progress, success
```

È inoltre possibile controllare l’aggregazione e lo smoothing:

```bash
python plots.py \
  seed_42.csv seed_43.csv seed_44.csv \
  --output-dir plots \
  --prefix ppo_multiseed \
  --bin-size 200000 \
  --smooth-bins 3 \
  --dpi 300
```


