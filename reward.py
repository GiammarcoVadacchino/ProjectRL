

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardConfig:

    # Peso del progresso in metri lungo la linea centrale.
    # È il termine positivo principale della ricompensa.
    progress_weight: float = 3.0

    # Penalità costante applicata a ogni step.
    # Spinge l'agente a completare il giro nel minor numero possibile di step.
    time_penalty: float = -0.15

    # Peso della penalità quadratica associata all'errore laterale normalizzato.
    lateral_error_weight: float = 0.8

    # Peso della penalità aggiuntiva applicata vicino ai bordi della pista.
    edge_weight: float = 20.0

    # Peso della penalità per il disallineamento rispetto alla pista.
    heading_weight: float = 0.4

    # Peso della penalità sulla componente laterale della velocità.
    lateral_velocity_weight: float = 0.025

    # Peso della penalità sull'ampiezza assoluta del comando di sterzo.
    steer_weight: float = 0.03

    # Peso della penalità sulle variazioni del comando di sterzo.
    steering_smooth_weight: float = 0.20

    # Peso della penalità sulle variazioni del comando di acceleratore.
    throttle_smooth_weight: float = 0.03

    # Peso della penalità sull'accelerazione laterale oltre la soglia.
    centripetal_weight: float = 0.01

    # Soglia di accelerazione laterale oltre la quale viene applicata la penalità.
    centripetal_limit: float = 45.0

    # Penalità applicata quando l'auto rimane quasi ferma dopo i primi step.
    low_speed_penalty: float = -1.0

    # Peso del premio per la velocità longitudinale sui rettilinei.
    straight_speed_weight: float = 0.5


@dataclass
class RewardBreakdown:
    """
    Memorizza il valore totale della ricompensa e il contributo di ogni termine.

    La scomposizione è utile per il logging e il debugging, perché permette di
    capire quali componenti stanno guidando il comportamento della policy.
    """

    # Ricompensa totale ottenuta sommando tutti i contributi.
    total: float

    # Contributo positivo legato al progresso lungo la pista.
    progress: float

    # Penalità temporale costante.
    time: float

    # Penalità dovuta all'errore laterale.
    lateral_error: float

    # Penalità aggiuntiva dovuta alla vicinanza ai bordi.
    edge: float

    # Penalità dovuta all'errore di orientamento.
    heading: float

    # Penalità dovuta alla velocità laterale.
    lateral_velocity: float

    # Penalità sull'ampiezza del comando di sterzo.
    steer: float

    # Penalità sulle variazioni del comando di sterzo.
    smooth_steer: float

    # Penalità sulle variazioni del comando di acceleratore.
    smooth_throttle: float

    # Penalità per accelerazione laterale eccessiva.
    centripetal: float

    # Penalità applicata quando la macchina rimane quasi ferma.
    low_speed: float

    # Premio per la velocità longitudinale sui rettilinei.
    straight_speed: float

    def as_dict(self) -> dict[str, float]:
        """
        Converte la scomposizione della ricompensa in un dizionario.

        Il formato risultante è adatto al salvataggio nei log o in un file CSV.
        Ogni valore viene convertito esplicitamente in `float` per evitare tipi
        numerici NumPy non sempre serializzabili direttamente.
        """
        return {
            "reward_total": float(self.total),
            "reward_progress": float(self.progress),
            "reward_time": float(self.time),
            "reward_lateral_error": float(self.lateral_error),
            "reward_edge": float(self.edge),
            "reward_heading": float(self.heading),
            "reward_lateral_velocity": float(self.lateral_velocity),
            "reward_steer": float(self.steer),
            "reward_smooth_steer": float(self.smooth_steer),
            "reward_smooth_throttle": float(self.smooth_throttle),
            "reward_centripetal": float(self.centripetal),
            "reward_low_speed": float(self.low_speed),
            "reward_straight_speed": float(self.straight_speed),
        }


def compute_racing_reward(
    *,
    delta_progress: float,
    normalized_lateral_error: float,
    heading_error: float,
    v_lateral: float,
    v_longitudinal: float,
    future_curvatures: np.ndarray,
    speed: float,
    max_speed: float,
    steer: float,
    throttle: float,
    previous_steer: float,
    previous_throttle: float,
    car_length: float,
    step_count: int,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """
    Calcola la ricompensa associata a un singolo step dell'ambiente.



    Parametri:
    - delta_progress: progresso in metri lungo la linea centrale dall'ultimo step;
    - normalized_lateral_error: errore laterale diviso per la semilarghezza locale;
    - heading_error: differenza angolare rispetto alla tangente locale della pista;
    - v_lateral: componente della velocità perpendicolare alla pista;
    - v_longitudinal: componente della velocità parallela alla pista;
    - future_curvatures: curvature della pista ai punti di lookahead;
    - speed: velocità scalare corrente della macchina;
    - max_speed: velocità massima usata per la normalizzazione;
    - steer: comando di sterzata corrente normalizzato;
    - throttle: comando di acceleratore corrente normalizzato;
    - previous_steer: comando di sterzata applicato allo step precedente;
    - previous_throttle: comando di acceleratore applicato allo step precedente;
    - car_length: lunghezza caratteristica del modello cinematico;
    - step_count: numero di step trascorsi nell'episodio;
    - config: configurazione opzionale dei pesi della ricompensa.

    Restituisce:
    - un oggetto `RewardBreakdown` contenente la ricompensa totale e tutti i
      singoli contributi.
    """

    # Se non viene fornita una configurazione personalizzata, vengono utilizzati
    # i valori predefiniti definiti nella dataclass RewardConfig.
    cfg = config or RewardConfig()

    # Termine principale della ricompensa:
    # un progresso positivo lungo la pista produce un premio proporzionale ai
    # metri percorsi. In questo modo l'agente viene incentivato ad avanzare.
    progress_reward = cfg.progress_weight * float(delta_progress)

    # Penalità temporale costante applicata a ogni step.
    # Poiché ogni step ha una durata fissa, questa penalità equivale a premiare
    # implicitamente il completamento del giro in meno tempo.
    time_reward = cfg.time_penalty

    # Si usa il valore assoluto perché la distanza dal centro è rilevante
    # indipendentemente dal fatto che l'auto si trovi a sinistra o a destra.
    abs_error = abs(float(normalized_lateral_error))

    # Penalità quadratica per la distanza dalla linea centrale.
    # La forma quadratica rende piccole le correzioni vicino al centro e aumenta
    # rapidamente la penalità quando l'errore laterale diventa elevato.
    # Il termine non è eccessivamente forte perché una traiettoria da gara può
    # legittimamente allargarsi per sfruttare tutta la larghezza della pista.
    lateral_error_reward = -cfg.lateral_error_weight * (abs_error ** 2)

    # Penalità aggiuntiva attivata solamente oltre l'80% della semilarghezza.
    # Serve a scoraggiare fortemente la guida vicino ai limiti fisici della pista,
    # dove una piccola variazione del controllo potrebbe causare un'uscita.
    edge_reward = 0.0
    if abs_error > 0.80:
        edge_reward = -cfg.edge_weight * ((abs_error - 0.80) ** 2)

    # Penalità per il disallineamento rispetto alla tangente locale della pista.
    # L'uso di |sin(heading_error)| produce:
    # - penalità nulla quando la macchina è allineata;
    # - penalità crescente con il disallineamento;
    # - un valore limitato tra 0 e 1 prima dell'applicazione del peso.
    heading_reward = -cfg.heading_weight * abs(float(np.sin(heading_error)))

    # Penalità proporzionale al valore assoluto della velocità laterale.
    # Una componente laterale elevata indica che la macchina si sta muovendo
    # trasversalmente rispetto alla direzione locale della pista.
    lateral_velocity_reward = -cfg.lateral_velocity_weight * abs(float(v_lateral))

    # Conversione del comando normalizzato di sterzo in un angolo espresso in
    # radianti. Il fattore 0.5 rappresenta l'angolo massimo utilizzato dalla
    # stima semplificata dell'accelerazione laterale.
    steer_rad = float(steer) * 0.5

    # Stima dell'accelerazione laterale mediante il modello cinematico bicycle:
    #
    #     a_lat = v^2 * |tan(delta)| / L
    #
    # dove v è la velocità, delta è l'angolo di sterzo e L è la lunghezza
    # caratteristica della macchina. Il valore minimo 1e-6 evita una divisione
    # per zero nel caso di una lunghezza non valida.
    lateral_acc = (
        (float(speed) ** 2)
        * abs(np.tan(steer_rad))
        / max(float(car_length), 1e-6)
    )

    # La penalità centripeta viene applicata solo alla parte di accelerazione
    # laterale che supera la soglia configurata. La crescita quadratica rende
    # particolarmente costose le manovre molto oltre il limite.
    centripetal_reward = (
        -cfg.centripetal_weight
        * max(0.0, lateral_acc - cfg.centripetal_limit) ** 2
    )

    # Penalità quadratica sull'ampiezza del comando di sterzo.
    # Favorisce comandi contenuti, senza impedire sterzate forti quando sono
    # necessarie per seguire la pista.
    steer_reward = -cfg.steer_weight * (float(steer) ** 2)

    # Penalità quadratica sulla variazione dello sterzo tra due step consecutivi.
    # Questo termine riduce lo zig-zag e favorisce traiettorie più regolari.
    smooth_steer_reward = -cfg.steering_smooth_weight * (
        (float(steer) - float(previous_steer)) ** 2
    )

    # Penalità quadratica sulla variazione del throttle.
    # Riduce le alternanze brusche tra accelerazione e frenata.
    smooth_throttle_reward = -cfg.throttle_smooth_weight * (
        (float(throttle) - float(previous_throttle)) ** 2
    )

    # Il termine è inizialmente nullo per consentire alla macchina di partire.
    low_speed_reward = 0.0

    # Dopo i primi 20 step, una velocità inferiore a 1 m/s produce una penalità.
    # In questo modo fermarsi non può diventare una strategia conveniente per
    # evitare le altre penalità.
    if float(speed) < 1.0 and int(step_count) > 20:
        low_speed_reward = cfg.low_speed_penalty

    # Premio di velocità applicato solamente quando la pista futura è quasi
    # rettilinea. Si considera la massima curvatura assoluta tra tutti i punti
    # osservati in avanti, adottando quindi una scelta prudente.
    future_curve = float(np.max(np.abs(future_curvatures)))

    # Il fattore di rettilineità decade esponenzialmente con la curvatura:
    #
    #     straight_factor = exp(-80 * future_curve)
    #
    # Se la curvatura è vicina a zero, il fattore è vicino a 1.
    # In presenza di curve, il fattore diminuisce rapidamente verso zero.
    straight_factor = float(np.exp(-80.0 * future_curve))

    # Il premio viene ridotto se la macchina non è allineata con la pista.
    # Il massimo con zero impedisce di premiare una macchina orientata nella
    # direzione opposta rispetto alla tangente locale.
    alignment_factor = float(max(np.cos(heading_error), 0.0))

    # La velocità longitudinale positiva viene normalizzata rispetto alla
    # velocità massima. Il bonus è quindi elevato solamente quando:
    # - la pista davanti è quasi rettilinea;
    # - la macchina è correttamente allineata;
    # - la macchina procede velocemente nella direzione della pista.
    speed_bonus = (
        straight_factor
        * alignment_factor
        * max(float(v_longitudinal), 0.0)
        / max(float(max_speed), 1e-6)
    )

    # Applica il peso configurato al bonus di velocità sui rettilinei.
    straight_speed_reward = cfg.straight_speed_weight * speed_bonus

    # Somma di tutti i contributi positivi e negativi.
    total = (
        progress_reward
        + time_reward
        + lateral_error_reward
        + edge_reward
        + heading_reward
        + lateral_velocity_reward
        + steer_reward
        + smooth_steer_reward
        + smooth_throttle_reward
        + centripetal_reward
        + low_speed_reward
        + straight_speed_reward
    )

    # Restituisce sia il totale sia ogni componente separata, così che
    # l'ambiente possa registrare e analizzare il comportamento della reward.
    return RewardBreakdown(
        total=float(total),
        progress=float(progress_reward),
        time=float(time_reward),
        lateral_error=float(lateral_error_reward),
        edge=float(edge_reward),
        heading=float(heading_reward),
        lateral_velocity=float(lateral_velocity_reward),
        steer=float(steer_reward),
        smooth_steer=float(smooth_steer_reward),
        smooth_throttle=float(smooth_throttle_reward),
        centripetal=float(centripetal_reward),
        low_speed=float(low_speed_reward),
        straight_speed=float(straight_speed_reward),
    )