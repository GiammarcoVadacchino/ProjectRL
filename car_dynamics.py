
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LocalStateInfo:

    # Vettore di osservazione dello stato fornito a PPO per calcolare la prossima azione.
    observation: np.ndarray

    # Distanza laterale con segno dell'auto dalla linea centrale del circuito.
    lateral_error: float

    # Distanza tra l'orientamento dell'auto e la direzione locale del circuito.
    heading_error: float

    # True se l'auto si trova oltre i limiti fisici del circuito.
    is_off_track: bool

    # Indice del segmento più vicino della linea centrale.
    closest_idx: int

    # Progresso dell'auto in metri lungo la linea centrale del circuito.
    current_progress: float

    # Semilarghezza locale del circuito, utilizzata per normalizzare l'errore laterale.
    # Il circuito può avere larghezze diverse nei vari punti.
    track_half_width: float

    # Componente della velocità allineata con la direzione locale del circuito.
    v_longitudinal: float

    # Componente della velocità perpendicolare alla direzione locale del circuito.
    v_lateral: float

    # Curvature future del circuito a distanze di lookahead fisse.
    # Sono utili per anticipare le curve successive.
    future_curvatures: np.ndarray


class CarDynamics:

    # Distanze future utilizzate per descrivere il circuito davanti all'auto.
    LOOKAHEAD_DISTANCES = [10.0, 30.0, 60.0, 100.0, 150.0]

    def __init__(
        self,
        start_x: float,
        start_y: float,
        start_yaw: float,
        max_speed: float = 95.0,
        initial_speed: float = 0.0,
    ):
        # Posa dell'auto nel piano 2D globale (x,y,yaw)
        self.x = float(start_x)
        self.y = float(start_y)
        self.yaw = float(start_yaw)

        # Velocità scalare
        self.v = float(initial_speed)

        # Memoria dell'azione precedente.
        # Questi valori sono inclusi nell'osservazione affinché PPO possa apprendere un comportamento di guida più fluido.
        self.prev_steer = 0.0
        self.prev_throttle = 0.0

        # Parametri fisici semplificati.
        self.car_length = 3.6
        self.max_speed = float(max_speed)
        self.dt = 0.1

        self.lookahead_distances = list(self.LOOKAHEAD_DISTANCES)

    @staticmethod
    def wrap_angle(angle: float) -> float:
        """
        Riporta un angolo nell'intervallo [-pi, pi].

        Questo evita discontinuità quando si confrontano angoli vicini al
        confine -pi / +pi.
        """
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def update_physics(self, throttle: float, steer: float) -> None:

        # Limita i comandi all'intervallo fisico valido delle azioni.
        # PPO dovrebbe già produrre azioni comprese in questo intervallo grazie allo
        # spazio delle azioni di Gymnasium, ma questo fornisce un ulteriore controllo di sicurezza.
        throttle = float(np.clip(throttle, -1.0, 1.0))
        steer = float(np.clip(steer, -1.0, 1.0))

        # Parametri fisici semplificati ispirati a una monoposto di F1.
        MAX_ENGINE_ACC = 10.0     # m/s^2, circa 1.0g
        MAX_BRAKE_ACC = 38.0      # m/s^2, circa 3.9g
        MAX_STEER = 0.45          # radianti, circa 25.8 gradi
        MAX_LATERAL_G = 3.5       # limite semplificato di aderenza laterale
        G = 9.81

        # Accelerazione longitudinale.
        # Un throttle positivo accelera l'auto, mentre un throttle negativo applica la frenata.
        if throttle >= 0.0:
            acc = throttle * MAX_ENGINE_ACC
        else:
            acc = throttle * MAX_BRAKE_ACC

        # Converte il comando di sterzata normalizzato in un angolo di sterzata fisico.
        delta = steer * MAX_STEER

        # Aggiorna la velocità scalare e la mantiene entro limiti validi.
        self.v += acc * self.dt
        self.v = float(np.clip(self.v, 0.0, self.max_speed))

        # Accelerazione laterale massima consentita dal modello.
        a_lat_max = MAX_LATERAL_G * G

        # Velocità angolare desiderata dal modello cinematico bicycle.
        yaw_rate_cmd = (self.v / self.car_length) * np.tan(delta)

        # Limita la velocità di imbardata affinché l'accelerazione laterale non superi il
        # limite di aderenza. Poiché a_lat ≈ v * yaw_rate, la velocità massima di imbardata diminuisce
        # all'aumentare della velocità.
        yaw_rate_max = a_lat_max / max(self.v, 1e-3)
        yaw_rate = np.clip(yaw_rate_cmd, -yaw_rate_max, yaw_rate_max)

        # Aggiorna l'orientamento dell'auto.
        self.yaw += yaw_rate * self.dt
        self.yaw = self.wrap_angle(self.yaw)

        # Aggiorna la posizione dell'auto nel sistema di riferimento globale.
        self.x += self.v * np.cos(self.yaw) * self.dt
        self.y += self.v * np.sin(self.yaw) * self.dt

        # Memorizza l'azione corrente affinché possa essere utilizzata come azione precedente al
        # passo successivo dell'ambiente.
        self.prev_steer = steer
        self.prev_throttle = throttle

    def _project_on_centerline(self, centerline, segment_lengths, s_start):

        # Posizione corrente dell'auto in coordinate globali.
        car_pos = np.array([self.x, self.y], dtype=np.float32)

        # Costruisce tutti i segmenti della linea centrale.
        # p0[i] è l'inizio del segmento i, mentre p1[i] è il punto successivo della linea centrale.
        p0 = centerline
        p1 = np.roll(centerline, -1, axis=0)
        seg = p1 - p0

        # Squared segment lengths, protected against zero-length segments.
        seg_len_sq = np.sum(seg * seg, axis=1)
        seg_len_sq = np.maximum(seg_len_sq, 1e-9)

        # Coordinata di proiezione normalizzata su ciascun segmento.
        # t = 0 indica che la proiezione si trova in p0.
        # t = 1 indica che la proiezione si trova in p1.
        # I valori vengono limitati affinché la proiezione rimanga all'interno del segmento.
        t = np.sum((car_pos - p0) * seg, axis=1) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)

        # Calcola il punto proiettato su ciascun segmento.
        projections = p0 + t[:, None] * seg

        # Seleziona il segmento il cui punto proiettato è più vicino all'auto.
        distances = np.linalg.norm(projections - car_pos, axis=1)
        closest_idx = int(np.argmin(distances))

        # Punto più vicino sulla linea centrale.
        projected_point = projections[closest_idx]

        # Progresso curvilineo lungo la linea centrale.
        # s_start[closest_idx] è la distanza dalla linea di partenza fino
        # all'inizio del segmento. Il secondo termine aggiunge la distanza percorsa
        # all'interno di quel segmento.
        projected_s = float(
            s_start[closest_idx] + t[closest_idx] * segment_lengths[closest_idx]
        )

        return car_pos, projected_point, closest_idx, projected_s

    def _point_at_progress(self, centerline, s_start, segment_lengths, progress: float):
        """
        Interpola un punto della linea centrale a una determinata distanza
        curvilinea.

        Questa funzione viene utilizzata per ottenere punti futuri del circuito
        a distanze di lookahead fisse rispetto alla posizione corrente dell'auto.
        """

        # Lunghezza totale del circuito.
        track_length = float(np.sum(segment_lengths))
        progress = float(progress % track_length)

        # Trova il segmento che contiene il valore di progresso richiesto.
        idx = int(np.searchsorted(s_start, progress, side="right") - 1)
        idx = max(0, min(idx, len(centerline) - 1))

        # Posizione locale all'interno del segmento selezionato.
        local_s = progress - float(s_start[idx])
        t = local_s / max(float(segment_lengths[idx]), 1e-9)

        # Interpolazione lineare tra i due estremi del segmento.
        p0 = centerline[idx]
        p1 = centerline[(idx + 1) % len(centerline)]
        return p0 + t * (p1 - p0), idx

    def _world_to_car_frame(self, point: np.ndarray) -> tuple[float, float]:
        """
        Converte un punto globale nel sistema di riferimento locale dell'auto.

        Nel sistema di riferimento locale:
        - una x locale positiva indica che il punto si trova davanti all'auto;
        - una y locale positiva indica che il punto si trova a sinistra dell'auto.
        """

        # Vettore dall'auto al punto espresso in coordinate globali.
        dx = float(point[0] - self.x)
        dy = float(point[1] - self.y)

        # Termini di rotazione ricavati dall'orientamento dell'auto.
        c = np.cos(self.yaw)
        s = np.sin(self.yaw)

        # Ruota lo spostamento globale nel sistema di riferimento dell'auto.
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy

        return float(local_x), float(local_y)

    def compute_local_state(
        self,
        centerline,
        normals_left,
        headings,
        w_left,
        w_right,
        s_start,
        curvature,
        track_length,
        segment_lengths=None,
    ) -> LocalStateInfo:
        """
        Build the normalized local observation for PPO.

        Final observation seen by the agent:

        [
            lateral_error_norm,
            sin_heading_error,
            cos_heading_error,

            speed_norm,
            v_longitudinal_norm,
            v_lateral_norm,

            previous_steering,
            previous_throttle,

            future_center_x_1,
            future_center_y_1,
            ...,
            future_center_x_5,
            future_center_y_5,

            future_curvature_1,
            ...,
            future_curvature_5,
        ]

        Future track points are expressed in the car frame and normalized. This
        makes the policy less dependent on absolute position on the circuit.
        """

        #TODO: VEDERE SERVE QUESTO IF
        # If segment lengths were not precomputed by the environment, compute
        # them here as a fallback.
        if segment_lengths is None:
            next_center = np.roll(centerline, -1, axis=0)
            segment_lengths = np.linalg.norm(next_center - centerline, axis=1)

        # Proietta l'auto sulla linea centrale del circuito per ottenere il suo progresso,
        # il segmento più vicino e il punto più vicino della linea centrale.
        car_pos, projected_point, closest_idx, current_progress = self._project_on_centerline(
            centerline=centerline,
            segment_lengths=segment_lengths,
            s_start=s_start,
        )

        # Mantiene il progresso nell'intervallo del circuito chiuso.
        current_progress = current_progress % float(track_length)

        # Calcola l'errore laterale con segno usando la normale sinistra del segmento
        # più vicino della linea centrale.
        normal_left = normals_left[closest_idx]
        rel_position = car_pos - projected_point
        lateral_error = float(np.dot(rel_position, normal_left))

        # La larghezza fisica del circuito può essere diversa sul lato sinistro e su quello destro.
        # Di conseguenza, il controllo di uscita dal circuito dipende dal segno di lateral_error.
        if lateral_error >= 0.0:
            track_half_width = float(w_left[closest_idx])
            is_off_track = lateral_error > track_half_width
        else:
            track_half_width = float(w_right[closest_idx])
            is_off_track = abs(lateral_error) > track_half_width

        # Calcola l'errore di orientamento tra l'orientamento dell'auto e la direzione
        # tangente locale del circuito.
        track_yaw = float(headings[closest_idx])
        heading_error = self.wrap_angle(self.yaw - track_yaw)

        # Scompone la velocità scalare in componenti rispetto alla direzione locale del circuito.
        # Una velocità laterale elevata indica un movimento instabile o trasversale.
        v_longitudinal = float(self.v * np.cos(heading_error))
        v_lateral = float(self.v * np.sin(heading_error))

        # Liste che conterranno i punti futuri della linea centrale e le curvature
        # future utilizzate come informazioni di lookahead.
        future_points_norm: list[float] = []
        future_curvatures: list[float] = []
        max_lookahead = max(self.lookahead_distances)

        # Calcola i punti futuri del circuito a distanze fisse davanti all'auto.
        for dist in self.lookahead_distances:
            # Distanza curvilinea del punto futuro.
            future_progress = (current_progress + dist) % float(track_length)

            # Punto futuro interpolato sulla linea centrale..
            future_point, future_idx = self._point_at_progress(
                centerline=centerline,
                s_start=s_start,
                segment_lengths=segment_lengths,
                progress=future_progress,
            )

            # Esprime il punto futuro nel sistema di riferimento dell'auto.
            local_x, local_y = self._world_to_car_frame(future_point)

            # Normalizza rispetto alla massima distanza di lookahead per mantenere i valori delle feature
            # su una scala simile a quella delle altre componenti dell'osservazione. (in questo modo si evitano gradienti troppo grandi)
            future_points_norm.extend(
                [
                    float(np.clip(local_x / max_lookahead, -2.0, 2.0)),
                    float(np.clip(local_y / max_lookahead, -2.0, 2.0)),
                ]
            )

            # Memorizza la curvatura futura non normalizzata.
            future_curvatures.append(float(curvature[future_idx]))

        # Normalizzazione delle principali caratteristiche dello stato.

        # lateral_error_norm = 0 indica che l'auto si trova sulla linea centrale.
        # lateral_error_norm = 1 indica che l'auto si trova approssimativamente sul bordo del circuito.
        lateral_error_norm = np.clip(
            lateral_error / max(track_half_width, 1e-6),
            -2.0,
            2.0,
        )

        # Normalizza la velocità totale rispetto alla velocità massima.
        speed_norm = np.clip(self.v / max(self.max_speed, 1e-6), 0.0, 1.5)

        # Normalizza le componenti della velocità rispetto alla velocità massima.
        v_longitudinal_norm = np.clip(
            v_longitudinal / max(self.max_speed, 1e-6),
            -1.5,
            1.5,
        )
        v_lateral_norm = np.clip(
            v_lateral / max(self.max_speed, 1e-6),
            -1.5,
            1.5,
        )

        # Codifica l'errore di orientamento usando seno e coseno invece dell'angolo
        # grezzo. Questo evita discontinuità intorno a -pi e +pi.
        sin_heading_error = float(np.sin(heading_error))
        cos_heading_error = float(np.cos(heading_error))

        # Le curvature dei circuiti reali sono generalmente valori numerici piccoli. Vengono
        # amplificate per renderle più facilmente utilizzabili dalla rete neurale e successivamente
        # limitate per evitare valori anomali instabili.
        future_curvatures_norm = [
            float(np.clip(c * 50.0, -5.0, 5.0))
            for c in future_curvatures
        ]

        # Vettore di osservazione finale fornito a PPO.
        # Dimensione:
        # 8 feature di base
        # + 10 coordinate dei punti futuri, poiché 5 punti * 2 coordinate
        # + 5 valori di curvatura futura
        # = 23 feature.
        obs = np.array(
            [
                float(lateral_error_norm),
                sin_heading_error,
                cos_heading_error,
                float(speed_norm),
                float(v_longitudinal_norm),
                float(v_lateral_norm),
                float(self.prev_steer),
                float(self.prev_throttle),
                *future_points_norm,
                *future_curvatures_norm,
            ],
            dtype=np.float32,
        )

        # Restituisce sia l'osservazione di PPO sia le informazioni ausiliarie utilizzate
        # dall'ambiente per il calcolo della ricompensa, i controlli di terminazione e il logging.
        return LocalStateInfo(
            observation=obs,
            lateral_error=lateral_error,
            heading_error=heading_error,
            is_off_track=bool(is_off_track),
            closest_idx=closest_idx,
            current_progress=float(current_progress),
            track_half_width=track_half_width,
            v_longitudinal=v_longitudinal,
            v_lateral=v_lateral,
            future_curvatures=np.asarray(future_curvatures, dtype=np.float32),
        )