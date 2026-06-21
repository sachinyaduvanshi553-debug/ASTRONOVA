import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger("astronova.event_lifecycle")

class EventLifecycleTracker:
    """
    Solar Flare Lifecycle Tracker.
    Implements a state machine representing the physical lifecycle of a flare event:
    Quiescent -> Precursor -> Initiation -> Growth -> Peak -> Decay -> Quiescent
    """
    
    STATES = ["Quiescent", "Precursor", "Initiation", "Growth", "Peak", "Decay"]
    
    def __init__(self, init_gradient_threshold: float = 1e-7, growth_gradient_threshold: float = 1e-6):
        """
        Args:
            init_gradient_threshold: Gradient threshold (W/m^2/min) to transition from Precursor to Initiation.
            growth_gradient_threshold: Gradient threshold (W/m^2/min) to transition from Initiation to Growth.
        """
        self.init_gradient_threshold = init_gradient_threshold
        self.growth_gradient_threshold = growth_gradient_threshold
        
        # Current state properties
        self.current_state = "Quiescent"
        self.state_start_time: Optional[datetime] = None
        self.peak_flux: float = 0.0
        self.pre_flare_bg_flux: float = 1e-9
        
        # State duration logging
        self.history: List[Dict[str, Any]] = []
        
    def reset(self, start_time: datetime, bg_flux: float = 1e-9):
        self.current_state = "Quiescent"
        self.state_start_time = start_time
        self.peak_flux = bg_flux
        self.pre_flare_bg_flux = bg_flux
        self.history = []
        
    def _change_state(self, new_state: str, current_time: datetime):
        if new_state == self.current_state:
            return
            
        prev_state = self.current_state
        duration = 0.0
        
        if self.state_start_time is not None:
            duration = (current_time - self.state_start_time).total_seconds() / 60.0
            
        self.history.append({
            "state": prev_state,
            "start_time": self.state_start_time,
            "end_time": current_time,
            "duration_minutes": duration
        })
        
        logger.info(f"Lifecycle state transition: {prev_state} -> {new_state} at {current_time} (duration in previous state: {duration:.2f} min)")
        
        self.current_state = new_state
        self.state_start_time = current_time
        
    def update(self, current_time: datetime, flux: float, grad_1st: float, grad_2nd: float) -> str:
        """
        Updates the state machine with the latest observation data.
        
        Args:
            current_time: Current timestamp.
            flux: Current soft X-ray flux.
            grad_1st: First derivative of flux (dF/dt) in W/m^2/min.
            grad_2nd: Second derivative of flux (d2F/dt2) in W/m^2/min^2.
            
        Returns:
            The new state string.
        """
        if self.state_start_time is None:
            self.state_start_time = current_time
            self.peak_flux = flux
            self.pre_flare_bg_flux = flux
            
        # 1. State logic transitions
        if self.current_state == "Quiescent":
            # Precursor: small but positive gradient and second derivative
            if grad_1st > 1e-8 and flux > self.pre_flare_bg_flux * 1.05:
                self.pre_flare_bg_flux = flux
                self._change_state("Precursor", current_time)
            elif grad_1st > self.init_gradient_threshold:
                self._change_state("Initiation", current_time)
                
        elif self.current_state == "Precursor":
            if grad_1st > self.init_gradient_threshold:
                self._change_state("Initiation", current_time)
            elif grad_1st <= 0 and flux <= self.pre_flare_bg_flux * 1.02:
                self._change_state("Quiescent", current_time)
                
        elif self.current_state == "Initiation":
            if grad_1st > self.growth_gradient_threshold:
                self._change_state("Growth", current_time)
            elif grad_1st <= 0:
                # Reached a premature peak or failed flare
                self._change_state("Peak", current_time)
                self.peak_flux = flux
                
        elif self.current_state == "Growth":
            if grad_1st <= 0 or grad_2nd < -1e-6:
                # Growth slowing down or stopping => transition to peak
                self._change_state("Peak", current_time)
                self.peak_flux = flux
                
        elif self.current_state == "Peak":
            # Peak is brief; once flux is decreasing, we enter Decay
            if grad_1st < -1e-8:
                self._change_state("Decay", current_time)
            # Update peak if flux keeps climbing (e.g. secondary peak)
            if flux > self.peak_flux:
                self.peak_flux = flux
                
        elif self.current_state == "Decay":
            # Decay continues until flux stabilizes near pre-flare background or gradient is flat
            flare_amplitude = self.peak_flux - self.pre_flare_bg_flux
            cutoff_flux = self.pre_flare_bg_flux + 0.1 * flare_amplitude
            
            if flux <= cutoff_flux or (abs(grad_1st) < 1e-8 and flux < 1e-6):
                self._change_state("Quiescent", current_time)
                self.pre_flare_bg_flux = flux
            elif grad_1st > self.init_gradient_threshold:
                # Re-initiation during decay (double peak)
                self._change_state("Initiation", current_time)
                self.pre_flare_bg_flux = flux
                
        return self.current_state

    def track_series(self, df: pd.DataFrame, flux_col: str = "soft_xray_flux", time_col: str = "time") -> pd.DataFrame:
        """
        Processes a full dataframe of sorted time-series observations,
        computing derivatives and tracking state transitions for the whole series.
        """
        df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
        n = len(df_sorted)
        
        states = []
        grad_1st_list = []
        grad_2nd_list = []
        
        # Precompute derivatives
        times = pd.to_datetime(df_sorted[time_col])
        fluxes = df_sorted[flux_col].fillna(1e-9).values
        
        # compute dt in minutes
        dt = times.diff().dt.total_seconds().fillna(60.0).values / 60.0
        
        # dF/dt
        grad_1st = np.zeros(n)
        for i in range(1, n):
            grad_1st[i] = (fluxes[i] - fluxes[i - 1]) / max(0.1, dt[i])
            
        # d2F/dt2
        grad_2nd = np.zeros(n)
        for i in range(2, n):
            grad_2nd[i] = (grad_1st[i] - grad_1st[i - 1]) / max(0.1, dt[i])
            
        # Run state machine
        if n > 0:
            self.reset(pd.to_datetime(times.iloc[0]), fluxes[0])
            
        for i in range(n):
            t = pd.to_datetime(times.iloc[i])
            f = fluxes[i]
            g1 = grad_1st[i]
            g2 = grad_2nd[i]
            
            state = self.update(t, f, g1, g2)
            states.append(state)
            grad_1st_list.append(g1)
            grad_2nd_list.append(g2)
            
        res_df = df_sorted.copy()
        res_df["flux_gradient"] = grad_1st_list
        res_df["flux_acceleration"] = grad_2nd_list
        res_df["lifecycle_state"] = states
        return res_df
