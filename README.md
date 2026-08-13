Weather is often used as an exogenous input in energy modeling, but its independent contribution has not been systemati- cally explored. It is well understood that weather drives energy supply and demand, which in turn drive market outcomes [1]. However, existing approaches typically follow one of two paradigms: they either model market outcomes from system state variables, or use a multi-modal suite of inputs, including weather, to regress locational marginal pricing (LMP) directly [2] [3]. This study isolates and characterizes the weather-dependent signal in downstream energy market variability. We find that system state variables are weather-driven, and exhibit nonlinear temporal structure that rewards sequence-based modeling. However, forecasting market-level outcomes requires conditioning on system state rather than raw meteorological inputs.

# Key Findings

- Weather alone has little predictive power for locational marginal
  pricing (LMP) and its components (congestion, energy cost).

- Calendar regimes capture most of the available signal in this setup
  when forecasting congestion spikes

- Net load ramp is substantially weather driven, with nonlinear temporal
  structure. A transformer model improves on the performance of an HGB
  tree at all horizons in the 1-6h range, with the performance gap
  widening at longer horizons.

# Setup

California Independent System Operator (CAISO) was used as the primary
data source for energy market data. From CAISO, via the gridstatus API,
LMP per hub (SP15, NP15, and ZP26) was obtained, as well as regional
load and fuel mix generation data, from 2023-01-01 to 2026-01-01. These
were preprocessed to 1h resolution. 2023 and 2024 were used for
training, and 2025 was held out for testing.\
ERA5 hourly weather data was obtained from CDS, including ssrd, fdir,
tcc, tp, t2m, and tcwv. The nearest ERA5 cell to each CAISO hub was
identified, and a spatial average over a 3x3 ERA5 patch was taken for
each variable.\
In addition to the raw weather features, lag and rolling window features
were computed, as well as time features which included sin/cos encodings
of hour of day, day of year, and day of week.

Please see [the study writeup](weather_energy_summary_v3.pdf) for a breakdown of the experiments conducted
and the results and figures.

# Interpretation

The results reveal a clear asymmetry between the role of weather in the
physical and market layers of the power system. Market outcomes in the
day-ahead setting (LMP, congestion, energy cost) are dominated by
temporal and institutional structure - transmission limits, bidding
structure, grid topology, etc - and weather adds little useful signal.
The physical layer (load, net load ramp) is weather-driven, and the
relationship is defined by nonlinear temporal dynamics. This suggests
that weather is a useful input for forecasting system state, but not a
sufficient signal for down- stream market outcomes, which rely on
context beyond the physical layer. In this study we utilized only
historical weather reanalysis data, to avoid the uncertainty introduced
by forecast weather, but from this, we can hypothesize that downstream
models incorporating weather forecasts might be better served by
conditioning on weather-forecast-driven system state rather than raw
weather forecast data.

# Future Directions

- Learn a structured intermediate state representation which feeds into
  downstream market forecasting models. Weather does not fully determine
  the intermediate system state, but when coupled with grid constraints
  such as outages, transmission limits etc, a learned latent
  representation could provide a spatially dense and far richer input to
  downstream market layer forecasting models than load or generation,
  which are spatially coarse and incomplete projections of the drivers
  of market dynamics.

- Incorporate Weather forecasts - this study intentionally restricted
  the weather inputs to historical reanalysis data to avoid the
  uncertainty introduced by forecast weather. In practice, however,
  day-ahead market forecasting typically relies on forecast weather
  data. A natural extension of this study would be to quantify
  introduced uncertainty from weather forecasts, derive forecast system
  state from weather forecasts and compare effectiveness of conditioning
  on weather forecasts vs weather-driven system state forecasts.

- Multi-modal inputs and outputs - expanding to more than just weather
  data, as weather is not the primary driver in a lot of these cases.
  Deep learning makes more sense as the complexity of interactions
  increases, both in the input and output layers.
