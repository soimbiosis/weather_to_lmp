# Where does weather matter? Isolating the role of weather in California energy markets

Weather is often used as an exogenous input in energy modeling, but its independent contribution has not been systemati- cally explored. It is well understood that weather drives energy supply and demand, which in turn drive market outcomes [1]. However, existing approaches typically follow one of two paradigms: they either model market outcomes from system state variables, or use a multi-modal suite of inputs, including weather, to regress locational marginal pricing (LMP) directly [2] [3]. This study isolates and characterizes the weather-dependent signal in downstream energy market variability. We find that system state variables are weather-driven, and exhibit nonlinear temporal structure that rewards sequence-based modeling. However, forecasting market-level outcomes requires conditioning on system state rather than raw meteorological inputs.

## Key Findings

- Weather alone has little predictive power for locational marginal
  pricing (LMP) and its components (congestion, energy cost).

- Calendar regimes capture most of the available signal in this setup
  when forecasting congestion spikes

- Net load ramp is substantially weather driven, with nonlinear temporal
  structure. A transformer model improves on the performance of an HGB
  tree at all horizons in the 1-6h range, with the performance gap
  widening at longer horizons.

## Setup

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

Please see [the study writeup](weather_energy_study_summary.pdf) for a breakdown of the experiments
conducted, interpreted results and figures.
