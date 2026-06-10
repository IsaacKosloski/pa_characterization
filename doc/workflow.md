# README for the workflows
## Workflow 01: Power Amplifier (PA) Linearization via Genetic Algorithm and DPD
### Project Overview
This project aims to compensate for the non-linearities and memory effects of an RF Power Amplifier (PA) operating in the saturation region. To achieve this, a Digital Pre-Distorter (DPD) ws designed using a polynomial behavioral model with memory (NARX/Volterra), whose inverse coefficients are optimized through a *Genetic Algorithm (GA)*.

### Phase 01: Understanding the Physical Model (The Plant)
The starting point was the analysis of a set of mathematical coefficients that describe the PA's behavior:

- **Intercepts (Bias)**:Represent the DC offset of the system.
- **Autoregressive Matrices ($A_1, A_2$)**: Model the amplifier's memory (delays $t -1$ and $ t - 2$), capturing the hardware's thermal and reactive effects.
- **Exogenous Variables (Betas)**: Model the base linear gain and non-linear distortions.
  - *Lags*: Dependency on the input signal at past time steps.
  - *Degrees (Degrees 2 and 3)*: Intermodulation distortions (e.g., IMD3) that compress the signal at high power levels.

### Phase 02: Data Preparation and Partitioning
Based on the raw CSV data (`X_real`, `X_img`, `Y_real`, `Y_img`), the data ingestion pipeline was established.

- **The Golden Rule for Times Series**: Because the model has memory (depends on previous samples), the data split between Training (70%) and Validation (30%) was done sequentially (whithout shuffing/ `shuffle=Flase`).
- If the data were shuffled, the signal's timeline would be destroyed, making it impossible to calculate the delay matrices ($A_1, A_2$).

### Phase 03: The Inversion Strategy (ILA Architecture)
To allow the PA to operate efficiently in the saturation zone, the Indirect Learning Architecture (ILA) was adopted.

- Instead of predicting the distortion (Forward Model: $X \rightarrow Y$), the goal shifted to crating the antidote (Inverse Model: $Y \rightarrow X$).
- The Training dataset was manipulated so that the optimizer's input was the distorted signal ($Y$) and the target to be reachd was ideal signal ($X$).

### Phase 04: Optimization with Genetic Algorithm (GA)
To find the optimal coefficients for the inverse filter, an evolutionary engine was built using the DEAP
Python library.

1. **Initial Population**: 50 individuals created from a mutation based on the original PA coefficients. Each individual represents a "candidate DPD model" containing the 46 betas/matrices.
2. **Fitness Function**: Evaluates each individual by applying its equation to signal $Y$ and calculating the RMSE (Root Mean Square Error) against the ideal signal $AX_{target}$.
3. **Evolution**: Over $N$ generations, the algorithm applies crossover and mutations to the best RMSE, escaping local minima iun the non-linear error surface.
4. **Result**: Extraction of the `best_individual`, which contains the final DPD coefficient matrix.

### Phase 05: The Digital Twin (Code Simulation)
To validate the model without needing physical hardware, a temporal simulation script (Digital Twin) was created.

- **Sample-by-Sample NARX Engine**: Due to the feedback loops ($Y$ depending on $Y_{t-1}$), the simulation could not be done with simples vectorized operations. A time loop `for t in rang(...)` was implemented to perfectly emulate the hardware's clock cycles.
- **The Signal Pipeline**:
  1. Generation of a baseband signal (e.g., Gaussian Noise / OFDM).
  2. Passing the signal through the DPD filter (intentionally expanding the signal). 
  3. Passing the pre-distorted signal through the PA Plant (compressing the signal in saturation).
  4. Obtaining the linearized signal at the antenna.

### Phase 06: Metric Evaluation and Business Value
The final phase focuses on translating the mathematical error (RMSE) into standard Telecommunications industry metrics to prove the solution's effectiveness:

- **Quantitative Metrics**:
  - **EVM (Erro Vector Magnitude)**: Measure in percentage (%), it evaluates modulation quality and distortion in the IQ constellation.
  - **NMSE (Normalized Mean Square Error)**: Measured in decibels (dB), it quantifies the error in the time domain (typical target: -35 to -45 db).
- **Critical Visualizations**:
  - **AM-AM Curve**: Plots input versus output amplitude, visually demonstrating the compression correction (the graph becomes a straight linear line again).
  - **IQ Constellation Diagram**: Shows symbol clustering, proving the reduction of point cloud scattering.
  - **PSD Spectrum (Power Spectral Density): Demonstrates the suppression of Spectral Regrowth and the improvement of ACLR, ensuring the compliance with regulatory standards (Anatel/FCC) by not interfering with adjacent channels.