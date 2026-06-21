%% ========================================================================
% Policy gradient HVAC with integral control, saturation and disturbances
% - controls deviation from weather-dependent steady state
% - uses u = u_ss + K*[x-x_ss; z] (Feedback + Feedforward control)
% - steady-state weather equilibrium is computed from identified model
% =========================================================================

clear; clc; close all;

T_sp = 20;   % Setpoint [deg C]

% Reproducibility
rng(1);

%% ========================================================================
% Load weather data
% =========================================================================

filename = 'Alle datasæt.xlsx';
sheet    = '2012_2023';

weather = load_weather_data(filename, sheet);

Phi_d_data = weather.Phi_d_data;
Phi_D_data = weather.Phi_D_data;
Tout_data  = weather.Tout_data;
Tvent_data = weather.Tvent_data;

N_weather = length(Tout_data);
N = numel(Tout_data);

% Absolute time vector in seconds
t_data_sec = (0:N-1)'*3600;   % [s] since dataset start

% Choose a 1-year window from the dataset start (8760 hours)
year_hours = min(365*24, N-1);
Tfinal_sec = year_hours * 3600;


%% ========================================================================
% Parameters
% =========================================================================

params = build_hvac_parameters();

% For readability, unpack frequently used parameters
A_g = params.A_g;
A_gl = params.A_gl;
g_window = params.g_window;
f_shade  = params.f_shade;

n_people = params.n_people;
Q_person_W = params.Q_person_W;
Q_equipment_per_person_W = params.Q_equipment_per_person_W;
Q_flatscreen_room_W = params.Q_flatscreen_room_W;


%% ========================================================================
% Continuous-time model
% =========================================================================

[Ac, Bc, E_c, B_full_c] = build_continuous_hvac_model(params);

% Example discretization
Ts = 3600;
sys_d_true = c2d(ss(Ac, B_full_c, [], []), Ts);

A_true = sys_d_true.A;
B_true = sys_d_true.B(:,1);
E_true = sys_d_true.B(:,2:6);

n = size(A_true,1);
m = size(B_true,2);


%% ========================================================================
% Saturation limits
% =========================================================================
% If the required steady-state heating/cooling u_ss is outside these limits,
% offset-free tracking is physically impossible.
%
% Sign convention:
%   positive u : heating
%   negative u : cooling
%
% In the comparison below, the heating limit is kept fixed and the cooling
% lower saturation limit is varied. This makes it possible to show what the
% controller wanted to do without saturation versus what the actuator could
% physically apply.

upper_limit = 500;

% Cooling limits for comparison:
% -500  W : low cooling authority
% -1500 W : medium cooling authority
% -2500 W : approximately saturation-free case

cooling_limits = [-500, -1500, -2500];


%% ========================================================================
% Historical data generation / collection for model identification
% =========================================================================
% For practical implementation, this block would be replaced by measured
% historical data:
%
%   X_id_data = measured temperatures
%   U_id_data = measured HVAC power
%   D_id_data = measured disturbances [T_out; T_vent; Q_sun]
%
% Here, the true model is used only to generate synthetic historical data.

N_id = min(365*24, N_weather); % number of historical samples for identification
sigma_u_id = 400;              % probing amplitude for historical data [W]
sigma_id_noise = 0.1;          % measurement/process noise for ID data

lower_limit_id = min(cooling_limits);

[X_id_data, U_id_data, D_id_data] = generate_identification_data( ...
    A_true, B_true, E_true, weather, params, ...
    N_id, sigma_u_id, sigma_id_noise, lower_limit_id, upper_limit);


%% ========================================================================
% Estimate A, B, E and affine bias c from historical data
% =========================================================================
% Identified model:
%   x_{k+1} = A_id*x_k + B_id*u_k + E_id*d_k + c_id
% The feedforward steady-state calculation will use this model only.

lambda_id = 1e-6;

[A_id, B_id, E_id, c_id, X1_id_pred, rmse_id] = ...
    estimate_hvac_model(X_id_data, U_id_data, D_id_data, lambda_id);

fprintf('\n===== Identified model diagnostics =====\n');
fprintf('RMSE T_room = %.4f deg C\n', rmse_id(1));
fprintf('RMSE T_wo   = %.4f deg C\n', rmse_id(2));
fprintf('RMSE T_wi   = %.4f deg C\n', rmse_id(3));
fprintf('rho(A_id)   = %.4f\n', max(abs(eig(A_id))));
fprintf('rho(A_true) = %.4f\n', max(abs(eig(A_true))));


%% ========================================================================
% Use identified model for controller and feedforward design
% =========================================================================

A = A_id;
B = B_id;
E = E_id;
c_aff = c_id;

%% ========================================================================
% Discrete model augmented with integral action
% =========================================================================
% The controlled state is:
%   e = x - x_ss(weather), and since x_ss(1) = T_sp,
%   z_{k+1} = z_k + T_sp - T_room = z_k - e_1
%
% This augmented model is based on the identified model.

C_track = [1 0 0];

A_aug = [A zeros(n,1);
        -C_track 1];

B_aug = [B;
         0];

n_aug = n + 1;

% Ground-truth augmented model for monitoring only
A_aug_true = [A_true zeros(n,1);
             -C_track 1];

B_aug_true = [B_true;
              0];


%% ========================================================================
% Cost
% =========================================================================

Q = diag([100, 1, 1, 500]);  % integral penalty is 500
R = 1e-2;


%% ========================================================================
% Initial controller
% =========================================================================
% K0 is designed from the identified model, not the true model.

K0 = -dlqr(A_aug, B_aug, Q, R);

fprintf('\n===== Controller initialization =====\n');
fprintf('Initial spectral radius rho(A_id_aug+B_id_aug*K0) = %.4f\n', ...
    max(abs(eig(A_aug+B_aug*K0))));
fprintf('Initial spectral radius rho(A_true_aug+B_true_aug*K0) = %.4f\n', ...
    max(abs(eig(A_aug_true+B_aug_true*K0))));


%% ========================================================================
% Initial condition
% =========================================================================

x0 = [25; 20; 18];  % real temperatures [T_room, T_wo, T_wi]
z0 = 0;             % integrator


%% ========================================================================
% Offline data to seed covariance parameterization
% =========================================================================
% This is based on the identified deviation model:
%   e_{k+1} = A_id*e_k + B_id*du_k
% The true model is not used here.

t0 = min(365*24, N_weather);
sigma_u_off = 50;    % probing noise during data collection

offline = generate_offline_PG_data( ...
    A, B, E, c_aff, C_track, T_sp, K0, x0, z0, ...
    weather, params, t0, sigma_u_off);

X0_off = offline.X0_off;
X1_off = offline.X1_off;
U0_off = offline.U0_off;
D0_off = offline.D0_off;

rank_D0 = rank(D0_off);
fprintf('\nrank(D0_off) = %d, required = %d\n', rank_D0, m+n_aug);


%% ========================================================================
% Sample covariances and blocks
% =========================================================================

Phi = (D0_off*D0_off.')/t0;
X0c = (X0_off*D0_off.')/t0;
U0c = (U0_off*D0_off.')/t0;
X1c = (X1_off*D0_off.')/t0;
Phi_inv = pinv(Phi);


%% ========================================================================
% Initialize PG
% =========================================================================

eta = 1e-6;
T_online = min(365*24, N_weather);
sigma_v0 = 5;
sigma_w = 0;
alpha = 0.6;

% Learning options
% If true, PG uses x_next - x_ss(k) for the covariance update. This is
% closer to an LTI deviation model when weather-dependent steady states move.
use_same_step_ss_for_PG = true;

% If true, saturated samples are not used in the policy-gradient update.
% For the actuator-authority comparison, this is kept false so that the
% data reflect the actual clipped actuator behavior.
skip_saturated_samples = false;


%% ========================================================================
% Multi-case PG simulations for different cooling saturation limits
% =========================================================================

case_results = cell(numel(cooling_limits),1);

for ii = 1:numel(cooling_limits)

    lower_limit = cooling_limits(ii);

    fprintf('\n\n============================================================\n');
    fprintf('Running cooling saturation case: lower_limit = %.0f W\n', lower_limit);
    fprintf('============================================================\n');

    sim_options.lower_limit = lower_limit;
    sim_options.upper_limit = upper_limit;
    sim_options.eta = eta;
    sim_options.T_online = T_online;
    sim_options.sigma_v0 = sigma_v0;
    sim_options.sigma_w = sigma_w;
    sim_options.alpha = alpha;
    sim_options.Ts = Ts;

    sim_options.use_same_step_ss_for_PG = use_same_step_ss_for_PG;
    sim_options.skip_saturated_samples = skip_saturated_samples;

    case_results{ii} = run_PG_hvac_case( ...
        A_true, B_true, E_true, ...
        A, B, E, c_aff, ...
        A_aug_true, B_aug_true, ...
        A_aug, B_aug, ...
        C_track, T_sp, ...
        Q, R, K0, ...
        x0, z0, ...
        weather, params, ...
        X0c, U0c, X1c, Phi_inv, ...
        t0, sim_options);

end


%% ========================================================================
% Plots: model identification quality
% =========================================================================

plot_identification_quality(X_id_data, X1_id_pred);


%% ========================================================================
% Plots: single-case detailed diagnostics
% =========================================================================
% Choose medium cooling authority case for detailed state/input plots.

idx_detail = find(cooling_limits == -1500, 1);
if isempty(idx_detail)
    idx_detail = 1;
end

plot_single_case_diagnostics(case_results{idx_detail}, T_sp);


%% ========================================================================
% Plots: cooling saturation comparison
% =========================================================================
% This figure compares what the controller wanted to apply without
% saturation, u_unsat, versus what it could actually apply, u, for three
% cooling limits:
%
%   -500 W  : low control authority
%   -1500 W : medium control authority
%   -2500 W : saturation-free or nearly saturation-free case

plot_cooling_saturation_comparison(case_results, cooling_limits, upper_limit, T_sp);
plot_tracking_error_comparison(case_results, cooling_limits);

% Frobenius norm comparison for learned controllers under different
% cooling saturation limits.
plot_frobenius_norm_comparison(case_results, cooling_limits);

% Zoomed inset plots for the detailed case.
% Here, idx_detail is the medium-authority case, i.e. -1500 W.
zoom_days = 5;

plot_tracking_error_with_inset(case_results{idx_detail}, T_sp, zoom_days);
plot_integral_action_with_inset(case_results{idx_detail}, zoom_days);
plot_sat_tracking_error_comparison(case_results, cooling_limits, [100 260]);
plot_cooling_clipping_comparison(case_results, cooling_limits, [100 260]);
plot_simplified_year_result(case_results{idx_detail}, T_sp);
plot_PG_eigenvalue_comparison(case_results, cooling_limits);
plot_PG_spectral_radius_comparison(case_results, cooling_limits);
plot_PG_stability_combined(case_results, cooling_limits);


%% ========================================================================
% Final closed-loop diagnostics
% =========================================================================

print_case_diagnostics(case_results, cooling_limits, T_sp, Ts);


% =========================================================================
% =========================================================================
% =========================================================================
%% ========================================================================
%  Weather data loader
% =========================================================================

function weather = load_weather_data(filename, sheet)

    D = readtable(filename, 'Sheet', sheet, 'TextType', 'string');

    % Keep rows where col 1 looks like YYYY-MM-DD-HH
    time_col = datetime(D{:,1}, ...
        'InputFormat','yyyy-MM-dd-HH', ...
        'TimeZone','UTC', ...
        'Format','yyyy-MM-dd HH');

    isTimeRow = ~ismissing(time_col);
    D = D(isTimeRow, :);

    % Extract columns
    Phi_d_data = D{:,6};
    Phi_D_data = D{:,8};
    Tout_data  = D{:,18};
    Tvent_data = D{:,21};

    % Basic sanitization
    Phi_d_data(isnan(Phi_d_data)) = 0;
    Phi_D_data(isnan(Phi_D_data)) = 0;
    Tout_data(isnan(Tout_data))   = mean(Tout_data,'omitnan');
    Tvent_data(isnan(Tvent_data)) = mean(Tvent_data,'omitnan');

    weather.Phi_d_data = Phi_d_data;
    weather.Phi_D_data = Phi_D_data;
    weather.Tout_data  = Tout_data;
    weather.Tvent_data = Tvent_data;
    weather.N_weather  = length(Tout_data);

end


%% ========================================================================
%  Build HVAC parameters
% =========================================================================

function params = build_hvac_parameters()

    %% Parameters
    % Geometry and areas
    d_room = 5;   % [m]
    w_room = 5;   % [m]
    h_room = 3;   % [m]

    V_air  = d_room * w_room * h_room;     % [m^3]
    A_facade = h_room * w_room;            % [m^2]
    A_wo_ratio = 0.7;                      % share of facade that is opaque wall
    A_wo = A_facade * A_wo_ratio;          % external wall [m^2]
    A_window = A_facade - A_wo;            % window [m^2]
    A_wi = h_room * (2*d_room + w_room);   % internal walls (room side) [m^2]
    A_floor = 2*d_room*w_room;             % Floor [m^2]

    % Solar glass areas
    A_g = 0.9 * A_window;   % total glass area
    A_gl = 0.9 * A_window;  % illuminated glass area

    % Properties & resistances
    % Air
    rho_air = 1.204;    % [kg/m^3]
    cp_air = 1005;      % [J/(kg*K)]

    % Furniture (lumped capacity)
    C_furn_lumped = 5e5;             % [J/K]

    % Outer wall materials
    rho_insul = 37;
    cp_insul = 1030;
    rho_brick = 1900;
    cp_brick = 960;

    lambda_insul_wo = 0.034;  % [W/(m*K)]
    lambda_brick = 0.62;      % [W/(m*K)]
    d_insul_wo = 0.24;        % [m]
    d_brick = 0.109;          % [m]

    % Inner wall materials
    rho_gypsum = 650;
    cp_gypsum = 1000;
    lambda_insul_wi = 0.037;
    d_insul_wi = 0.045;
    lambda_gypsum = 0.16;
    d_gypsum = 0.025;

    % Floor materials
    rho_concrete = 2300;
    cp_concrete = 880;
    lambda_concrete = 1.7;
    d_concrete = 0.15;

    % Convective resistances
    R_conv_out = 0.04;     % [(m^2*K)/W]
    R_conv_in = 0.13;      % [(m^2*K)/W]
    R_conv_in_down = 0.17; % [(m^2*K)/W]

    % Conductive symmetrical wall resistances per m^2
    R_cond_out_wo = d_brick/lambda_brick + 0.5*d_insul_wo/lambda_insul_wo;
    R_cond_wo_room = R_cond_out_wo;    % symmetrical outer wall

    R_cond_room_wi = d_gypsum/lambda_gypsum + 0.5*d_insul_wi/lambda_insul_wi;
    R_cond_wi_in = R_cond_room_wi;   % symmetrical inner wall

    R_cond_room_floor = 0.5*d_concrete/lambda_concrete;
    R_cond_floor_in = R_cond_room_floor;   % symmetrical floor

    % Sum with convective resistances
    R_out_wo = R_conv_out + R_cond_out_wo;     % [ (m^2*K)/W ]
    R_wo_room = R_conv_in  + R_cond_wo_room;
    R_room_wi = R_conv_in  + R_cond_room_wi;
    R_wi_in = R_conv_in  + R_cond_wi_in;
    R_room_floor = R_conv_in_down + R_cond_room_floor;
    R_floor_in = R_conv_in_down + R_cond_floor_in;

    % Capacities (J/K)
    % Room node (air + furniture lumped)
    C_air = rho_air*cp_air*V_air;                 % [J/K]
    C_room = C_air + C_furn_lumped;               % [J/K]

    % Outer wall
    V_insul_wo = A_wo * d_insul_wo;
    V_brick = A_wo * d_brick * 2;               % two brick layers
    C_wo = rho_insul*cp_insul*V_insul_wo + rho_brick*cp_brick*V_brick;

    % Inner wall
    V_insul_wi = A_wi * d_insul_wi;
    V_gypsum = A_wi * d_gypsum * 2;              % two gypsum boards
    V_concrete = A_floor * d_concrete;
    C_wi = rho_insul*cp_insul*V_insul_wi + rho_gypsum*cp_gypsum*V_gypsum + rho_concrete*cp_concrete*V_concrete;

    % Window & solar
    U_window = 1.1;     % [W/(m^2*K)]
    g_window = 0.5;
    f_shade = 1.0;     % possible control between 0 & 1

    % Ventilation / infiltration
    n_total_ach = 0.5; % [1/h] total airchange (possible control variable)
    n_infil_ach = 0.1;
    n_vent_ach = n_total_ach; % possible control variable

    q_infil = n_infil_ach * V_air / 3600;   % [m^3/s]
    q_vent = n_vent_ach  * V_air / 3600;    % [m^3/s]

    % CO2 parameters
    ci = 400e-6;     % [m^3/m^3] supply/outdoor CO2
    q_people = 0.02; % [m^3/h] (can be scheduled by time if needed)
    n_CO2 = n_total_ach; % [1/h]

    n_people = 3;
    Q_person_W = 100;
    Q_equipment_per_person_W = 80;
    Q_flatscreen_room_W = 100;

    % Conductances [W/K]
    G_out_wo = A_wo / R_out_wo;
    G_wo_room = A_wo / R_wo_room;
    G_room_wi = A_wi / R_room_wi + A_floor / R_room_floor;
    G_wi_in = A_wi / R_wi_in + A_floor / R_floor_in;

    G_window = U_window * A_window;
    G_vent = rho_air * cp_air * q_vent;
    G_infil = rho_air * cp_air * q_infil;

    T_vent_min = 18;

    % Since T_vent_air = T_out, ventilation, infiltration and window losses
    % all connect room to outside.
    G_room_out = G_window + G_infil;

    % Solar split fractions
    A_total = A_wo + A_wi + A_floor;

    f_sun_room = 2/3;
    f_sun_wo = (1/3) * (A_wo / A_total);
    f_sun_wi = (1/3) * ((A_wi + A_floor) / A_total);

    f_people_room = 6/10;
    f_people_wo = (4/10) * (A_wo / A_total);
    f_people_wi = (4/10) * ((A_wi + A_floor) / A_total);

    f_equip_room = 3/4;
    f_equip_wo = (1/4) * (A_wo / A_total);
    f_equip_wi = (1/4) * ((A_wi + A_floor) / A_total);

    params.d_room = d_room;
    params.w_room = w_room;
    params.h_room = h_room;
    params.V_air = V_air;
    params.A_facade = A_facade;
    params.A_wo = A_wo;
    params.A_window = A_window;
    params.A_wi = A_wi;
    params.A_floor = A_floor;
    params.A_g = A_g;
    params.A_gl = A_gl;

    params.rho_air = rho_air;
    params.cp_air = cp_air;
    params.C_room = C_room;
    params.C_wo = C_wo;
    params.C_wi = C_wi;

    params.U_window = U_window;
    params.g_window = g_window;
    params.f_shade = f_shade;

    params.n_total_ach = n_total_ach;
    params.n_infil_ach = n_infil_ach;
    params.n_vent_ach = n_vent_ach;
    params.q_infil = q_infil;
    params.q_vent = q_vent;
    params.T_vent_min = T_vent_min;

    params.ci = ci;
    params.q_people = q_people;
    params.n_CO2 = n_CO2;

    params.n_people = n_people;
    params.Q_person_W = Q_person_W;
    params.Q_equipment_per_person_W = Q_equipment_per_person_W;
    params.Q_flatscreen_room_W = Q_flatscreen_room_W;

    params.G_out_wo = G_out_wo;
    params.G_wo_room = G_wo_room;
    params.G_room_wi = G_room_wi;
    params.G_wi_in = G_wi_in;
    params.G_window = G_window;
    params.G_vent = G_vent;
    params.G_infil = G_infil;
    params.G_room_out = G_room_out;

    params.f_sun_room = f_sun_room;
    params.f_sun_wo = f_sun_wo;
    params.f_sun_wi = f_sun_wi;

    params.f_people_room = f_people_room;
    params.f_people_wo = f_people_wo;
    params.f_people_wi = f_people_wi;

    params.f_equip_room = f_equip_room;
    params.f_equip_wo = f_equip_wo;
    params.f_equip_wi = f_equip_wi;

end


%% ========================================================================
%  Build continuous-time HVAC model
% =========================================================================

function [Ac, Bc, E_c, B_full_c] = build_continuous_hvac_model(params)

    C_room = params.C_room;
    C_wo = params.C_wo;
    C_wi = params.C_wi;

    G_out_wo = params.G_out_wo;
    G_wo_room = params.G_wo_room;
    G_room_wi = params.G_room_wi;
    G_wi_in = params.G_wi_in;
    G_room_out = params.G_room_out;
    G_vent = params.G_vent;

    f_sun_room = params.f_sun_room;
    f_sun_wo = params.f_sun_wo;
    f_sun_wi = params.f_sun_wi;

    f_people_room = params.f_people_room;
    f_people_wo = params.f_people_wo;
    f_people_wi = params.f_people_wi;

    f_equip_room = params.f_equip_room;
    f_equip_wo = params.f_equip_wo;
    f_equip_wi = params.f_equip_wi;

    % -------------------------------------------------------------------------
    % Continuous-time A matrix
    % -------------------------------------------------------------------------
    %
    % T_room dynamics:
    %   C_room*dT_room =
    %       u + G_wo_room*(T_wo - T_room) - G_room_wi*(T_room - T_wi)
    % + G_room_out*(T_out - T_room) + f_sun_room*Q_sun + f_people_room*Q_people
    % + f_equip_room*Q_equipment + G_vent * (Tvent_data - T_room)
    %
    % T_wo dynamics:
    %   C_wo*dT_wo =
    %       G_out_wo*(T_out - T_wo)
    %     - G_wo_room*(T_wo - T_room)
    %     + f_sun_wo*Q_sun + f_people_wo*Q_people + f_equip_wo*Q_equipment
    %
    % T_wi dynamics:
    %   C_wi*dT_wi =
    %       G_room_wi*(T_room - T_wi)
    %     - G_wi_in*(T_wi - T_room)
    %     + f_sun_wi*Q_sun + f_people_wi*Q_people + f_equip_wi*Q_equipment
    %
    % Since T_adjacent = T_room, the last term becomes
    %   -G_wi_in*(T_wi - T_room)
    %
    % -------------------------------------------------------------------------

    Ac = [ ...
        -(G_wo_room + G_room_wi + G_room_out + G_vent)/C_room,   G_wo_room/C_room,              G_room_wi/C_room;
         G_wo_room/C_wo,                                        -(G_out_wo + G_wo_room)/C_wo,   0;
        (G_room_wi + G_wi_in)/C_wi,                              0,                            -(G_room_wi + G_wi_in)/C_wi ...
    ];

    % HVAC input enters room node only
    Bc = [1/C_room;
          0;
          0];

    % Disturbance matrix for d = [T_out; T_vent; Q_sun; Q_people; Q_equipment]
    E_c = [ ...
        G_room_out/C_room,     G_vent/C_room,     f_sun_room/C_room,      f_people_room/C_room,      f_equip_room/C_room;
        G_out_wo/C_wo,         0,                 f_sun_wo/C_wo,          f_people_wo/C_wo,          f_equip_wo/C_wo;
        0,                     0,                 f_sun_wi/C_wi,          f_people_wi/C_wi,          f_equip_wi/C_wi ...
    ];

    % Full continuous-time input matrix:
    % columns correspond to [u; T_out; T_vent; Q_sun; Q_people; Q_equipment]
    B_full_c = [Bc E_c];

end


%% ========================================================================
%  occupancy schedule and internal gains
% =========================================================================

function [Q_people, Q_equipment, occ_frac, equip_frac] = occupancy_schedule(hour, params)

    n_people = params.n_people;

    if n_people <= 2

        if hour >= 8 && hour <= 10
            occ_frac = 1.0;
        elseif hour == 11
            occ_frac = 0.5;
        elseif hour >= 12 && hour <= 15
            occ_frac = 1.0;
        else
            occ_frac = 0.0;
        end

        if hour >= 8 && hour <= 15
            equip_frac = 1.0;
        else
            equip_frac = 0.0;
        end

    else

        if hour == 7
            occ_frac = 0.4;
        elseif hour >= 8 && hour <= 10
            occ_frac = 0.8;
        elseif hour == 11
            occ_frac = 0.4;
        elseif hour >= 12 && hour <= 15
            occ_frac = 0.8;
        elseif hour == 16
            occ_frac = 0.4;
        else
            occ_frac = 0.0;
        end

        if hour == 7
            equip_frac = 0.4;
        elseif hour >= 8 && hour <= 15
            equip_frac = 0.8;
        elseif hour == 16
            equip_frac = 0.4;
        else
            equip_frac = 0.0;
        end

    end

    N_eff = n_people * occ_frac;

    Q_people = params.Q_person_W * N_eff;

    Q_equipment_full = ...
        params.Q_equipment_per_person_W*n_people + ...
        params.Q_flatscreen_room_W;

    Q_equipment = equip_frac * Q_equipment_full;

end


%% ========================================================================
%  weather/disturbance generation
% =========================================================================

function d_input = get_disturbance(k, weather, params)

    N_weather = weather.N_weather;

    idx = mod(k-1, N_weather) + 1;

    T_out  = weather.Tout_data(idx);
    T_vent = weather.Tvent_data(idx);
    Phi_d  = weather.Phi_d_data(idx);
    Phi_D  = weather.Phi_D_data(idx);

    Q_sun = (params.A_g*Phi_d + params.A_gl*Phi_D) * ...
        params.g_window * params.f_shade;

    hour = mod(idx-1, 24);

    [Q_people, Q_equipment] = occupancy_schedule(hour, params);

    d_input = [T_out; T_vent; Q_sun; Q_people; Q_equipment];

end


%% ========================================================================
%  generate historical identification data
% =========================================================================

function [X_id_data, U_id_data, D_id_data] = generate_identification_data( ...
    A_true, B_true, E_true, weather, params, ...
    N_id, sigma_u_id, sigma_id_noise, lower_limit, upper_limit)

    n = size(A_true,1);
    m = size(B_true,2);

    X_id_data = zeros(n, N_id+1);
    U_id_data = zeros(m, N_id);
    D_id_data = zeros(5, N_id);

    x_id = [22; 19; 18]; % historical initial condition
    X_id_data(:,1) = x_id;

    for k = 1:N_id

        d_input = get_disturbance(k, weather, params);

        % Historical probing input
        u_id = sigma_u_id*randn(m,1);
        u_id = min(max(u_id, lower_limit), upper_limit);

        % Ground truth historical evolution from the unknown dynamics
        x_id_next = A_true*x_id + B_true*u_id + E_true*d_input ...
                    + sigma_id_noise*randn(n,1);

        % Data collected
        X_id_data(:,k+1) = x_id_next;
        U_id_data(:,k) = u_id;
        D_id_data(:,k) = d_input;

        x_id = x_id_next;

    end

end


%% ========================================================================
%  generate offline PG initialization data
% =========================================================================

function offline = generate_offline_PG_data( ...
    A, B, E, c_aff, C_track, T_sp, K0, x0, z0, ...
    weather, params, t0, sigma_u_off)

    n = size(A,1);
    m = size(B,2);
    n_aug = n + 1;

    X = zeros(n_aug, t0+1);
    U = zeros(m, t0);

    % Nominal weather for offline initialization
    T_out_nom = mean(weather.Tout_data,'omitnan');
    T_vent_nom = mean(weather.Tvent_data,'omitnan');
    Phi_d_nom = mean(weather.Phi_d_data,'omitnan');
    Phi_D_nom = mean(weather.Phi_D_data,'omitnan');

    Q_sun_nom = (params.A_g*Phi_d_nom + params.A_gl*Phi_D_nom)*params.g_window*params.f_shade;

    Q_people_day = zeros(24,1);
    Q_equipment_day = zeros(24,1);

    for hh = 1:24

        hour = mod(hh-1, 24);
        [Q_people_day(hh), Q_equipment_day(hh)] = occupancy_schedule(hour, params);

    end

    Q_people_nom = mean(Q_people_day);
    Q_equipment_nom = mean(Q_equipment_day);

    d_nom = [T_out_nom; T_vent_nom; Q_sun_nom; Q_people_nom; Q_equipment_nom];

    [x_ss_nom, ~] = steady_state_hvac_id(A,B,E,c_aff,C_track,T_sp,d_nom);

    e0 = x0 - x_ss_nom;
    X(:,1) = [e0; z0];

    for t = 1:t0

        U(:,t) = K0*X(:,t) + sigma_u_off*randn(m,1);  % deviation input

        e = X(1:n,t);
        z = X(end,t);

        e_next = A*e + B*U(:,t);
        z_next = z - C_track*e;

        X(:,t+1) = [e_next; z_next];

    end

    X0_off = X(:,1:end-1);
    X1_off = X(:,2:end);
    U0_off = U;
    D0_off = [U0_off; X0_off];

    offline.X0_off = X0_off;
    offline.X1_off = X1_off;
    offline.U0_off = U0_off;
    offline.D0_off = D0_off;

end


%% ========================================================================
%  Run PG HVAC case
% =========================================================================

function result = run_PG_hvac_case( ...
    A_true, B_true, E_true, ...
    A, B, E, c_aff, ...
    A_aug_true, B_aug_true, ...
    A_aug, B_aug, ...
    C_track, T_sp, ...
    Q, R, K0, ...
    x0, z0, ...
    weather, params, ...
    X0c_init, U0c_init, X1c_init, Phi_inv_init, ...
    t0, sim_options)

    lower_limit = sim_options.lower_limit;
    upper_limit = sim_options.upper_limit;
    eta = sim_options.eta;
    T_online = sim_options.T_online;
    sigma_v0 = sim_options.sigma_v0;
    sigma_w = sim_options.sigma_w;
    alpha = sim_options.alpha;

    n = size(A,1);
    m = size(B,2);
    n_aug = n + 1;

    K = K0;
    x = x0;
    z = z0;

    % Storing variables for plotting
    cost_true = zeros(1, T_online + 1);
    rho_hat = zeros(1, T_online + 1);
    rho_true = zeros(1, T_online + 1);
    rho_id = zeros(1, T_online + 1);
    normK = zeros(1, T_online + 1);
    eig_hat_all = nan(n_aug, T_online + 1);

    x_all = zeros(n, T_online + 1);
    e_all = zeros(n, T_online + 1);
    xss_all = zeros(n, T_online + 1);
    z_all = zeros(1, T_online + 1);

    u_all = zeros(m, T_online);
    u_unsat_all = zeros(m, T_online);
    uss_all = zeros(m, T_online);
    du_all = zeros(m, T_online);
    du_unsat_all = zeros(m, T_online);

    err_all = zeros(1, T_online + 1);
    sat_all = zeros(1, T_online);
    sat_upper_all = zeros(1, T_online);
    sat_lower_all = zeros(1, T_online);

    K_hist = zeros(numel(K), T_online + 1);

    x_all(:,1) = x;
    z_all(1) = z;
    err_all(1) = T_sp - x(1);
    K_hist(:,1) = K(:);

    [C0, ~] = true_LQR_cost(A_aug_true,B_aug_true,Q,R,K);
    cost_true(1) = C0;
    rho_true(1) = max(abs(eig(A_aug_true+B_aug_true*K)));
    rho_id(1) = max(abs(eig(A_aug+B_aug*K)));
    normK(1) = norm(K, 'fro');

    covar.X0c = X0c_init;
    covar.U0c = U0c_init;
    covar.X1c = X1c_init;
    covar.Phi_inv = Phi_inv_init;
    covar.t = t0;

    % Initial covariance-parameterized closed-loop matrix
    V = covar.Phi_inv * [K; eye(n_aug)];
    Acl_hat = covar.X1c * V;

    eig_hat_all(:,1) = eig(Acl_hat);
    rho_hat(1) = max(abs(eig_hat_all(:,1)));

    w = randn(n, T_online); % Pregenerate process noise

    %% Online loop implementing PG

    for k = 1:T_online

        % Current disturbances
        d_input = get_disturbance(k, weather, params);

        % Weather-dependent steady state
        % This uses only the identified model A_id, B_id, E_id, c_id.
        % It does NOT use A_true, B_true, E_true.

        [x_ss, u_ss] = steady_state_hvac_id(A,B,E,c_aff,C_track,T_sp,d_input);

        if k == 1
            fprintf('\nInitial required steady-state input from identified model u_ss = %.2f W\n', u_ss);
            if u_ss > upper_limit || u_ss < lower_limit
                warning(['Required steady-state input is outside saturation limits. ', ...
                         'Offset-free tracking is impossible unless actuator limits are changed.']);
            end
        end

        % Augmented error state
        e = x - x_ss;
        x_aug = [e; z];

        % Apply input
        % K acts on error state and returns error input.
        % Actual input is u = u_ss + du.

        sigma_v = sigma_v0 / (k+1)^alpha;
        v = sigma_v * randn(m,1);

        du_unsat = K*x_aug + v;
        u_unsat  = u_ss + du_unsat;

        u = min(max(u_unsat, lower_limit), upper_limit);
        du = u - u_ss;

        if abs(u-u_unsat) > 1e-8
            sat_all(k) = 1; % To record how many times we run out of control
        end

        if u >= upper_limit - 1e-8
            sat_upper_all(k) = 1;
        end

        if u <= lower_limit + 1e-8
            sat_lower_all(k) = 1;
        end

        % System evolution
        % The true physical model is used only here as the simulator.

        x_next = A_true*x + B_true*u + E_true*d_input + sigma_w*w(:,k);

        % Next-step steady state from identified model

        d_input_next = get_disturbance(k+1, weather, params);
        [x_ss_next, ~] = steady_state_hvac_id(A,B,E,c_aff,C_track,T_sp,d_input_next);

        e_next_plot = x_next - x_ss_next;

        % For PG data update, use the same-step steady state if desired.
        % This makes the data closer to an LTI deviation model.

        if sim_options.use_same_step_ss_for_PG
            e_next_data = x_next - x_ss;
        else
            e_next_data = e_next_plot;
        end

        % Integral update with conditional anti-windup:
        % Integrate room-temperature error unless actuator is saturated
        % in the same direction as the error.

        err = T_sp - x(1);

        if (u >= upper_limit && err > 0)
            z_next = z;
        elseif (u <= lower_limit && err < 0)
            z_next = z;
        else
            z_next = z + err;
        end

        x_aug_next_data = [e_next_data; z_next];

        %% For plotting

        x_all(:,k+1) = x_next;
        e_all(:,k) = e;
        xss_all(:,k) = x_ss;
        u_all(:,k) = u;
        u_unsat_all(:,k) = u_unsat;
        uss_all(:,k) = u_ss;
        du_all(:,k) = du;
        du_unsat_all(:,k) = du_unsat;
        z_all(k+1) = z_next;
        err_all(k+1) = T_sp - x_next(1);

        %% Update sample covariances recursively
        % PG data must use deviation input du and augmented deviation state.
        % The deviation state is defined using the identified steady state.

        use_sample_for_learning = true;

        if sim_options.skip_saturated_samples && sat_all(k)
            use_sample_for_learning = false;
        end

        if use_sample_for_learning

            covar = update_PG_covariances(covar, du, x_aug, x_aug_next_data);

            %% Solve V from covariance parameterization

            V = covar.Phi_inv * [K; eye(n_aug)];

            %% PG gradient step

            K_hat   = covar.U0c * V;
            Acl_hat = covar.X1c * V;

            eig_hat_now = eig(Acl_hat);

            eig_hat_all(:,k+1) = eig_hat_now;
            rho_hat(k+1) = max(abs(eig_hat_now));

            % Safety feature. Skip gradient step if A_cl is unstable.

            try
                Sigma = dlyap(Acl_hat, eye(n_aug));
                P = dlyap(Acl_hat', Q + K_hat'*R*K_hat);
            catch
                warning('Acl_hat not stable at step %d; skipping gradient update.', k);

                K = K_hat;

                x = x_next;
                z = z_next;

                [Ct, ~] = true_LQR_cost(A_aug_true,B_aug_true,Q,R,K);
                cost_true(k+1) = Ct;
                rho_true(k+1) = max(abs(eig(A_aug_true+B_aug_true*K)));
                rho_id(k+1) = max(abs(eig(A_aug+B_aug*K)));
                normK(k+1) = norm(K,'fro');
                K_hist(:,k+1) = K(:);

                continue;
            end

            G = 2*(covar.U0c'*R*covar.U0c + covar.X1c'*P*covar.X1c)*V*Sigma;

            M = covar.X0c*covar.X0c.';
            reg = 1e-10*max(trace(M)/n_aug, eps);

            Proj = eye(m + n_aug) - covar.X0c'*((M + reg*eye(n_aug)) \ covar.X0c);

            V = V - eta*(Proj*G);

            K = covar.U0c*V;

            if any(isnan(eig_hat_all(:,k+1)))
                eig_hat_all(:,k+1) = eig_hat_all(:,k);
            end

            if isnan(rho_hat(k+1))
                rho_hat(k+1) = rho_hat(k);
            end

        end

        %% Advance time/state

        x = x_next;
        z = z_next;

        %% Monitoring with ground-truth augmented model

        [Ct, ~] = true_LQR_cost(A_aug_true,B_aug_true,Q,R,K);
        cost_true(k+1) = Ct;
        rho_true(k+1) = max(abs(eig(A_aug_true+B_aug_true*K)));
        rho_id(k+1) = max(abs(eig(A_aug+B_aug*K)));
        normK(k+1) = norm(K,'fro');
        K_hist(:,k+1) = K(:);

    end

    % Fill final steady-state logs

    d_input_final = get_disturbance(T_online, weather, params);
    [x_ss_final, ~] = steady_state_hvac_id(A,B,E,c_aff,C_track,T_sp,d_input_final);

    xss_all(:,end) = x_ss_final;
    e_all(:,end)   = x_all(:,end) - x_ss_final;

    % Outdoor temperature plot
    Tout_plot = zeros(1,T_online+1);
    for k = 1:T_online+1
        idx = mod(k-1, weather.N_weather) + 1;
        Tout_plot(k) = weather.Tout_data(idx);
    end

    result.lower_limit = lower_limit;
    result.upper_limit = upper_limit;
    result.T_online = T_online;

    result.cost_true = cost_true;
    result.rho_hat = rho_hat;
    result.rho_true = rho_true;
    result.rho_id = rho_id;
    result.normK = normK;
    result.K_hist = K_hist;
    result.eig_hat_all = eig_hat_all;

    result.x_all = x_all;
    result.e_all = e_all;
    result.xss_all = xss_all;
    result.z_all = z_all;

    result.u_all = u_all;
    result.u_unsat_all = u_unsat_all;
    result.uss_all = uss_all;
    result.du_all = du_all;
    result.du_unsat_all = du_unsat_all;

    result.err_all = err_all;
    result.sat_all = sat_all;
    result.sat_upper_all = sat_upper_all;
    result.sat_lower_all = sat_lower_all;
    result.Tout_plot = Tout_plot;

end


%% ========================================================================
%  PG covariance update
% =========================================================================

function covar = update_PG_covariances(covar, du, x_aug, x_aug_next)

    t = covar.t;
    tp1 = t + 1;

    phi = [du; x_aug];

    covar.X0c = (t/tp1)*covar.X0c + (1/tp1)*(x_aug*phi');
    covar.U0c = (t/tp1)*covar.U0c + (1/tp1)*(du*phi');
    covar.X1c = (t/tp1)*covar.X1c + (1/tp1)*(x_aug_next*phi');

    %% Rank-one update of Phi^{-1}

    denom = t + (phi' * (covar.Phi_inv * phi));

    covar.Phi_inv = (tp1/t) * ...
        (covar.Phi_inv - ...
        (covar.Phi_inv*(phi*phi')*covar.Phi_inv) / denom);

    % Symmetrize to reduce numerical drift
    covar.Phi_inv = 0.5*(covar.Phi_inv + covar.Phi_inv');

    covar.t = tp1;

end


%% ========================================================================
% Plots: model identification quality
% =========================================================================

function plot_identification_quality(X_id_data, X1_id_pred)

    figure;

    subplot(3,1,1);
    plot(X_id_data(1,2:end), 'b', 'LineWidth',1.2); hold on;
    plot(X1_id_pred(1,:), 'r--', 'LineWidth',1.0);
    grid on;
    xlabel('Identification sample');
    ylabel('T_{room} [deg C]', 'Interpreter','tex');
    legend('True data','One-step prediction','Location','best');
    title('Identified model one-step prediction: Room temperature');

    subplot(3,1,2);
    plot(X_id_data(2,2:end), 'b', 'LineWidth',1.2); hold on;
    plot(X1_id_pred(2,:), 'r--', 'LineWidth',1.0);
    grid on;
    xlabel('Identification sample');
    ylabel('T_{wo} [deg C]', 'Interpreter','tex');
    legend('True data','One-step prediction','Location','best');
    title('Identified model one-step prediction: Outer wall');

    subplot(3,1,3);
    plot(X_id_data(3,2:end), 'b', 'LineWidth',1.2); hold on;
    plot(X1_id_pred(3,:), 'r--', 'LineWidth',1.0);
    grid on;
    xlabel('Identification sample');
    ylabel('T_{wi} [deg C]', 'Interpreter','tex');
    legend('True data','One-step prediction','Location','best');
    title('Identified model one-step prediction: Inner wall');

end


%% ========================================================================
% Plots: single-case detailed diagnostics
% =========================================================================

function plot_single_case_diagnostics(result, T_sp)

    T_online = result.T_online;
    time_plus = (0:T_online)/24;
    time      = (0:T_online-1)/24;

    upper_limit = result.upper_limit;
    lower_limit = result.lower_limit;

    %% Plots: cost and spectral radii

    figure;
    subplot(2,1,1);
    semilogy(0:T_online, result.normK, 'LineWidth',1.5); grid on;
    xlabel('Online step'); ylabel('Frobenius Norm of Gain K');
    title('Frobenius Norm of Gain');

    subplot(2,1,2);
    plot(0:T_online, result.rho_true, 'k', 'LineWidth',1.2); hold on;
    plot(0:T_online, result.rho_id, 'b-.', 'LineWidth',1.0);
    plot(1:T_online, result.rho_hat(1:T_online), 'r--', 'LineWidth',1.0);
    yline(1,'k:');
    grid on; xlabel('Online step'); ylabel('Spectral radius');
    legend('\rho(A_{true}+B_{true}K)', ...
           '\rho(A_{id}+B_{id}K)', ...
           '\rho(X_1 V)', ...
           'Location','best','Interpreter','tex');
    title('Closed-loop spectral radii');

    %% Closed-loop state and input

    figure;
    subplot(2,1,1);
    plot(time_plus, result.x_all(1,:), 'b', 'LineWidth',1.5); hold on;
    plot(time_plus, result.x_all(2,:), 'g', 'LineWidth',1.5);
    plot(time_plus, result.x_all(3,:), 'm', 'LineWidth',1.5);
    plot(time_plus, result.Tout_plot, 'c', 'LineWidth',1.2);
    plot(time_plus, result.xss_all(1,:), 'b--', 'LineWidth',1.0);
    yline(T_sp,'r--');
    grid on; xlabel('Time [days]'); ylabel('States');
    legend('T_{room}','T_{wo}','T_{wi}','T_{out}','T_{room,ss}^{id}','SP', ...
           'Location','best','Interpreter','tex');
    title(sprintf('Closed-loop dynamics, cooling limit = %.0f W', lower_limit));

    subplot(2,1,2);
    plot(time, result.u_all, 'k', 'LineWidth',1.2); hold on;
    plot(time, result.u_unsat_all, 'r--', 'LineWidth',1.0);
    plot(time, result.uss_all, 'b-.', 'LineWidth',1.0);
    yline(upper_limit,'k:');
    yline(lower_limit,'k:');
    grid on; xlabel('Time [days]'); ylabel('Control [W]');
    legend('u','u_{unsat}','u_{ss}^{id}','limits','Location','best','Interpreter','tex');
    title('Control input: applied versus requested');

    %% Temperature tracking plot

    figure;

    subplot(2,1,1)
    plot(time_plus, result.x_all(1,:), 'b', 'LineWidth',2); hold on;
    plot(time_plus, result.x_all(2,:), 'g');
    plot(time_plus, result.x_all(3,:), 'm');
    plot(time_plus, result.Tout_plot, 'c');
    plot(time_plus, result.xss_all(1,:), 'b--');
    yline(T_sp,'r--');

    grid on;
    xlabel('Time [days]', 'Interpreter','none');
    ylabel('Temperature [^{\circ}C]', 'Interpreter', 'tex');
    legend('T_{room}','T_{wo}','T_{wi}','T_{out}','T_{room}^{id}','Setpoint', ...
           'Interpreter', 'tex');
    title('Offset-free temperature control using identified steady state');

    subplot(2,1,2)
    plot(time, result.u_all, 'k'); hold on;
    plot(time, result.u_unsat_all, 'r--');
    plot(time, result.uss_all, 'b:');
    plot(time, result.du_all, 'c:');
    yline(upper_limit,'k:');
    yline(lower_limit,'k:');
    grid on;
    xlabel('Time [days]', 'Interpreter','none');
    ylabel('Control [W]', 'Interpreter','none');
    legend('u','u_{unsat}','u_{ss}^{id}','\Delta u','sat. limits', 'Interpreter', 'tex');
    title('Control input');

    %% Integral action

    figure;
    plot(time_plus, result.z_all, 'b', 'LineWidth',1.5);
    grid on;
    xlabel('Time [days]', 'Interpreter', 'none');
    ylabel('Integrator state');
    title('Integral action');

    %% Tracking error

    figure;
    plot(time_plus, result.err_all, 'r', 'LineWidth',1.5);
    yline(0,'k--');
    grid on;
    xlabel('Time [days]', 'Interpreter','none');
    ylabel('Tracking error T_{sp}-T_{room}', 'Interpreter', 'tex');
    title('Tracking error');

    %% Deviation-state plot

    figure;
    subplot(2,1,1)
    plot(time_plus, result.e_all(2,:), 'g'); hold on;
    plot(time_plus, result.e_all(3,:), 'm');
    plot(time_plus, result.e_all(1,:), 'b', 'LineWidth',2);
    yline(0,'k--');
    grid on;
    xlabel('Time [days]');
    ylabel('Error states');
    legend('e_2 = T_{wo} - T_{wo,ss}^{id}', ...
           'e_3 = T_{wi} - T_{wi,ss}^{id}', ...
           'e_1 = T_{room} - T_{room,ss}^{id}', 'Interpreter', 'tex');
    title('Error states: room-temperature error is e_1', 'Interpreter', 'tex');

    subplot(2,1,2)
    stairs(time, result.sat_all, 'LineWidth',1.2);
    ylim([-0.1 1.1]);
    grid on;
    xlabel('Time [days]');
    ylabel('Saturation flag');
    title('Input saturation indicator');

    %% Individual gain entries

    figure;
    plot(0:T_online, result.K_hist', 'LineWidth',1.2);
    grid on;
    xlabel('Online step');
    ylabel('Gain entries');
    title('Evolution of controller gain entries');

end

%% ========================================================================
% Plots: integral action with zoomed inset
% =========================================================================

function plot_integral_action_with_inset(result, zoom_days)

    T_online = result.T_online;
    time_plus = (0:T_online)/24;

    if nargin < 2
        zoom_days = 5;
    end

    zoom_idx = time_plus <= zoom_days;

    figure;

    %% Main axes

    ax_main = axes;

    plot(ax_main, time_plus, result.z_all, 'b', 'LineWidth',1.5);
    hold(ax_main,'on');

    grid(ax_main,'on');
    box(ax_main,'on');

    xlabel(ax_main, 'Time [days]', 'Interpreter','none');
    ylabel(ax_main, 'Integrator state');

    title(ax_main, sprintf('Integral action with first %g days zoomed', zoom_days),'Interpreter','tex');

    %% Inset axes
    % Bottom-right inset location:
    % [left bottom width height]

    ax_inset = axes('Position',[0.54 0.18 0.34 0.28]);
    box(ax_inset,'on');

    plot(ax_inset, time_plus(zoom_idx), result.z_all(zoom_idx), ...
        'b', 'LineWidth',1.2);

    grid(ax_inset,'on');
    box(ax_inset,'on');

    xlabel(ax_inset, 'Time [days]', 'Interpreter','none');
    ylabel(ax_inset, 'z', 'Interpreter','none');
    title(ax_inset, sprintf('First %g days', zoom_days),'Interpreter','tex');

    xlim(ax_inset, [0 zoom_days]);

    z_zoom = result.z_all(zoom_idx);

    if ~isempty(z_zoom)

        z_min = min(z_zoom);
        z_max = max(z_zoom);

        if abs(z_max-z_min) < 1e-8
            z_min = z_min - 0.1;
            z_max = z_max + 0.1;
        end

        z_margin = 0.15*(z_max-z_min + eps);
        ylim(ax_inset, [z_min-z_margin, z_max+z_margin]);

    end

    % Make sure inset is visible on top of the main axes.
    uistack(ax_inset,'top');

end

%% ========================================================================
% Plots: cooling saturation comparison
% =========================================================================

function plot_cooling_saturation_comparison(case_results, cooling_limits, upper_limit, T_sp)

    n_cases = numel(case_results);

    T_online = case_results{1}.T_online;
    time_plus = (0:T_online)/24;
    time = (0:T_online-1)/24;

    figure;

    for ii = 1:n_cases

        result = case_results{ii};
        lower_limit = cooling_limits(ii);

        subplot(n_cases,2,2*ii-1);

        plot(time_plus, result.x_all(1,:), 'b', 'LineWidth',1.5); hold on;
        % plot(time_plus, result.xss_all(1,:), 'b--', 'LineWidth',1.0);
        yline(T_sp,'r--');

        grid on;
        xlabel('Time [days]', 'Interpreter','none');
        ylabel('T_{room} [^{\circ}C]', 'Interpreter','tex');

        title(sprintf('Room temperature, cooling limit = %.0f W', lower_limit));

        % legend('T_{room}', 'T_{room,ss}^{id}', 'Setpoint', ...
        %     'Location','best', 'Interpreter','tex');

         legend('T_{room}', 'Setpoint', ...
            'Location','best', 'Interpreter','tex');

        subplot(n_cases,2,2*ii);

        plot(time, result.u_unsat_all, 'r', 'LineWidth',1.0); hold on;
        plot(time, result.u_all, 'k', 'LineWidth',1.1);
        yline(upper_limit,'k:');
        yline(lower_limit,'k:');

        grid on;
        xlabel('Time [days]', 'Interpreter','none');
        ylabel('Control [W]', 'Interpreter','none');

        title(sprintf('Requested vs applied input, cooling limit = %.0f W', lower_limit));

        legend('u_{unsat}', 'u', 'sat. limits', ...
            'Location','best', 'Interpreter','tex');

    end

end


%% ========================================================================
% Plots: tracking-error comparison
% =========================================================================

function plot_tracking_error_comparison(case_results, cooling_limits)

    n_cases = numel(case_results);

    T_online = case_results{1}.T_online;
    time_plus = (0:T_online)/24;

    figure;

    subplot(2,1,1);

    for ii = 1:n_cases
        plot(time_plus, case_results{ii}.err_all, 'LineWidth',1.2); hold on;
    end

    yline(0,'k--');

    grid on;
    xlabel('Time [days]', 'Interpreter','none');
    ylabel('Tracking error T_{sp}-T_{room}', 'Interpreter','tex');
    title('Tracking error comparison for different cooling limits');

    labels = cell(n_cases,1);
    for ii = 1:n_cases
        labels{ii} = sprintf('Cooling limit %.0f W', cooling_limits(ii));
    end
    legend(labels, 'Location','best');

    subplot(2,1,2);

    sat_fraction = zeros(n_cases,1);
    lower_sat_fraction = zeros(n_cases,1);

    for ii = 1:n_cases
        sat_fraction(ii) = mean(case_results{ii}.sat_all);
        lower_sat_fraction(ii) = mean(case_results{ii}.sat_lower_all);
    end

    bar(categorical(string(cooling_limits)), 100*[sat_fraction lower_sat_fraction]);

    grid on;
    xlabel('Cooling lower limit [W]', 'Interpreter','none');
    ylabel('Saturation fraction [%]', 'Interpreter','none');
    title('Total saturation and cooling saturation fraction');

    legend('Total saturation', 'Cooling saturation', 'Location','best');

end

%% ========================================================================
% Plots: tracking error with zoomed inset
% =========================================================================

function plot_tracking_error_with_inset(result, T_sp, zoom_days)

    T_online = result.T_online;
    time_plus = (0:T_online)/24;

    if nargin < 3
        zoom_days = 5;
    end

    zoom_idx = time_plus <= zoom_days;

    figure;

    %% Main axes

    ax_main = axes;

    plot(ax_main, time_plus, result.err_all, 'r', 'LineWidth',1.5);
    hold(ax_main,'on');

    yline(ax_main, 0,'k--');

    grid(ax_main,'on');
    box(ax_main,'on');

    xlabel(ax_main, 'Time [days]', 'Interpreter','none');
    ylabel(ax_main, 'Tracking error T_{sp}-T_{room}', 'Interpreter','tex');

    % title(ax_main, sprintf('Tracking error with first %g days zoomed', zoom_days),'Interpreter','tex');

    %% Inset axes
    % Bottom-right inset location:
    % [left bottom width height]
    %
    % Increase left to move right.
    % Decrease bottom to move down.
    % Decrease width/height if it overlaps the x-axis label.

    ax_inset = axes('Position',[0.54 0.18 0.34 0.28]);
    box(ax_inset,'on');

    plot(ax_inset, time_plus(zoom_idx), result.err_all(zoom_idx), ...
        'r', 'LineWidth',1.2);
    hold(ax_inset,'on');

    yline(ax_inset, 0,'k--');

    grid(ax_inset,'on');
    box(ax_inset,'on');

    xlabel(ax_inset, 'Time [days]', 'Interpreter','none');
    ylabel(ax_inset, 'Error', 'Interpreter','none');
    % title(ax_inset, sprintf('First %g days', zoom_days),'Interpreter','tex');

    xlim(ax_inset, [0 zoom_days]);

    y_zoom = result.err_all(zoom_idx);

    if ~isempty(y_zoom)

        y_min = min(y_zoom);
        y_max = max(y_zoom);

        if abs(y_max-y_min) < 1e-8
            y_min = y_min - 0.1;
            y_max = y_max + 0.1;
        end

        y_margin = 0.15*(y_max-y_min + eps);
        ylim(ax_inset, [y_min-y_margin, y_max+y_margin]);

    end

    % Make sure inset is visible on top of the main axes.
    uistack(ax_inset,'top');

end


%% ========================================================================
% Plots: Frobenius norm comparison for different saturation limits
% =========================================================================

function plot_frobenius_norm_comparison(case_results, cooling_limits)

    n_cases = numel(case_results);

    T_online = case_results{1}.T_online;
    time_days = (0:T_online)/24;

    figure;

    for ii = 1:n_cases
        semilogy(time_days, case_results{ii}.normK, 'LineWidth',1.5); hold on;
    end

    grid on;
    xlabel('Time [days]', 'Interpreter','none');
    ylabel('Frobenius norm ||K||_F', 'Interpreter','tex');
    title('Frobenius norm of learned controller gains for different cooling limits');

    labels = cell(n_cases,1);
    for ii = 1:n_cases
        labels{ii} = sprintf('Cooling limit %.0f W', cooling_limits(ii));
    end

    legend(labels, 'Location','best');

end

%% ========================================================================
% Plots: tracking-error comparison
% =========================================================================

function plot_sat_tracking_error_comparison(case_results, cooling_limits, day_window)

    if nargin < 3
        day_window = [100 260];
    end

    n_cases = numel(case_results);

    T_online = case_results{1}.T_online;
    time_plus = (0:T_online)/24;

    idx_x = time_plus >= day_window(1) & time_plus <= day_window(2);

    figure;

    for ii = 1:n_cases
        plot(time_plus(idx_x), case_results{ii}.err_all(idx_x), ...
            'LineWidth',1.4); 
        hold on;
    end

    yline(0, 'k--', 'LineWidth',1.0);

    grid on;
    xlabel('Time [days]', 'Interpreter','none');
    ylabel('Tracking error T_{sp}-T_{room} [^{\circ}C]', 'Interpreter','tex');

    labels = cell(n_cases,1);
    for ii = 1:n_cases
        labels{ii} = sprintf('Cooling limit %.0f W', cooling_limits(ii));
    end

    legend(labels, 'Location','best');

    title('Tracking error for different cooling authority limits');

end

%% ========================================================================
% Plots: cooling clipping comparison
% =========================================================================

function plot_cooling_clipping_comparison(case_results, cooling_limits, day_window)

    if nargin < 3
        day_window = [100 260];
    end

    n_cases = numel(case_results);

    T_online = case_results{1}.T_online;
    time = (0:T_online-1)/24;

    idx_u = time >= day_window(1) & time <= day_window(2);

    figure;

    for ii = 1:n_cases

        result = case_results{ii};

        % Positive value means missing cooling power due to saturation.
        cooling_clipping = max(0, result.u_all - result.u_unsat_all);

        subplot(n_cases,1,ii);

        area(time(idx_u), cooling_clipping(idx_u), ...
            'FaceAlpha',0.45, ...
            'LineStyle','none');

        grid on;
        xlabel('Time [days]', 'Interpreter','none');
        ylabel('Missing cooling [W]', 'Interpreter','none');

        title(sprintf('Cooling clipping, lower limit = %.0f W', cooling_limits(ii)));

    end

end

%% ========================================================================
% Plots: simplified year-long closed-loop result
% =========================================================================

function plot_simplified_year_result(result, T_sp)

    T_online = result.T_online;

    time_plus = (0:T_online)/24;
    time      = (0:T_online-1)/24;

    upper_limit = result.upper_limit;
    lower_limit = result.lower_limit;

    figure;

    %% Room temperature and outdoor temperature

    subplot(2,1,1);

    % Plot outdoor temperature first so it stays in the background.
    h_Tout = plot(time_plus, result.Tout_plot, ...
        'Color',[0.65 0.65 0.65], ...
        'LineWidth',0.8); 
    hold on;

    % Plot setpoint next.
    h_sp = yline(T_sp, 'r--', 'LineWidth',1.2);

    % Plot room temperature last so it appears on top.
    h_Troom = plot(time_plus, result.x_all(1,:), ...
        'b', ...
        'LineWidth',1.5);

    grid on;
    xlabel('Time [days]', 'Interpreter','none');
    ylabel('Temperature [^{\circ}C]', 'Interpreter','tex');

    legend([h_Troom, h_sp, h_Tout], ...
        {'T_{room}', 'Setpoint', 'T_{out}'}, ...
        'Location','best', ...
        'Interpreter','tex');

    title(sprintf('Year-long room temperature, cooling limit = %.0f W', lower_limit));


    %% Applied and requested control input

    subplot(2,1,2);

    % Plot requested unconstrained input first.
    h_u_unsat = plot(time, result.u_unsat_all, ...
        'r--', ...
        'LineWidth',0.9); 
    hold on;

    % Plot saturation limits.
    h_upper = yline(upper_limit, 'k:', 'LineWidth',1.0);
    h_lower = yline(lower_limit, 'k:', 'LineWidth',1.0);

    % Plot applied input last so black u appears on top of u_unsat.
    h_u = plot(time, result.u_all, ...
        'k', ...
        'LineWidth',1.1);

    grid on;
    xlabel('Time [days]', 'Interpreter','none');
    ylabel('Control [W]', 'Interpreter','none');

    legend([h_u, h_u_unsat, h_lower], ...
        {'u', 'u_{unsat}', 'sat. limits'}, ...
        'Location','best', ...
        'Interpreter','tex');

    title('Applied and requested control input');

end

%% ========================================================================
% Plots: PG covariance-parameterized eigenvalue comparison
% =========================================================================

function plot_PG_eigenvalue_comparison(case_results, cooling_limits)

    n_cases = numel(case_results);

    theta = linspace(0, 2*pi, 500);

    figure;

    % Unit circle for discrete-time stability
    plot(cos(theta), sin(theta), 'k--', 'LineWidth',1.2);
    hold on;

    for ii = 1:n_cases

        eig_final = case_results{ii}.eig_hat_all(:,end);

        plot(real(eig_final), imag(eig_final), 'o', ...
            'MarkerSize',8, ...
            'LineWidth',1.5);

    end

    axis equal;
    grid on;
    box on;

    xlabel('Real part', 'Interpreter','none');
    ylabel('Imaginary part', 'Interpreter','none');

    title('Final PG covariance-parameterized eigenvalues');

    labels = cell(n_cases+1,1);
    labels{1} = 'Unit circle';

    for ii = 1:n_cases
        labels{ii+1} = sprintf('Cooling limit %.0f W', cooling_limits(ii));
    end

    legend(labels, 'Location','best');

end

%% ========================================================================
% Plots: PG covariance-parameterized spectral-radius comparison
% =========================================================================

function plot_PG_spectral_radius_comparison(case_results, cooling_limits)

    n_cases = numel(case_results);

    T_online = case_results{1}.T_online;
    time_days = (0:T_online)/24;

    figure;

    for ii = 1:n_cases
        plot(time_days, case_results{ii}.rho_hat, 'LineWidth',1.5);
        hold on;
    end

    yline(1, 'k--', 'LineWidth',1.2);

    grid on;
    box on;

    xlabel('Time [days]', 'Interpreter','none');
    ylabel('\rho(X_1 V)', 'Interpreter','tex');

    title('PG covariance-parameterized stability under different cooling limits');

    labels = cell(n_cases+1,1);

    for ii = 1:n_cases
        labels{ii} = sprintf('Cooling limit %.0f W', cooling_limits(ii));
    end

    labels{end} = '\rho = 1';

    legend(labels, 'Location','best', 'Interpreter','tex');

end

% %% ========================================================================
% % Plots: combined PG covariance-parameterized stability figure
% % =========================================================================
% 
% function plot_PG_stability_combined(case_results, cooling_limits)
% 
%     n_cases = numel(case_results);
% 
%     T_online = case_results{1}.T_online;
%     time_days = (0:T_online)/24;
% 
%     theta = linspace(0, 2*pi, 500);
% 
%     figure;
% 
%     %% Left panel: spectral-radius evolution
% 
%     subplot(1,2,1);
% 
%     for ii = 1:n_cases
%         plot(time_days, case_results{ii}.rho_hat, 'LineWidth',1.5);
%         hold on;
%     end
% 
%     yline(1, 'k--', 'LineWidth',1.2);
% 
%     grid on;
%     box on;
% 
%     xlabel('Time [days]', 'Interpreter','none');
%     ylabel('\rho(X_1 V)', 'Interpreter','tex');
% 
%     title('Evolution of spectral radius of largest eigenvalue', 'Interpreter','tex');
% 
%     labels_left = cell(n_cases+1,1);
% 
%     for ii = 1:n_cases
%         labels_left{ii} = sprintf('Cooling limit %.0f W', cooling_limits(ii));
%     end
% 
%     labels_left{end} = '\rho = 1';
% 
%     legend(labels_left, 'Location','best', 'Interpreter','tex');
% 
% 
%     %% Right panel: final covariance-parameterized eigenvalues
% 
%     subplot(1,2,2);
% 
%     plot(cos(theta), sin(theta), 'k--', 'LineWidth',1.2);
%     hold on;
% 
%     for ii = 1:n_cases
% 
%         eig_final = case_results{ii}.eig_hat_all(:,end);
% 
%         plot(real(eig_final), imag(eig_final), 'o', ...
%             'MarkerSize',8, ...
%             'LineWidth',1.5);
% 
%     end
% 
%     axis equal;
%     grid on;
%     box on;
% 
%     xlabel('Real part', 'Interpreter','none');
%     ylabel('Imaginary part', 'Interpreter','none');
% 
%     title('Final eigenvalues', 'Interpreter','tex');
% 
%     labels_right = cell(n_cases+1,1);
%     labels_right{1} = 'Unit circle';
% 
%     for ii = 1:n_cases
%         labels_right{ii+1} = sprintf('Cooling limit %.0f W', cooling_limits(ii));
%     end
% 
%     legend(labels_right, 'Location','best');
% 
%     % sgtitle('Covariance-parameterized stability under different cooling limits');
% 
% end

%% ========================================================================
% Plots: compact combined PG covariance-parameterized stability figure
% =========================================================================

function plot_PG_stability_combined(case_results, cooling_limits)


    n_cases = numel(case_results);

    T_online = case_results{1}.T_online;
    time_days = (0:T_online)/24;

    theta = linspace(0, 2*pi, 500);

    % Consistent colors for the three saturation cases
    case_colors = [ ...
        0.0000 0.4470 0.7410;   % blue   : -500 W
        0.8500 0.3250 0.0980;   % orange : -1500 W
        0.9290 0.6940 0.1250];  % yellow : -2500 W

    % Font settings for paper readability
    font_name = 'Times New Roman';
    font_size = 28;
    title_font_size = 28;
    legend_font_size = 20;

    figure;
    set(gcf, 'Color','w');

    % Compact but readable figure size in centimeters
    set(gcf, 'Units','centimeters');
    set(gcf, 'Position',[3 3 18 7.5]);

    tiledlayout(1,2, ...
        'TileSpacing','compact', ...
        'Padding','compact');


    %% Left panel: spectral-radius evolution

    ax1 = nexttile;

    for ii = 1:n_cases
        plot(ax1, time_days, case_results{ii}.rho_hat, ...
            'Color', case_colors(ii,:), ...
            'LineWidth',1.6);
        hold(ax1,'on');
    end

    yline(ax1, 1, 'k--', 'LineWidth',1.2);

    grid(ax1,'on');
    box(ax1,'on');

    xlabel(ax1, 'Time [days]', ...
        'Interpreter','none', ...
        'FontName',font_name, ...
        'FontSize',font_size);

    ylabel(ax1, '\rho(X_1 V)', ...
        'Interpreter','tex', ...
        'FontName',font_name, ...
        'FontSize',font_size);

    title(ax1, 'Spectral radius evolution', ...
        'Interpreter','none', ...
        'FontName',font_name, ...
        'FontSize',title_font_size);

    labels_left = cell(n_cases+1,1);

    for ii = 1:n_cases
        labels_left{ii} = sprintf('%.0f W', cooling_limits(ii));
    end

    labels_left{end} = '\rho = 1';

    leg1 = legend(ax1, labels_left, ...
        'Location','best', ...
        'Interpreter','tex');

    set(leg1, ...
        'FontName',font_name, ...
        'FontSize',legend_font_size, ...
        'Box','off');

    set(ax1, ...
        'FontName',font_name, ...
        'FontSize',font_size, ...
        'LineWidth',1.0);


    %% Right panel: final covariance-parameterized eigenvalues

    ax2 = nexttile;

    plot(ax2, cos(theta), sin(theta), 'k--', ...
        'LineWidth',1.2);
    hold(ax2,'on');

    for ii = 1:n_cases

        eig_final = case_results{ii}.eig_hat_all(:,end);

        plot(ax2, real(eig_final), imag(eig_final), 'o', ...
            'Color', case_colors(ii,:), ...
            'MarkerEdgeColor', case_colors(ii,:), ...
            'MarkerFaceColor','none', ...
            'MarkerSize',6.5, ...
            'LineWidth',1.4);

    end

    axis(ax2,'equal');
    grid(ax2,'on');
    box(ax2,'on');

    % Compact unit-circle view
    xlim(ax2, [-1.08 1.08]);
    ylim(ax2, [-1.08 1.08]);

    xlabel(ax2, 'Real part', ...
        'Interpreter','none', ...
        'FontName',font_name, ...
        'FontSize',font_size);

    ylabel(ax2, 'Imaginary part', ...
        'Interpreter','none', ...
        'FontName',font_name, ...
        'FontSize',font_size);

    title(ax2, 'Final eigenvalues', ...
        'Interpreter','none', ...
        'FontName',font_name, ...
        'FontSize',title_font_size);

    labels_right = cell(n_cases+1,1);
    labels_right{1} = 'Unit circle';

    for ii = 1:n_cases
        labels_right{ii+1} = sprintf('%.0f W', cooling_limits(ii));
    end

    leg2 = legend(ax2, labels_right, ...
        'Location','best', ...
        'Interpreter','none');

    set(leg2, ...
        'FontName',font_name, ...
        'FontSize',legend_font_size, ...
        'Box','off');

    set(ax2, ...
        'FontName',font_name, ...
        'FontSize',font_size, ...
        'LineWidth',1.0);

end



%% ========================================================================
% Final closed-loop diagnostics
% =========================================================================

function print_case_diagnostics(case_results, cooling_limits, T_sp, Ts)

    fprintf('\n===== Cooling saturation comparison diagnostics =====\n');

    for ii = 1:numel(case_results)

        result = case_results{ii};

        tracking_error = T_sp - result.x_all(1,:);

        rms_err = sqrt(mean(tracking_error.^2));
        max_abs_err = max(abs(tracking_error));
        mean_abs_err = mean(abs(tracking_error));

        sat_frac = mean(result.sat_all);
        upper_sat_frac = mean(result.sat_upper_all);
        lower_sat_frac = mean(result.sat_lower_all);

        heating_energy_kWh = sum(max(result.u_all,0))*Ts/3600/1000;
        cooling_energy_kWh = sum(max(-result.u_all,0))*Ts/3600/1000;

        fprintf('\nCooling lower limit = %.0f W\n', cooling_limits(ii));
        fprintf('Final room temperature       = %.3f deg C\n', result.x_all(1,end));
        fprintf('Final tracking error         = %.3f deg C\n', T_sp - result.x_all(1,end));
        fprintf('Final control input          = %.3f W\n', result.u_all(1,end));
        fprintf('RMS tracking error           = %.3f deg C\n', rms_err);
        fprintf('Mean abs tracking error      = %.3f deg C\n', mean_abs_err);
        fprintf('Max abs tracking error       = %.3f deg C\n', max_abs_err);
        fprintf('Total saturation fraction    = %.2f %%\n', 100*sat_frac);
        fprintf('Heating saturation fraction  = %.2f %%\n', 100*upper_sat_frac);
        fprintf('Cooling saturation fraction  = %.2f %%\n', 100*lower_sat_frac);
        fprintf('Heating energy               = %.2f kWh\n', heating_energy_kWh);
        fprintf('Cooling energy               = %.2f kWh\n', cooling_energy_kWh);

        if sat_frac > 0.2
            warning(['The actuator is saturated for a large fraction of the simulation. ', ...
                     'If the room does not track the setpoint, this is due to lack of sufficient control authority.']);
        end

    end

end


%% ========================================================================
% True cost for monitoring only
% =========================================================================

function [C, SigmaK] = true_LQR_cost(A,B,Q,R,K)

    Acl = A + B*K;

    try
        SigmaK = dlyap(Acl, eye(size(A,1)));
        C = trace((Q + K'*R*K) * SigmaK);
    catch
        SigmaK = inf(size(A,1));
        C = inf;
    end

end


%% ========================================================================
% Estimate A, B, E and affine bias from historical data
% =========================================================================

function [A_id, B_id, E_id, c_id, X1_pred, rmse] = ...
    estimate_hvac_model(X_data, U_data, D_data, lambda)

    % Identification model:
    %
    %   x_{k+1} = A*x_k + B*u_k + E*d_k + c
    %
    % Data:
    %   X_data : n x (N+1)
    %   U_data : m x N
    %   D_data : p x N

    n = size(X_data,1);
    m = size(U_data,1);
    p = size(D_data,1);
    N = size(U_data,2);

    X0 = X_data(:,1:N);
    X1 = X_data(:,2:N+1);
    U0 = U_data(:,1:N);
    D0 = D_data(:,1:N);

    Ones = ones(1,N);

    Z = [X0;
         U0;
         D0;
         Ones];

    % Ridge-regularized least squares:
    %
    %   Theta = X1*Z'/(Z*Z' + lambda*I)
    %
    % where Theta = [A B E c]

    Theta = X1 * Z' / (Z*Z' + lambda*eye(size(Z,1)));

    A_id = Theta(:,1:n);
    B_id = Theta(:,n+1:n+m);
    E_id = Theta(:,n+m+1:n+m+p);
    c_id = Theta(:,end);

    X1_pred = A_id*X0 + B_id*U0 + E_id*D0 + c_id;

    pred_err = X1 - X1_pred;
    rmse = sqrt(mean(pred_err.^2,2));

end


%% ========================================================================
% Weather-dependent steady state using identified model
% =========================================================================

function [x_ss, u_ss] = steady_state_hvac_id(A,B,E,c_aff,C_track,T_sp,d_input)

    n = size(A,1);
    m = size(B,2);

    % Identified steady-state equations:
    %
    % x_ss = A*x_ss + B*u_ss + E*d + c_aff
    % C_track*x_ss = T_sp
    %
    % Rearranged:
    %
    % (I-A)*x_ss - B*u_ss = E*d + c_aff
    % C_track*x_ss         = T_sp

    M = [eye(n)-A, -B;
         C_track,  zeros(size(C_track,1),m)];

    rhs = [E*d_input + c_aff;
           T_sp];

    sol = M\rhs;

    x_ss = sol(1:n);
    u_ss = sol(n+1:end);

end
