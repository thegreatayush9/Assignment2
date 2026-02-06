clc;
clear;

% Parameters
F       = 1.0;       % m3/h
V       = 1.0;       % m3
CAf     = 10.0;      % kgmol/m3
k0      = 36e6;      % h^-1
E       = 12000.0;   % kcal/kgmol
R       = 1.987;     % kcal/(kgmol.K)
dH_neg  = 6500.0;    % (-deltaH) kcal/kgmol
UA      = 150.0;     % kcal/(C h)
Tf      = 298.0;     % K
Tj0     = 298.0;     % K
rhoCp   = 500.0;     % (rho Cp) kcal/(m3 C)
rhojCj  = 600.0;     % (rhoj Cj) kcal/(m3 C)
Fj      = 1.25;      % m3/h
Vj      = 0.25;      % m3

% Define the system of equations
F_system = @(x) [
    F*CAf - F*x(1) - k0*exp(-E/(R*x(2)))*x(1)*V;           % Mass balance
    rhoCp*F*(Tf - x(2)) + dH_neg*V*k0*exp(-E/(R*x(2)))*x(1) - UA*(x(2) - x(3)); % Reactor energy
    rhojCj*Fj*(Tj0 - x(3)) + UA*(x(2) - x(3))               % Jacket energy
];

% Initial guesses
initial_guesses = [
    9.5, 300.0, 300.0;
    5.0, 350.0, 320.0;
    1.0, 410.0, 340.0
];

labels = {'Low Conversion', 'Intermediate', 'High Conversion'};

options = optimoptions('fsolve','Display','off'); 

fprintf('%-15s | %-15s | %-10s | %-10s\n', 'Steady State', 'CA (kgmol/m3)', 'T (K)', 'Tj (K)');
fprintf(repmat('-',1,60)); fprintf('\n');

% Solve using fsolve
for i = 1:size(initial_guesses,1)
    guess = initial_guesses(i,:);
    [sol, ~, exitflag] = fsolve(F_system, guess, options);
    
    if exitflag > 0
        fprintf('%-15s | %-15.4f | %-10.2f | %-10.2f\n', labels{i}, sol(1), sol(2), sol(3));
    else
        fprintf('%-15s | Failed to converge\n', labels{i});
    end
end


