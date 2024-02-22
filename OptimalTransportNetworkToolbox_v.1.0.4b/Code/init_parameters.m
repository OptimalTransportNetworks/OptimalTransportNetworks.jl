%{
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

[param] = init_parameters (varargin): returns a 'param' structure with the model parameters.

Arguments:
- varargin: (optional) list of optional parameters with specified values for model parameters
to be set different from default.

The list of parameters is:
 - 'alpha': Cobb-Douglas coefficient on final good c^alpha * h^(1-alpha)  (default 0.5)
 - 'beta': parameter governing congestion in transport cost (default 1)
 - 'gamma': elasticity of transport cost relative to infrastructure (default 1)
 - 'K': amount of concrete/asphalt (default 1)
 - 'sigma': elasticity of substitution across goods (CES) (default 5)
 - 'rho': curvature in utility (c^alpha * h^(1-alpha))^(1-rho)/(1-rho) (default 2)
  - 'a': curvature of the production function L^alpha (default .8)
  - 'm': vector of weights Nx1 in the cross congestion cost function (default ones(N,1))
 - 'N': number of goods (default 1)
 - 'nu': elasticity of substitution b/w goods in transport costs if cross-good congestion (default 1)
 - 'LaborMobility': switch for labor mobility ('on'/'off'/'partial') (default off)
 - 'CrossGoodCongestion': switch for cross-good congestion (1 if yes, 0 otherwise) (default off)
 - 'Annealing': switch for the use of annealing at the end of iterations (default on)
 - 'Custom': switch for the use of custom lagrangian function (1 if yes, 0 otherwise) (default off)
 - 'Verbose': switch to turn on/off text output (default on)
 - 'ADiGator': use autodifferentiation with Adigator (default off)
 - 'Duality': switch to turn on/off duality whenever available (default on)
 - 'param': provide an already existing 'param' structure if you just want
            to change parameters.
 
Additional parameters for the numerical part: tol_kappa, MIN_KAPPA, solve_allocation_handle.

-----------------------------------------------------------------------------------
REFERENCE: "Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
Fajgelbaum and Edouard Schaal.

Copyright (c) 2017-2019, Pablo D. Fajgelbaum, Edouard Schaal
pfajgelbaum@ucla.edu, eschaal@crei.cat

This code is distributed under BSD-3 License. See LICENSE.txt for more information.
-----------------------------------------------------------------------------------
%}

function param = init_parameters(varargin)

p = inputParser;

% ====================================
% DEFINE STANDARD VALUES OF PARAMETERS

default_mobility = 'off';
default_rho=2;
default_alpha=0.5;
default_sigma=5;
default_gamma=1;
default_beta=1;
default_K=1;
default_a=.8;
default_tol_kappa=1e-7;
default_min_kappa=1e-5;
default_cong='off';
default_N=1;
default_nu=1;
default_custom='off';
default_annealing='on';
default_verbose='on';
default_adigator='off';
default_duality='on';
default_m=1;
default_param=[];

% =============
% DEFINE INPUTS

validSwitch = {'on','off'};
checkSwitch = @(x) any(validatestring(x,validSwitch));

addParameter(p,'LaborMobility',default_mobility,@(x) any(validatestring(x,{'on','off','partial'}))); % whether labor is mobile or not
addParameter(p,'CrossGoodCongestion',default_cong,checkSwitch); % whether we have cross-good congestion or not
addParameter(p,'Custom',default_custom,checkSwitch); % whether to use a custom specification or not
addParameter(p,'Annealing',default_annealing,checkSwitch); % whether to use annealing
addParameter(p,'Verbose',default_verbose,checkSwitch); % whether to display output
addParameter(p,'ADiGator',default_adigator,checkSwitch); % whether to display output
addParameter(p,'Duality',default_duality,checkSwitch); % whether to use duality (when available)

% Allow to specify utility parameters

addParameter(p,'rho',default_rho,@isnumeric);
addParameter(p,'alpha',default_alpha,@isnumeric);
addParameter(p,'sigma',default_sigma,@isnumeric);

% Allow to specify economy parameters

addParameter(p,'gamma',default_gamma,@isnumeric);
addParameter(p,'beta',default_beta,@isnumeric);
addParameter(p,'K',default_K,@isnumeric);
addParameter(p,'a',default_a,@isnumeric);
addParameter(p,'N',default_N,@isnumeric);
addParameter(p,'nu',default_nu,@isnumeric);
addParameter(p,'m',default_m,@isvector);

% Allow to specify resolution parameters

addParameter(p,'TolKappa',default_tol_kappa,@isnumeric);
addParameter(p,'MinKappa',default_min_kappa,@isnumeric);

% Allow to provide existing param structure as default

fields = {'gamma','alpha','beta','K','sigma','rho','tol_kappa','MIN_KAPPA',...
           'N','nu','a','m','mobility','cong','custom','annealing','verbose',...
           'adigator','minpop','MAX_ITER_KAPPA','MAX_ITER_L','u','uprime',...
           'usecond','uprimeinv','F','Fprime'};

checkParam = @(x) ~any(isfield(x,fields)==0);

addParameter(p,'param',default_param,checkParam);

% Parse inputs
parse(p,varargin{:});

% ===================
% RETURN PARAM OBJECT

param_specified = ~any(strcmp(p.UsingDefaults,'param')); % if a param object has been specified

if param_specified 
    param = p.Results.param;
end

if ~param_specified || ~any(strcmp(p.UsingDefaults,'gamma'))  % if gamma has been specified or no param object given 
    param.gamma=p.Results.gamma; % parameter of the cost function
end

if ~param_specified || ~any(strcmp(p.UsingDefaults,'alpha'))  
    param.alpha=p.Results.alpha; 
end

if ~param_specified || ~any(strcmp(p.UsingDefaults,'beta'))  
    param.beta=p.Results.beta; 
end

if ~param_specified || ~any(strcmp(p.UsingDefaults,'K'))  
    param.K=p.Results.K; 
end

if ~param_specified || ~any(strcmp(p.UsingDefaults,'sigma'))  
    param.sigma=p.Results.sigma; 
end

if ~param_specified || ~any(strcmp(p.UsingDefaults,'rho'))  
    param.rho=p.Results.rho; 
end

if ~param_specified || ~any(strcmp(p.UsingDefaults,'TolKappa'))  
    param.tol_kappa=p.Results.TolKappa; 
end

if ~param_specified || ~any(strcmp(p.UsingDefaults,'MinKappa'))  
    param.MIN_KAPPA=p.Results.MinKappa; 
end

if ~param_specified || ~any(strcmp(p.UsingDefaults,'N'))  
    param.N=p.Results.N; 
end

if ~param_specified || ~any(strcmp(p.UsingDefaults,'nu'))  
    param.nu=p.Results.nu; 
end

if ~param_specified || ~any(strcmp(p.UsingDefaults,'a'))  
    param.a=p.Results.a; 
end

if ~param_specified || ~any(strcmp(p.UsingDefaults,'m'))  
    param.m=p.Results.m; 
end

if ~param_specified || ~any(strcmp(p.UsingDefaults,'LaborMobility'))  
	if strcmp(p.Results.LaborMobility,'partial')
		param.mobility = 0.5;
	else
		param.mobility = strcmp(p.Results.LaborMobility,'on');
	end 
end

if param.mobility == true
    param.rho = 0;
end

if ~isequal(size(param.m),[param.N 1])
    if any(strcmp(p.UsingDefaults,'m')) && ~param_specified
        param.m=ones(param.N,1);
    else
        error('%s.m: vector of congestion weights m should have size Nx1.',mfilename);
    end
end

if ~param_specified || ~any(strcmp(p.UsingDefaults,'CrossGoodCongestion'))  
    param.cong = strcmp(p.Results.CrossGoodCongestion,'on');
end

if ~param_specified || ~any(strcmp(p.UsingDefaults,'Custom'))  
    param.custom = strcmp(p.Results.Custom,'on');
end

if ~param_specified || ~any(strcmp(p.UsingDefaults,'Annealing'))  
    param.annealing = strcmp(p.Results.Annealing,'on');
end

if ~param_specified || ~any(strcmp(p.UsingDefaults,'Verbose'))  
    param.verbose = strcmp(p.Results.Verbose,'on');
end

if ~param_specified || ~any(strcmp(p.UsingDefaults,'ADiGator'))  
    param.adigator = strcmp(p.Results.ADiGator,'on');
end

if ~param_specified || ~any(strcmp(p.UsingDefaults,'Duality'))  
    param.duality = strcmp(p.Results.Duality,'on');
end

if param.custom % always use ADiGator in the custom case
    param.adigator = true;
end

% Parameters that never change
if ~param_specified
    param.minpop=1e-3; % algorithm unstable if 0 pop, so keep always a minimum population number
    param.MAX_ITER_KAPPA=200;
    param.MAX_ITER_L=100;
    param.MIN_KAPPA=1e-5;
end

% =======================
% DEFINE UTILITY FUNCTION

param.u = @(c,h) ((c/param.alpha).^param.alpha.*(h/(1-param.alpha)).^(1-param.alpha)).^(1-param.rho)/(1-param.rho);
param.uprime = @(c,h) ((c/param.alpha).^param.alpha.*(h/(1-param.alpha)).^(1-param.alpha)).^(-param.rho).*((c/param.alpha).^(param.alpha-1).*(h/(1-param.alpha)).^(1-param.alpha));
param.usecond = @(c,h) -param.rho*((c/param.alpha).^param.alpha.*(h/(1-param.alpha)).^(1-param.alpha)).^(-param.rho-1).*((c/param.alpha).^(param.alpha-1).*(h/(1-param.alpha)).^(1-param.alpha)).^2+...
    (param.alpha-1)/param.alpha*((c/param.alpha).^param.alpha.*(h/(1-param.alpha)).^(1-param.alpha)).^(-param.rho).*((c/param.alpha).^(param.alpha-2).*(h/(1-param.alpha)).^(1-param.alpha));

param.uprimeinv = @(x,h) param.alpha*x.^(-1/(1+param.alpha*(param.rho-1))).*(h/(1-param.alpha)).^-((1-param.alpha)*(param.rho-1)/(1+param.alpha*(param.rho-1)));

% ==========================
% DEFINE PRODUCTION FUNCTION

param.F = @(L,a) L.^a;
param.Fprime = @(L,a) a*L.^(a-1);

% ==============================================================
% CHECK CONSISTENCY WITH ENDOWMENTS/PRODUCTIVITY (if applicable)
% only happens when object param is specified

if isfield(param,'omegaj') 
    if ~isequal(size(param.omegaj),[param.J 1])
        warning('%s.m: omegaj does not have the right size Jx1.',mfilename);
    end
end

if isfield(param,'Lj') && param.mobility==false 
    if ~isequal(size(param.Lj),[param.J 1])
        warning('%s.m: Lj does not have the right size Jx1.',mfilename);
    end
end

if isfield(param,'Zjn') 
    if ~isequal(size(param.Zjn),[param.J param.N])
        warning('%s.m: Zjn does not have the right size JxN.',mfilename);
    end
end

if isfield(param,'Hj')
    if ~isequal(size(param.Hj),[param.J 1])
        warning('%s.m: Hj does not have the right size Jx1.',mfilename);
    end
end

if isfield(param,'hj')
    if ~isequal(size(param.hj),[param.J 1])
        warning('%s.m: hj does not have the right size Jx1.',mfilename);
    end
end

end

