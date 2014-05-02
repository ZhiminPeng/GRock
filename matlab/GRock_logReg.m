%% GRock algorithm applied to solve sparse logistic regression
function [x, hist] = GRock_logReg(A, b, lambda, opts)
% function [x, hist] = GRock_LASSO(A, b, lambda, opts)
%
% Solves the minimization problem
%
%  min_x 0.5 * ||A*x - b||^2 + lambda ||x||_1
%
% by the so-called GRock method from Zhimin, Ming and Wotao.
%
% Input:
%   b:      Given data to be approximated
%   lambda: Regularization parameter > 0.
%   A:      Data matrix or measurement matrix
%
% Optional input:
%   opt.maxIter: maximal number of iteration (default: 300)
%   opt.print:   switch to turn on some output (default: false)
%   opt.tol_err: tolerance for the error norm(u-udagger)
%   opt.tol_obj: tolerance on the distance to the optimal objective
%
%
% Output:
%  x:            calulated minimizer.
%  hist.obj:     history of the objective values.
%  hist.rel_err: history of the relative error (if true solution (xs) is
%                 given)
%  hist.time:    history of the accumulated elapsed time in each iteration.
%
% References:
% - Z. Peng, M. Yan, and W. Yin. Parallel and Distributed Sparse Optimization,
%   IEEE Asilomar Conference on Signals, Systems, and Computers, 2013
%
%
% Copyright(c) Zhimin Peng 2014
% Email: zhimin.peng@math.ucla.edu
%
%TODO:
%
%
%
%
%--------------------------------------------------------------------------

%% initialize parameters
t_start = tic;
[m, n] = size(A);
[maxIter, obj_tol, rel_tol, print, xs, P, sigma, mode] = get_opts;
x            = zeros(n,1); % initialize x
hist.obj_err = zeros(maxIter, 1); % initialize history objective values
hist.rel_err = zeros(maxIter, 1); % initialize history relative error
hist.time    = zeros(maxIter, 1); % initialize computation time
hist.epoch   = zeros(maxIter, 1);
opt_obj      = lambda*norm(xs,1)+0.5*norm(A*xs - b)^2; % optimal obj
nrm_xs       = norm(xs); % ||xs||_2
if(nrm_xs==0)
    nrm_xs=1; opt_obj=0;
end;
nrm_A        = zeros(n,1); % calcuate the norm of each column of A
for i = 1:n
    nrm_A(i) = norm(A(:,i))^2;
end
shk          = lambda./nrm_A; % shrinkage amount for each coordinate
g            = -A'*b; % initial gradient

disp('GRock is solving LASSO!');


%% main loop
for iter = 1:maxIter
    x_old    = x;
    Ax_b     = A*x_old - b;
    obj_old  = lambda*norm(x_old, 1) + 0.5*(Ax_b'*Ax_b);
    
    %% calcuate the potential updates
    beta     = x - g./nrm_A;
    x_all    = sign(beta).*max(abs(beta) - shk, 0);
    d        = x_all - x_old;
    [~, pos] = sort(abs(d), 'descend');
    x_tmp    = x_old;
    %keyboard;
    upd_idx  = pos(1:P);
    x_tmp(upd_idx) = x_old(upd_idx) + d(upd_idx);
    Ax_b     = A*x_tmp - b;
    %Ax_b     = Ax_b + A(:, upd_idx)*d(upd_idx);
    obj_new  = lambda*norm(x_tmp,1) + 0.5*(Ax_b'*Ax_b);
    delta    = -d(upd_idx)'*d(upd_idx);
    if(strcmp(mode,'dynamic'))
            %% dynamic update P
            while(obj_new - obj_old>= sigma * delta)
                x_tmp    = x_old;
                P        = min(max(floor(P*0.8), 1), n);
                upd_idx  = pos(1:P);
                x_tmp(upd_idx) = x_old(upd_idx) + d(upd_idx);
                Ax_b     = A*x_tmp - b;
                %Ax_b     = Ax_b + A(:, upd_idx)*d(upd_idx);
                obj_new  = lambda*norm(x_tmp,1) + 0.5*(Ax_b'*Ax_b);
                delta    = -d(upd_idx)'*d(upd_idx);
                if(P==1)
                    break;
                end
            end
            P = min(max(floor(P * 1.2), floor(0.05*n)),n);
    end
    %% prepare for the next update
    x = x_tmp;
    g = g + A'*(A(:,upd_idx)*d(upd_idx));
    
    %% stopping criterion
    if(abs(obj_new -obj_old)<obj_tol * obj_old ...
            && norm(x-x_old)<rel_tol*norm(x_old))
        break;
    end
    t_end = toc(t_start);
    
    %% history record
    hist.time(iter) = t_end;
    hist.obj_err(iter) = obj_new - opt_obj;
    hist.rel_err(iter) = norm(x - xs)/nrm_xs;
    hist.epoch(iter) = iter;
end

%% nested functions
    function [maxIter, obj_tol, rel_tol, print, xs, P, sigma, mode] = get_opts
        % get or set options
        obj_tol = 1e-20;
        rel_tol = 1e-20;
        maxIter = 9999;
        print   = 0;
        P       = floor(0.1*n);
        sigma   = 0.07;
        xs      = zeros(n,1);
        mode    = 'dynamic';
        if isfield(opts,'maxIter'); maxIter = opts.maxIter; end
        if isfield(opts,'print'); print = opts.print; end
        if isfield(opts,'obj_tol'); obj_tol = opts.obj_tol; end
        if isfield(opts,'rel_tol'); rel_tol = opts.rel_tol; end
        if isfield(opts,'xs'); xs = opts.xs; end
        if isfield(opts,'P'); P = opts.P; end
        if isfield(opts,'sigma'); sigma = opts.sigma; end
        if isfield(opts,'mode'); mode = opts.mode; end
    end

end
